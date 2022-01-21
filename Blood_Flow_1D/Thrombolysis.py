#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Permeability Simulator

Script to run a simulation of the 1D BF model with changing permeability.
Input argument is the patient folder with a folder for input files and a folder for modelling files
The input folder contains patient data.
The modelling file folder contains files for the models such as the parameters and the surface mesh.

Usage:
  tavernaBloodFlow.py <patient_folder>
  tavernaBloodFlow.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

import contextlib
import math
import os
from datetime import datetime

import numpy as np
import vtk
from tqdm import tqdm
import copy

from Blood_Flow_1D import Vessel, Node, GeneralFunctions, Collaterals, Patient, \
    Results, docopt, transcript, ContrastModel, CollateralsSimulation


def permeability_simulation(patient_folder):
    """
    Run the 1D model with data from patient_folder

    Parameters
    ----------
    patient_folder : str
        Patient folder location

    Returns
    -------

    """
    start_time = datetime.now()
    transcript.start(patient_folder + 'logfile.log')

    # Load files
    patient = Patient.Patient(patient_folder)
    patient.LoadBFSimFiles()
    patient.LoadPositions()

    # todo replace, temp solution to handle different velocity values inside vessels
    splitclotvessel(patient)

    time = 0.0
    dt = 0.001
    end_time = 5.0
    maxiter = math.ceil(end_time / dt) + 2
    contrastFiles = 100  # number of output files for animations
    iterstep = math.floor(maxiter / contrastFiles)

    # overwrite Clots.txt
    for clot in patient.Topology.Clots:
        clot[1] = 1e-5  # units mm^2, uniform permeability
        clot[2] = 0.2  # porosity

    patient.Initiate1DSteadyStateModel()

    # healthy state at t==-1
    if patient.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
            patient.Folders.ModellingFolder + "PAmapping.csv"):
        Collaterals.ImportPAnodes(patient)
        # if PAnode are present, run the model with the autoregulation method.
        CollateralsSimulation.collateral_simulation(patient, clot_active=False)
    else:
        patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=False, PressureInlets=True,
                                      FlowRateOutlets=False, coarseCollaterals=False,
                                      frictionconstant=patient.ModelParameters["FRICTION_C"])

    mtx, Initialsolution, nodesetlist = contrast_matrix(patient, dt)
    for node in patient.Topology.InletNodes:
        node_number = nodesetlist.index(node[0])
        # Initialsolution[node_number] = 10 * np.exp(-1 * (time - 1) * (time - 1) / (0.1))
        Initialsolution[node_number] = 1  # constant injection
    for node, contrast in zip(nodesetlist, Initialsolution):
        node.Contrast = contrast
    for node in patient.Topology.BifurcationNodes:
        node.Contrast = node.ContrastNode.Contrast

    patient.Results.Time.append([-1])
    patient.Results.TimePoints.append(Results.TimePoint(-1))
    patient.Results.TimePoints[-1].Flow = [node.FlowRate for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].Pressure = [node.Pressure for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].Radius = [node.Radius for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].Velocity = [node.Velocity for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].contrast = [node.Contrast for node in patient.Topology.Nodes]

    # per time step, list of clots (usually one)
    patient.Results.PressureDrop = [[]]
    patient.Results.Permeability = [[]]
    patient.Results.ClotFlow = [[]]
    patient.Results.ClotVelocity = [[]]
    for clot in patient.Topology.Clots:
        proximal_node = min(clot[0], key=lambda item: item.Number)
        distal_node = max(clot[0], key=lambda item: item.Number)
        dp = proximal_node.Pressure - distal_node.Pressure
        patient.Results.PressureDrop[0].append(dp)
        patient.Results.Permeability[0].append(clot[1])
        patient.Results.ClotFlow[0].append(proximal_node.FlowRate)
        patient.Results.ClotVelocity[0].append(proximal_node.Velocity)

    # Stroke simulation
    for nt in tqdm(range(0, maxiter)):
        time = nt * dt
        # print(f"Simulation time:{time}")
        # 1D BF model
        with contextlib.redirect_stdout(None):
            if patient.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
                    patient.Folders.ModellingFolder + "PAmapping.csv"):
                # if PAnode are present, run the model with the autoregulation method.
                CollateralsSimulation.collateral_simulation(patient, clot_active=True)
            else:
                patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=True, PressureInlets=True,
                                              FlowRateOutlets=False, coarseCollaterals=False,
                                              frictionconstant=patient.ModelParameters["FRICTION_C"])
            # update contrast matrix
            mtx, _, _ = contrast_matrix(patient, dt)
            # update solution (overwrite boundary or other nodes)
            for node in patient.Topology.InletNodes:
                node_number = nodesetlist.index(node[0])
                # Initialsolution[node_number] = 10 * np.exp(-1 * (time - 1) * (time - 1) / (0.1))  # sharp Gaussian pulse
                Initialsolution[node_number] = 1  # constant injection

            # advance one step
            Initialsolution = mtx.dot(Initialsolution)
            # update contrast of nodes
            for node, contrast in zip(nodesetlist, Initialsolution):
                node.Contrast = contrast
            for node in patient.Topology.BifurcationNodes:
                node.Contrast = node.ContrastNode.Contrast

        # Save results
        if nt % iterstep == 0:
            patient.Results.Time[-1].append(time)
            patient.Results.TimePoints.append(Results.TimePoint(time))
            patient.Results.TimePoints[-1].Flow = [node.FlowRate for node in patient.Topology.Nodes]
            patient.Results.TimePoints[-1].Pressure = [node.Pressure for node in patient.Topology.Nodes]
            patient.Results.TimePoints[-1].Radius = [node.Radius for node in patient.Topology.Nodes]
            patient.Results.TimePoints[-1].Velocity = [node.Velocity for node in patient.Topology.Nodes]
            patient.Results.TimePoints[-1].contrast = [node.Contrast for node in patient.Topology.Nodes]

            patient.Results.PressureDrop.append([])
            patient.Results.Permeability.append([])
            patient.Results.ClotFlow.append([])
            patient.Results.ClotVelocity.append([])
            for clot in patient.Topology.Clots:
                proximal_node = min(clot[0], key=lambda item: item.Number)
                distal_node = max(clot[0], key=lambda item: item.Number)
                dp = proximal_node.Pressure - distal_node.Pressure
                patient.Results.PressureDrop[-1].append(dp)
                patient.Results.Permeability[-1].append(clot[1])
                patient.Results.ClotFlow[-1].append(proximal_node.FlowRate)
                patient.Results.ClotVelocity[-1].append(proximal_node.Velocity)
                proximal_tPA = proximal_node.Contrast
                distal_tPA = distal_node.Contrast

        for clot in patient.Topology.Clots:
            proximal_node = min(clot[0], key=lambda item: item.Number)
            distal_node = max(clot[0], key=lambda item: item.Number)
            dp = proximal_node.Pressure - distal_node.Pressure
            proximal_tPA = proximal_node.Contrast
            distal_tPA = distal_node.Contrast

            # change permeability
            k = 10
            clot[1] += k*clot[1]*(proximal_tPA+distal_tPA)*dt # units mm^2, uniform permeability
            clot[1] = min(1e-1, clot[1])
            clot[2] = min(1, 0.2*(clot[1] / 1e-5))  # porosity

            clot[1] += 1e-7  # units mm^2, uniform permeability
            clot[2] = 0.2  # porosity

    # output results
    write_time_series(patient)  # open ResultsPermeableClots.pvd in Paraview

    filename = patient.Folders.ModellingFolder + "Clot_Permeability.csv"
    with open(filename, "w") as f:
        f.write("Time [s],Pressure drop [pa],Permeability [mm^2],Flow rate [mL/s],Velocity before clot [m/s]\n")
        for t, dp, permeability, flow_rate, velocity in zip(patient.Results.Time[-1], patient.Results.PressureDrop,
                                                            patient.Results.Permeability,
                                                            patient.Results.ClotFlow, patient.Results.ClotVelocity):
            f.write("%f,\"%s\",\"%s\",\"%s\",\"%s\"\n" % (t, ",".join([str(i) for i in dp]),
                                                          ",".join([str(i) for i in permeability]),
                                                          ",".join([str(i) for i in flow_rate]),
                                                          ",".join([str(i) for i in velocity])))

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    transcript.stop()


def contrast_matrix(patient, dt):
    for vessel in patient.Topology.Vessels:
        velocity = np.mean([node.Velocity for node in vessel.Nodes]) * 1e3  # mm/s
        if velocity < 0:
            vessel.Nodes.reverse()  # if velocity is negative, reverse the order of the nodes.
        vessel.Velocity = abs(velocity)
        vessel.Nodes[0].FlowRate = abs(vessel.Nodes[0].FlowRate)
        vessel.Nodes[-1].FlowRate = abs(vessel.Nodes[-1].FlowRate)
        vessel.Nodes[0].Start = -1
        vessel.Nodes[-1].Start = 1

    mtx, inputvector, nodesetlist = ContrastModel.contrastMatrixUpwind(patient, dt)

    Initialsolution = inputvector

    BoundaryUpdateMat = ContrastModel.boundaryUpdateMatrix(patient, len(nodesetlist))
    mtx = BoundaryUpdateMat.dot(mtx)  # update matrix to include bifucation conditions

    return mtx, Initialsolution, nodesetlist


def splitclotvessel(patient):
    patient.Topology.NodeVesselDict()
    clotvessels = set()
    clotnodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    for clotnode in clotnodes:
        vessel = patient.Topology.NodeDict[clotnode]
        clotvessels.add(vessel)
    clotvessels = list(clotvessels)

    # temp solution
    clot_vessel_inside = Vessel.Vessel()
    clot_vessel_inside.SetName("Inside Clot")
    clot_vessel_inside.SetID(patient.Topology.Vessels[-1].ID+1)
    clot_vessel_inside.Nodes = sorted(patient.Topology.Clots[0][0], key=lambda item: item.Number)  # needs sort
    patient.Topology.Vessels.append(clot_vessel_inside)
    # check if clot at start or end of vessel
    # split in two if that is the case.

    proximal_node = min(clot_vessel_inside.Nodes, key=lambda item: item.Number)
    distal_node = max(clot_vessel_inside.Nodes, key=lambda item: item.Number)

    if clotvessels[0].Nodes[0] == proximal_node:
        # clot at start
        # split in two
        # remove clot nodes from vessel
        clotvessels[0].Nodes = [node for node in clotvessels[0].Nodes if node.Number > distal_node.Number]
        newbif1 = Node.Node()
        newbif1.Number = len(patient.Topology.Nodes)
        newbif1.LengthAlongVessel = distal_node.LengthAlongVessel
        newbif1.Radius = distal_node.Radius
        newbif1.YoungsModules = distal_node.YoungsModules
        newbif1.Thickness = distal_node.Thickness
        newbif1.RefRadius = distal_node.RefRadius
        newbif1.Type = 1

        new_end = Node.Node()
        new_end.Number = len(patient.Topology.Nodes) + 1
        new_end.LengthAlongVessel = distal_node.LengthAlongVessel
        new_end.Radius = distal_node.Radius
        new_end.YoungsModules = distal_node.YoungsModules
        new_end.Thickness = distal_node.Thickness
        new_end.RefRadius = distal_node.RefRadius
        new_end.Velocity = distal_node.Velocity
        new_end.Position = distal_node.Position

        distal_node.RemoveConnection(clotvessels[0].Nodes[0])
        clotvessels[0].Nodes[0].RemoveConnection(distal_node)

        newbif1.AddConnection(new_end)
        newbif1.AddConnection(distal_node)
        new_end.AddConnection(newbif1)
        distal_node.AddConnection(newbif1)

        new_end.AddConnection(clotvessels[0].Nodes[0])
        clotvessels[0].Nodes[0].AddConnection(new_end)

        clotvessels[0].Nodes.append(new_end)

        patient.Topology.BifurcationNodes.append(newbif1)
        patient.Topology.Nodes.append(newbif1)
        patient.Topology.Nodes.append(new_end)

        clotvessels[0].Velocity = np.mean([node.Velocity for node in clotvessels[0].Nodes])
        clot_vessel_inside.Velocity = np.mean([node.Velocity for node in clot_vessel_inside.Nodes])

        start_length = copy.deepcopy(clotvessels[0].Nodes[0].LengthAlongVessel)
        for node in clotvessels[0].Nodes:
            node.LengthAlongVessel -= start_length
        start_length = copy.deepcopy(clot_vessel_inside.Nodes[0].LengthAlongVessel)
        for node in clot_vessel_inside.Nodes:
            node.LengthAlongVessel -= start_length

        clotvessels[0].Length = clotvessels[0].Nodes[-1].LengthAlongVessel
        clot_vessel_inside.Length = clot_vessel_inside.Nodes[-1].LengthAlongVessel
        clot_vessel_inside.GridSize = abs(clot_vessel_inside.Nodes[0].LengthAlongVessel-clot_vessel_inside.Nodes[1].LengthAlongVessel)
    elif clotvessels[0].Nodes[-1] == distal_node:
        # clot at end
        # split in two
        # remove clot nodes from vessel
        clotvessels[0].Nodes = [node for node in clotvessels[0].Nodes if node.Number < proximal_node.Number]
        newbif1 = Node.Node()
        newbif1.Number = len(patient.Topology.Nodes)
        newbif1.LengthAlongVessel = proximal_node.LengthAlongVessel
        newbif1.Radius = proximal_node.Radius
        newbif1.YoungsModules = proximal_node.YoungsModules
        newbif1.Thickness = proximal_node.Thickness
        newbif1.RefRadius = proximal_node.RefRadius
        newbif1.Type = 1

        new_end = Node.Node()
        new_end.Number = len(patient.Topology.Nodes) + 1
        new_end.LengthAlongVessel = proximal_node.LengthAlongVessel
        new_end.Radius = proximal_node.Radius
        new_end.YoungsModules = proximal_node.YoungsModules
        new_end.Thickness = proximal_node.Thickness
        new_end.RefRadius = proximal_node.RefRadius
        new_end.Velocity = proximal_node.Velocity
        new_end.Position = proximal_node.Position

        proximal_node.RemoveConnection(clotvessels[0].Nodes[-1])
        clotvessels[0].Nodes[-1].RemoveConnection(proximal_node)

        newbif1.AddConnection(new_end)
        newbif1.AddConnection(proximal_node)
        new_end.AddConnection(newbif1)
        proximal_node.AddConnection(newbif1)

        new_end.AddConnection(clotvessels[0].Nodes[-1])
        clotvessels[0].Nodes[-1].AddConnection(new_end)

        clotvessels[0].Nodes.append(new_end)

        patient.Topology.BifurcationNodes.append(newbif1)
        patient.Topology.Nodes.append(newbif1)
        patient.Topology.Nodes.append(new_end)

        clotvessels[0].Velocity = np.mean([node.Velocity for node in clotvessels[0].Nodes])
        clot_vessel_inside.Velocity = np.mean([node.Velocity for node in clot_vessel_inside.Nodes])

        start_length = copy.deepcopy(clotvessels[0].Nodes[0].LengthAlongVessel)
        for node in clotvessels[0].Nodes:
            node.LengthAlongVessel -= start_length
        start_length = copy.deepcopy(clot_vessel_inside.Nodes[0].LengthAlongVessel)
        for node in clot_vessel_inside.Nodes:
            node.LengthAlongVessel -= start_length

        clotvessels[0].Length = clotvessels[0].Nodes[-1].LengthAlongVessel
        clot_vessel_inside.Length = clot_vessel_inside.Nodes[-1].LengthAlongVessel
        clot_vessel_inside.GridSize = abs(clot_vessel_inside.Nodes[0].LengthAlongVessel-clot_vessel_inside.Nodes[1].LengthAlongVessel)
    else:
        # split in three
        clot_vessel_After = Vessel.Vessel()
        clot_vessel_After.SetName("After Clot")
        clot_vessel_After.Nodes = [node for node in clotvessels[0].Nodes if node.Number > distal_node.Number]
        clotvessels[0].Nodes = [node for node in clotvessels[0].Nodes if node.Number < proximal_node.Number]
        clot_vessel_After.SetID(patient.Topology.Vessels[-1].ID + 1)
        patient.Topology.Vessels.append(clot_vessel_After)

        newbif1 = Node.Node()
        newbif1.Number = len(patient.Topology.Nodes)
        newbif1.LengthAlongVessel = proximal_node.LengthAlongVessel
        newbif1.Radius = proximal_node.Radius
        newbif1.YoungsModules = proximal_node.YoungsModules
        newbif1.Thickness = proximal_node.Thickness
        newbif1.RefRadius = proximal_node.RefRadius
        newbif1.Type = 1

        newbif2 = Node.Node()
        newbif2.Number = len(patient.Topology.Nodes) + 1
        newbif2.LengthAlongVessel = distal_node.LengthAlongVessel
        newbif2.Radius = distal_node.Radius
        newbif2.YoungsModules = distal_node.YoungsModules
        newbif2.Thickness = distal_node.Thickness
        newbif2.RefRadius = distal_node.RefRadius
        newbif2.Type = 1

        new_end1 = Node.Node()
        new_end1.Number = len(patient.Topology.Nodes) + 2
        new_end1.LengthAlongVessel = proximal_node.LengthAlongVessel
        new_end1.Radius = proximal_node.Radius
        new_end1.Position = proximal_node.Position
        new_end1.YoungsModules = proximal_node.YoungsModules
        new_end1.Thickness = proximal_node.Thickness
        new_end1.RefRadius = proximal_node.RefRadius

        new_end2 = Node.Node()
        new_end2.Number = len(patient.Topology.Nodes) + 3
        new_end2.LengthAlongVessel = distal_node.LengthAlongVessel
        new_end2.Radius = distal_node.Radius
        new_end2.Position = distal_node.Position
        new_end2.YoungsModules = distal_node.YoungsModules
        new_end2.Thickness = distal_node.Thickness
        new_end2.RefRadius = distal_node.RefRadius

        proximal_node.RemoveConnection(clotvessels[0].Nodes[-1])
        clotvessels[0].Nodes[-1].RemoveConnection(proximal_node)
        newbif1.AddConnection(new_end1)
        newbif1.AddConnection(proximal_node)
        new_end1.AddConnection(newbif1)
        proximal_node.AddConnection(newbif1)

        new_end1.AddConnection(clotvessels[0].Nodes[-1])
        clotvessels[0].Nodes[-1].AddConnection(new_end1)

        distal_node.RemoveConnection(clot_vessel_After.Nodes[0])
        clot_vessel_After.Nodes[0].RemoveConnection(distal_node)
        newbif2.AddConnection(new_end2)
        newbif2.AddConnection(distal_node)
        new_end2.AddConnection(newbif2)
        distal_node.AddConnection(newbif2)
        new_end2.AddConnection(clot_vessel_After.Nodes[0])
        clot_vessel_After.Nodes[0].AddConnection(new_end2)

        clotvessels[0].Nodes.append(new_end1)
        clot_vessel_After.Nodes.insert(0, new_end2)

        patient.Topology.BifurcationNodes.append(newbif1)
        patient.Topology.Nodes.append(newbif1)
        patient.Topology.Nodes.append(new_end1)
        patient.Topology.BifurcationNodes.append(newbif2)
        patient.Topology.Nodes.append(newbif2)
        patient.Topology.Nodes.append(new_end2)

        clotvessels[0].Velocity = np.mean([node.Velocity for node in clotvessels[0].Nodes])
        clot_vessel_inside.Velocity = np.mean([node.Velocity for node in clot_vessel_inside.Nodes])
        clot_vessel_After.Velocity = np.mean([node.Velocity for node in clot_vessel_After.Nodes])

        start_length = copy.deepcopy(clotvessels[0].Nodes[0].LengthAlongVessel)
        for node in clotvessels[0].Nodes:
            node.LengthAlongVessel -= start_length
        start_length = copy.deepcopy(clot_vessel_inside.Nodes[0].LengthAlongVessel)
        for node in clot_vessel_inside.Nodes:
            node.LengthAlongVessel -= start_length
        start_length = copy.deepcopy(clot_vessel_After.Nodes[0].LengthAlongVessel)
        for node in clot_vessel_After.Nodes:
            node.LengthAlongVessel -= start_length

        clotvessels[0].Length = clotvessels[0].Nodes[-1].LengthAlongVessel
        clot_vessel_inside.Length = clot_vessel_inside.Nodes[-1].LengthAlongVessel
        clot_vessel_inside.GridSize = abs(clot_vessel_inside.Nodes[0].LengthAlongVessel-clot_vessel_inside.Nodes[1].LengthAlongVessel)
        clot_vessel_After.GridSize = abs(clot_vessel_After.Nodes[0].LengthAlongVessel-clot_vessel_After.Nodes[1].LengthAlongVessel)
        clot_vessel_After.Length = clot_vessel_After.Nodes[-1].LengthAlongVessel
    patient.Topology.NumberNodes()

def write_time_series(patient):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(patient.Folders.ModellingFolder + "Topology.vtp")
    reader.Update()
    data = reader.GetOutput()
    output_folder = patient.Folders.ModellingFolder + "TimeSeriesPermeableClots/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for time_index, time_point in enumerate(patient.Results.TimePoints):
        flow = vtk.vtkFloatArray()
        flow.SetNumberOfComponents(1)
        flow.SetName("Volume Flow Rate [mL/s]")

        velocity = vtk.vtkFloatArray()
        velocity.SetNumberOfComponents(1)
        velocity.SetName("Velocity [mm/s]")

        pressure = vtk.vtkFloatArray()
        pressure.SetNumberOfComponents(1)
        pressure.SetName("Pressure [pa]")

        radius = vtk.vtkFloatArray()
        radius.SetNumberOfComponents(1)
        radius.SetName("Radius [mm]")

        contrast = vtk.vtkFloatArray()
        contrast.SetNumberOfComponents(1)
        contrast.SetName("Contrast []")

        for node_velocity, node_flow, node_radius, node_pressure, node_contrast in \
                zip(time_point.Velocity, time_point.Flow, time_point.Radius, time_point.Pressure, time_point.contrast):
            flow.InsertNextValue(node_flow)
            velocity.InsertNextValue(node_velocity)
            pressure.InsertNextValue(node_pressure)
            radius.InsertNextValue(node_radius)
            contrast.InsertNextValue(node_contrast)

        writer = vtk.vtkXMLPolyDataWriter()
        filename = output_folder + "Blood_flow" + str(time_index) + ".vtp"
        writer.SetFileName(filename)
        data.GetPointData().AddArray(flow)
        data.GetPointData().AddArray(velocity)
        data.GetPointData().AddArray(pressure)
        data.GetPointData().AddArray(radius)
        data.GetPointData().AddArray(contrast)
        writer.SetInputData(data)
        writer.Write()

    with open(patient.Folders.ModellingFolder + "ResultsPermeableClots.pvd", "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write(
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
        f.write("<Collection>\n")
        for time_index, time_point in enumerate(patient.Results.TimePoints):
            name = "TimeSeriesPermeableClots/Blood_flow" + str(time_index) + ".vtp"
            time = str(time_point.WT)
            f.write("<DataSet timestep=\"" + time + "\" group=\"\" part=\"0\" file=\"" + name + "\"/>\n")
        f.write("</Collection>\n")
        f.write("</VTKFile>")


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    input_patient_folder = arguments["<patient_folder>"]
    # input_patient_folder = "/home/raymond/Desktop/Thrombo_test/"
    permeability_simulation(input_patient_folder)
