#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Simulate various permeable clots and colleterals

Usage:
  ContrastModel.py <number>
  ContrastModel.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
import contextlib
import csv
import os
import shutil
import sys
from datetime import datetime
from distutils.dir_util import copy_tree

import matplotlib.pyplot as plt
import numpy as np
import vtk
from tqdm import tqdm
import copy

from Blood_Flow_1D import transcript, Patient, ContrastModel, Vessel, Collaterals, GeneralFunctions, \
    Results, Metadata, GenerateBloodflowFiles, CollateralsSimulation, Node, docopt

auto_regulation = True
# auto_regulation = False

def permeability_simulation(patient_folder, iterations=100, contrast_sim_time=30):
    """
    Run the 1D model with data from patient_folder

    Parameters
    ----------
    iterations
    contrast_sim_time : simulation time
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

    for clot in patient.Topology.Clots:
        clot[1] = 0  # units mm^2, uniform permeability
        # clot[2] = 0.06  # porosity
        clot[2] = 0.20  # porosity

    # get clot vessels
    patient.Topology.NodeVesselDict()
    clotvessels = set()
    clotvesselsbifucations = set()
    clotnodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    for clotnode in clotnodes:
        vessel = patient.Topology.NodeDict[clotnode]
        clotvessels.add(vessel)
        NodeB1 = [vessel.GetProximalBifurcation() for vessel in clotvessels][0]
        NodeB2 = [vessel.GetDistalBifurcation() for vessel in clotvessels][0]
        clotvesselsbifucations.add((NodeB1, NodeB2))

    splitclotvessel(patient)
    patient.Topology.WriteNodesCSV(patient.Folders.ModellingFolder + "Nodes.csv")
    patient.Topology.WriteVesselCSV(patient.Folders.ModellingFolder + "Vessels.csv")

    coarse_collaterals = False  # set to true for coarse collaterals
    # patient.Topology.coarse_collaterals_effective_radius = 0.2
    # patient.Topology.coarse_collaterals_number = 1
    # patient.Topology.coarse_collaterals_After_WKNodes = False

    patient.Initiate1DSteadyStateModel()

    # always scale BC?
    patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=False, PressureInlets=True,
                                  coarseCollaterals=False, frictionconstant=patient.ModelParameters["FRICTION_C"],
                                  scale_resistance=True)

    if patient.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
            patient.Folders.ModellingFolder + "PAmapping.csv") and auto_regulation:
        Collaterals.ImportPAnodes(patient)
        # if PAnode are present, run the model with the autoregulation method.
        CollateralsSimulation.collateral_simulation(patient, clot_active=False)
    else:
        patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=False, PressureInlets=True,
                                      FlowRateOutlets=False, coarseCollaterals=coarse_collaterals,
                                      frictionconstant=patient.ModelParameters["FRICTION_C"],
                                      scale_resistance=False)

    for vessel in patient.Topology.Vessels:
        vessel.Velocity = np.mean([node.Velocity for node in vessel.Nodes]) * 1e3
    # ContrastModel.contrastSimulationUpwind(patient, simulationtime=contrast_sim_time, hdf5DataFile=None,
    #                                        number_output_files=100, inlet_section=False, clot_contrast_disable=True)
    for node in patient.Topology.Nodes:
        node.peak_time = 0

    # save R MCA flow rate over the region
    # first get all outlet nodes for that region
    brain_outlet_nodes = [[node for node in patient.Topology.OutletNodes if node.MajorVesselID == i] for i in range(2, 10)]
    flow_rate_outlet_nodes = [sum([node.FlowRate for node in nodes]) for nodes in brain_outlet_nodes]
    print(flow_rate_outlet_nodes)
    # todo activate for results per iteration
    # output_folder = patient.Folders.ModellingFolder + "mean_results/"
    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)
    # patient.Results1DSteadyStateModel()
    # file_name = "ResultsPerVessel_"+"-1"+".csv"
    # patient.Results.ExportMeanResults(folder=output_folder, file=file_name)

    # healthy flowrate per outlet
    for nodes in brain_outlet_nodes:
        for node in nodes:
            node.healthy_flow = node.FlowRate

    # initial state at t==-1 (healthy)
    patient.Results.Time.append([-1])
    patient.Results.TimePoints.append(Results.TimePoint(-1))
    patient.Results.TimePoints[-1].Flow = [node.FlowRate for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].Pressure = [node.Pressure for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].Radius = [node.Radius for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].Velocity = [node.Velocity for node in patient.Topology.Nodes]
    patient.Results.TimePoints[-1].peak_time = [node.peak_time for node in patient.Topology.Nodes]

    patient.Results.PressureDrop = [[]]  # per time step, list of clots (usually one)
    patient.Results.Permeability = [[]]
    patient.Results.ClotFlow = [[]]
    patient.Results.ClotVelocity = [[]]
    patient.Results.TimeDelay = [[]]
    patient.Results.BifTimeDelay = [[]]
    patient.Results.CerebralFlowRates = []
    patient.Results.CerebralFlowRates.append(flow_rate_outlet_nodes)
    patient.Results.Infarctvolumes = []
    patient.Results.Infarctvolumes.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for clot in patient.Topology.Clots:
        node1 = min(clot[0], key=lambda item: item.Number)
        node2 = max(clot[0], key=lambda item: item.Number)
        dp = node1.Pressure - node2.Pressure
        time_delay = node2.peak_time - node1.peak_time
        patient.Results.PressureDrop[0].append(dp)
        patient.Results.Permeability[0].append(clot[1])
        patient.Results.ClotFlow[0].append(node1.FlowRate)
        patient.Results.ClotVelocity[0].append(node1.Velocity)
        patient.Results.TimeDelay[0].append(time_delay)

    for clotvessel in clotvesselsbifucations:
        NodeB1 = clotvessel[0]
        NodeB2 = clotvessel[1]
        time_delay_bifurcation = NodeB2.peak_time - NodeB1.peak_time
        patient.Results.BifTimeDelay[0].append(time_delay_bifurcation)

    # simulation
    for i in tqdm(range(0, iterations + 2)):
        time = i

        # 1D BF model
        with contextlib.redirect_stdout(None):
            # We should be able to always use Run1DSteadyStateModel here
            # patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=True, PressureInlets=True,
            #                               FlowRateOutlets=False, coarseCollaterals=coarse_collaterals,
            #                               frictionconstant=patient.ModelParameters["FRICTION_C"])
            if patient.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
                    patient.Folders.ModellingFolder + "PAmapping.csv") and auto_regulation:
                # if PAnode are present, run the model with the autoregulation method.
                CollateralsSimulation.collateral_simulation(patient, clot_active=True)
            else:
                patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=True, PressureInlets=True,
                                              FlowRateOutlets=False, coarseCollaterals=coarse_collaterals,
                                              frictionconstant=patient.ModelParameters["FRICTION_C"],
                                              scale_resistance=False)

            for vessel in patient.Topology.Vessels:
                vessel.Velocity = np.mean([node.Velocity for node in vessel.Nodes]) * 1e3
            # ContrastModel.contrastSimulationUpwind(patient, simulationtime=contrast_sim_time, hdf5DataFile=None,
            #                                        number_output_files=100, inlet_section=False, clot_contrast_disable=True)
        patient.Results1DSteadyStateModel()
        # todo activate for results per iteration
        # file_name = "ResultsPerVessel_" + str(time) + ".csv"
        # patient.Results.ExportMeanResults(folder=output_folder, file=file_name)
        # Save results
        patient.Results.Time[-1].append(time)
        patient.Results.TimePoints.append(Results.TimePoint(time))
        patient.Results.TimePoints[-1].Flow = [node.FlowRate for node in patient.Topology.Nodes]
        patient.Results.TimePoints[-1].Pressure = [node.Pressure for node in patient.Topology.Nodes]
        patient.Results.TimePoints[-1].Radius = [node.Radius for node in patient.Topology.Nodes]
        patient.Results.TimePoints[-1].Velocity = [node.Velocity for node in patient.Topology.Nodes]
        patient.Results.TimePoints[-1].peak_time = [node.peak_time for node in patient.Topology.Nodes]

        patient.Results.PressureDrop.append([])
        patient.Results.Permeability.append([])
        patient.Results.ClotFlow.append([])
        patient.Results.ClotVelocity.append([])
        patient.Results.TimeDelay.append([])
        patient.Results.BifTimeDelay.append([])

        flow_rate_outlet_nodes = [sum([node.FlowRate for node in nodes]) for nodes in brain_outlet_nodes]
        patient.Results.CerebralFlowRates.append(flow_rate_outlet_nodes)

        for clot in patient.Topology.Clots:
            node1 = min(clot[0], key=lambda item: item.Number)
            node2 = max(clot[0], key=lambda item: item.Number)
            dp = node1.Pressure - node2.Pressure
            time_delay = node2.peak_time - node1.peak_time
            patient.Results.PressureDrop[-1].append(dp)
            patient.Results.Permeability[-1].append(clot[1])
            patient.Results.ClotFlow[-1].append(node1.FlowRate)
            patient.Results.ClotVelocity[-1].append(node1.Velocity)
            patient.Results.TimeDelay[-1].append(time_delay)
        for clot_vessel in clotvesselsbifucations:
            NodeB1 = clot_vessel[0]
            NodeB2 = clot_vessel[1]
            time_delay_bifurcation = NodeB2.peak_time - NodeB1.peak_time
            patient.Results.BifTimeDelay[-1].append(time_delay_bifurcation)

        # change permeability
        for clot in patient.Topology.Clots:
            # clot[1] += 2e-5  # units mm^2, uniform permeability
            # clot[1] = 1e-7 + 5e-7*i  # units mm^2, uniform permeability
            clot[1] = 10**-(7-7*i/iterations)
            # clot[2] = 0.06  # porosity
            clot[2] = 0.20  # porosity

        # infarct volume estimate based on percentage drop
        # total brain volume
        print(1)
        brain_volume = 1390409  # mm^3
        brain_volume_ml = brain_volume/1000
        # average volume per outlet
        number_nodes = sum([len(nodes) for nodes in brain_outlet_nodes])
        average_volume_outlet = brain_volume_ml/number_nodes
        change_flowrate = [[(node.healthy_flow-node.FlowRate)/node.healthy_flow for node in nodes]
                           for nodes in brain_outlet_nodes]

        threshold = 0.40
        number_below_threshold = [sum([flow_rate_change > threshold for flow_rate_change in flow_rate_change_nodes])
                                  for flow_rate_change_nodes in change_flowrate]
        infarct_volume_estimate = [number*average_volume_outlet for number in number_below_threshold]
        print(f"Infarct volume estimate: {infarct_volume_estimate}")
        patient.Results.Infarctvolumes.append(infarct_volume_estimate)

        with open(patient.Folders.ModellingFolder+'Infarct.csv', 'a+') as fd:
            # write change ratio to file
            writer = csv.writer(fd, delimiter=',')
            flat_list = [item for sublist in change_flowrate for item in sublist]
            writer.writerow(flat_list)

        file = patient.Folders.ModellingFolder+"DualPA.vtp"
        name_array = "Change flow rate"+str(i)
        array = [(node.healthy_flow-node.FlowRate)/node.healthy_flow for node in patient.Topology.PAnodes]
        GeneralFunctions.AddArrayToFile(file, array, name_array, cell=True)

    # output results
    # todo activate for results per iteration of topology
    # write_time_series(patient)

    filename = patient.Folders.ModellingFolder + "Clot_Permeability.csv"
    with open(filename, "w") as f:
        f.write(
            "Time [s],Pressure drop [pa],Time delay [s],Time delay bifurcation [s],Permeability [mm^2],Flow rate [mL/s],Velocity inside clot [m/s],Cerebral flow rate [mL/s],Infarct volume[mL]\n")
        for t, dp, dt, bif_dt, permeability, flow_rate, velocity,flowrates,infractvolumes in zip(patient.Results.Time[-1],
                                                                        patient.Results.PressureDrop,
                                                                        patient.Results.TimeDelay,
                                                                        patient.Results.BifTimeDelay,
                                                                        patient.Results.Permeability,
                                                                        patient.Results.ClotFlow,
                                                                        patient.Results.ClotVelocity,
                                                                        patient.Results.CerebralFlowRates,
                                                                        patient.Results.Infarctvolumes):
            f.write("%f,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (t, ",".join([str(i) for i in dp]),
                                                                        ",".join([str(i) for i in dt]),
                                                                        ",".join([str(i) for i in bif_dt]),
                                                                        ",".join([str(i) for i in permeability]),
                                                                        ",".join([str(i) for i in flow_rate]),
                                                                        ",".join([str(i) for i in velocity]),
                                                                        ",".join([str(i) for i in flowrates]),
                                                                        ",".join([str(i) for i in infractvolumes])))

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    transcript.stop()

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

        peak_times = vtk.vtkFloatArray()
        peak_times.SetNumberOfComponents(1)
        peak_times.SetName("Peak Time [s]")

        for node_velocity, node_flow, node_radius, node_pressure, node_peak_time in \
                zip(time_point.Velocity, time_point.Flow, time_point.Radius, time_point.Pressure,
                    time_point.peak_time):
            flow.InsertNextValue(node_flow)
            velocity.InsertNextValue(node_velocity)
            pressure.InsertNextValue(node_pressure)
            radius.InsertNextValue(node_radius)
            peak_times.InsertNextValue(node_peak_time)

        writer = vtk.vtkXMLPolyDataWriter()
        filename = output_folder + "Blood_flow" + str(time_index) + ".vtp"
        writer.SetFileName(filename)
        data.GetPointData().AddArray(flow)
        data.GetPointData().AddArray(velocity)
        data.GetPointData().AddArray(pressure)
        data.GetPointData().AddArray(radius)
        data.GetPointData().AddArray(peak_times)
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
    # location of the patient data
    sim_number = int(arguments["<number>"])

    start_time = datetime.now()
    Collateral_Simulation_Active = True  # set to True for collaterals simulations

    if Collateral_Simulation_Active:  # if collateral simulations
        # Folder = "/home/raymond/Desktop/Permeable_Clots_Collaterals_autoreg_refradius/"
        # Folder = "/home/raymond/Desktop/infarct_collaterals/"
        # Folder = "/home/raymond/Desktop/permeable_clots_autoreg/"
        Folder = "/home/raymond/Desktop/surface_view/"
        Folders = ["longMCA"]
        # Folders = ["CoW_Complete"]
        # Clots = ["R. MCA (1mm)"]
        # Clots = ["R. MCA (3mm)"]
        # Clots = ["R. MCA (6mm)"]
        # Clots = ["R. MCA (12mm)"]
        # Clots = ["R. MCA (16mm)"]
        # Clots = ["R. MCA (20mm)"]
        all_clots = [["R. MCA (1mm)"], ["R. MCA (3mm)"], ["R. MCA (6mm)"],
                     ["R. MCA (9mm)"], ["R. MCA (12mm)"], ["R. MCA (15mm)"],
                     ["R. MCA (18mm)"], ["R. MCA (21mm)"], ["R. MCA (24mm)"], ["R. MCA (28mm)"]]
        Clots = all_clots[sim_number]
        # Folders = ["longMCA1.1mm"]
        # Clots = ["R. MCA (7mm)"]
        # Clots = ["R. MCA (12mm)"]
        #
        # Clots = ["R. MCA (1mm)", "R. MCA (3mm)", "R. MCA (6mm)", "R. MCA (12mm)",
        #          "R. MCA (16mm)", "R. MCA (20mm)", "R. MCA (25mm)"]

        # collateralsProbability = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        collateralsProbability = [0.1, 0.3, 0.5, 0.8]

    else:
        # main folder
        # Folder = "/home/raymond/Desktop/Permeability/"
        Folder = "/home/raymond/Desktop/Permeable_Clots_rerun/"
        # Folder = "/home/raymond/Desktop/Permeable_clots_patients/"
        # BravaFolder = "/home/raymond/Desktop/1d-blood-flow/DataFiles/Brava"
        # BraVa = [f.split(".vtp")[0] for f in os.listdir(BravaFolder) if os.path.isfile(os.path.join(BravaFolder, f)) and f[-3:] == "vtp"]

        # Circle of Willis variations
        # Folders = ["CoW_ACoA",
        #            "CoW_Complete",
        #            "CoW_L_ACA_A1",
        #            "CoW_L_PCA_ P1",
        #            "CoW_L_PCoA",
        #            "CoW_L_PCoA_&_R_PCA_P1",
        #            "CoW_L+R_PCoA",
        #            "CoW_R_ACA_A1",
        #            "CoW_R_PCA_P1",
        #            "CoW_R_PCoA",
        #            "CoW_R_PCoA_&_L_PCA_P1", ]
        # Folders = ["CoW_Complete"]
        Folders = ["longMCA"]

        # Clots = ["Basilar I",
        #          "ICA-T",
        #          "LONG",
        #          "R. ACA, A2",
        #          "R. MCA",R. MCA (12mm)
        #          "R. PCA, P2",
        #          "R. PCoA",collateralSimulation
        #          "R. vertebral",
        #          "R int. Carotid I",
        #          "subclavian", ]
        Clots = ["R. MCA (1mm)", "R. MCA (3mm)", "R. MCA (6mm)", "R. MCA (12mm)",
                 "R. MCA (16mm)", "R. MCA (20mm)"]
        # Clots = ["R. MCA (20mm)"]

        collateralsProbability = [0.0]

    f = open(Folder + "Completed.txt", "w")
    # copy default files
    for collateralp in collateralsProbability:  # collateral score
        for clotfile in Clots:  # clot lengths
            for name in Folders:  # circle of Willis

                if Collateral_Simulation_Active:
                    patient_folder = Folder + name + '-' + clotfile + "-Collaterals(" + str(
                        collateralp) + ")/"  # name for collaterals
                else:
                    patient_folder = Folder + name + '-' + clotfile + "/"

                try:
                    # os.mkdir(Folder + name)
                    copy_tree("/home/raymond/Desktop/1d-blood-flow/DataFiles/DefaultPatient/", patient_folder)
                except OSError:
                    print("Creation of the directory %s failed" % name)
                else:
                    print("Successfully created the directory %s " % name)

                # Update CoW config in model_parameters.txt
                ModelParameters = Metadata.ModelParameter()
                ModelParameters.LoadModelParameters(patient_folder + "/bf_sim/Model_parameters.txt")
                ModelParameters["NewPatientFiles"] = True
                ModelParameters["UsingPatientSegmentation"] = False
                ModelParameters["Donor_Network"] = "BH0025_ColorCoded.CNG.vtp"
                ModelParameters["Circle_of_Willis"] = name
                ModelParameters["NewBravaSet"] = False
                ModelParameters["ClotFile"] = clotfile
                ModelParameters["UsingPAfile"] = True
                ModelParameters["ClusteringCompute"] = False
                ModelParameters["collateralpropability"] = collateralp
                ModelParameters["CollateralPatientGen"] = Collateral_Simulation_Active
                ModelParameters["pialsurfacepressure"] = 8000
                ModelParameters["FRICTION_C"] = 22
                ModelParameters["Permeability"] = 0
                ModelParameters["Porosity"] = 0
                ModelParameters.WriteModelParameters(patient_folder + "/bf_sim/Model_parameters.txt")

                try:
                    modelclotfile = "/home/raymond/Desktop/1d-blood-flow/DataFiles/Clots/" + clotfile + ".txt"
                    shutil.copy(modelclotfile, patient_folder)

                    with open(modelclotfile) as csvfile:
                        reader = csv.DictReader(csvfile)
                        clotinfo = [clot for clot in reader]

                    # overwrite file
                    with open(patient_folder + clotfile + ".txt", "w") as clotwriter:
                        clotwriter.write("Vesselname,Clotlocation(mm),Clotlength(mm),Permeability,Porosity\n")
                        for clot in clotinfo:
                            vesselname = clot['Vesselname']
                            location = clot['Clotlocation(mm)']
                            length = clot['Clotlength(mm)']

                            permeability = 0
                            porosity = 0
                            clotwriter.write(
                                "\"%s\",%s,%s,%0.12f,%f\n" % (vesselname, location, length, permeability, porosity))

                    try:
                        os.remove(patient_folder + 'Clots.txt')
                        print("Deleting file: ", patient_folder + 'Clots.txt')
                    except OSError:
                        print("Error while deleting file: ", patient_folder + 'Clots.txt')

                except OSError:
                    print("Copy of clot file failed!")

                # run the simulations
                try:
                    GenerateBloodflowFiles.generatebloodflowfiles(patient_folder)
                    permeability_simulation(patient_folder, iterations=2, contrast_sim_time=50)

                except OSError as err:
                    print("OS error: {0}".format(err))
                except ValueError:
                    print("Could not convert data to an integer.")
                except:
                    print("Unexpected error:", sys.exc_info()[0])

                f.write(patient_folder + "\n")
                f.flush()
                plt.close('all')
                try:  # remove distance matrix after generation
                    os.remove(patient_folder + "/bf_sim/Distancemat.npy")
                except:
                    pass

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    f.close()
