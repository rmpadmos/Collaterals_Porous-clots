#!/usr/bin/env python
# coding: utf-8
"""
Contrast advection simulator

Usage:
  ContrastModel.py <patient_folder> <correct> <duration>
  ContrastModel.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

import math
import os
from datetime import datetime
from multiprocessing.pool import ThreadPool as Pool

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse
import vtk
import copy

from Blood_Flow_1D import Collaterals, ContrastGraphModel, GeneralFunctions, Node, Patient, docopt


def boundaryUpdateMatrix(patient, number_nodes):
    """
    Calculate the matrix to update the bifurcation nodes.

    Parameters
    ----------
    patient : Patient object
        patient object with patient and flow data
    number_nodes: : int
        Number of nodes in the matrix

    Returns
    -------
    bifurcationMatrix : scipy.lilmatrix
        sparse matrix that calculates the concentration of the nodes at the start of the bifurcation

    """
    bifurcation_matrix = scipy.sparse.identity(number_nodes, dtype='float64', format="lil")
    # nodes are labelled if they belong to the distal or proximal part of the vessel

    for bifnode in patient.Topology.BifurcationNodes:
        # get total inflow into bif node
        inflow = sum([abs(node.FlowRate) for node in bifnode.Connections if node.Start * node.FlowRate > 0])
        bifnode.inflow = inflow
        bifin = [node for node in bifnode.Connections if node.Start * node.FlowRate >= 0]
        bifout = [node for node in bifnode.Connections if node.Start * node.FlowRate < 0]

        if len(bifout) > 0:
            bifnode.ContrastNode = bifout[0]  # store node for easy access later.
        else:
            bifnode.ContrastNode = bifin[0]

        # for nodeout in bifout:
        #     for nodein in bifin:
        #         bifurcation_matrix[nodeout.ssindex, nodein.ssindex] = nodein.FlowRate / inflow

        for nodeout in bifout:
            for nodein in bifin:
                if inflow == 0:
                    bifurcation_matrix[nodeout.ssindex, nodein.ssindex] = 0
                    # print(f"aaaaaaaaa:{nodein.FlowRate}")
                else:
                    bifurcation_matrix[nodeout.ssindex, nodein.ssindex] = nodein.FlowRate / inflow

    return bifurcation_matrix


def contrast_clot_ends(patient, hdf5DataFile):
    """
    Create figures of the contrast profiles at the proximal and distal parts of a vessel with a clot.
    Figures are saved in the modelling folder. Time delay values are stored in ContrastDelay.csv

    Parameters
    ----------
    patient : patient object

    Returns
    -------
    fig : matplotlib figure

    """
    print("Calculating Time delay.")
    # get nodes
    patient.Topology.NodeVesselDict()
    clotvessels = set()
    clotnodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    for clotnode in clotnodes:
        vessel = patient.Topology.NodeDict[clotnode]
        clotvessels.add(vessel)

    if len(clotvessels) == 0:
        return 1
    #
    NodeB1 = [vessel.GetProximalBifurcation() for vessel in clotvessels][0]
    NodeB2 = [vessel.GetDistalBifurcation() for vessel in clotvessels][0]

    contrastConcentration = np.array(hdf5DataFile.get('Contrast Concentration'))
    time = np.array(hdf5DataFile.get('Contrast Time'))

    profile1 = contrastConcentration[:, NodeB1.Number]
    profile2 = contrastConcentration[:, NodeB2.Number]

    print(f"\tnode 1:{NodeB1.Number}")
    print(f"\tnode 2:{NodeB2.Number}")
    print(f'\tTime delay: {abs(NodeB1.peak_time - NodeB2.peak_time)}')
    with open(patient.Folders.ModellingFolder + "ContrastDelay.csv", 'w') as f:
        f.write("Time Delay (s)\n")
        f.write("%f" % abs(NodeB1.peak_time - NodeB2.peak_time))

    # create figures
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    # DPI = fig.get_dpi()
    # fig.set_size_inches(1000.0 / float(DPI), 1000.0 / float(DPI))

    axis1 = fig.add_subplot()
    # axis1.set_title("Contrast over time", fontsize=30)
    axis1.set_xlabel("Time (s)", fontsize=30)
    axis1.set_ylabel("Contrast (arb. unit)", fontsize=30)
    axis1.xaxis.set_tick_params(labelsize=25)
    axis1.yaxis.set_tick_params(labelsize=25)

    axis1.grid()
    axis1.plot(time, profile1, label="Proximal End", linewidth=5)
    axis1.plot(time, profile2, label="Distal End", linewidth=5)
    axis1.legend(fontsize=30)

    axis1.set_xlim([0, time[-1]])
    # axis1.set_xscale('log')
    fig.canvas.draw_idle()
    fig.tight_layout()
    # plt.show()

    # save results
    figname = "ContrastModelling.png"
    fig.savefig(patient.Folders.ModellingFolder + figname)
    # fig.savefig(patient.Folders.ModellingFolder + figname, dpi=72)

    plt.close("all")

    return fig


def contrastMatrixUpwind(patient, dt):
    """
    Generate matrix using the upwind numerical scheme.
    Parameters
    ----------
    patient : patient object
        patient object with vasculature and flow data
    dt : float
        Time step

    Returns
    -------
    mtx : scipy.csc_matrix
        matrix obtained
    inputvection : matrix with zeros
        Initial condution
    nodesetlist : list
        List of nodes in the order that they appear in the matrix.

    """
    self = patient.Topology

    nodesetlist = [node for vessel in self.Vessels for node in vessel.Nodes]

    for index, node in enumerate(nodesetlist):
        node.ssindex = index

    mtx = scipy.sparse.lil_matrix((len(nodesetlist), len(nodesetlist)))

    def updatemat(vessel):
        c = vessel.Velocity * dt / vessel.GridSize
        for index, node in enumerate(vessel.Nodes[1:]):
            leftnode = vessel.Nodes[index].ssindex
            middlenode = node.ssindex

            # upwind
            mtx[middlenode, middlenode] = 1 - c
            mtx[middlenode, leftnode] = c

    with Pool() as pool:
        pool.map(updatemat, self.Vessels)

    inputvector = np.zeros((len(nodesetlist),))

    for inletnode in self.InletNodes:
        mtx[inletnode[0].ssindex, inletnode[0].ssindex] = 1

    return mtx.tocsc(), inputvector, nodesetlist


def contrastSimulationUpwind(patient, simulationtime=30, number_output_files=100, hdf5DataFile=None,
                             inlet_section=True, clot_contrast_disable=False):
    """
    Simulation of contrast through the patient vasculature.

    Parameters
    ----------
    clot_contrast_disable: disable contrast after the clot
    number_output_files
    hdf5DataFile
    inlet_section
    patient : patient object
        patient object with flow data
    simulationtime : float
        duration of the simulation

    """
    print(f"Simulating contrast through the patient vasculature. Duration:{simulationtime}s")
    dt = 0.05

    maxC = 0
    # for vessel, data in zip(patient.Topology.Vessels, patient.Results.MeanResults):
    for vessel in patient.Topology.Vessels:
        # velocity = data[1][3] * 1e3  # mm/s
        if vessel.Velocity < 0:
            vessel.Nodes.reverse()  # if velocity is negative, reverse the order of the nodes.

            lengthalongvessels = copy.deepcopy([node.LengthAlongVessel for node in vessel.Nodes])
            lengthalongvessels.reverse()
            [node.SetLengthAlongVessel(length) for node, length in zip(vessel.Nodes,lengthalongvessels)]

        vessel.Velocity = abs(vessel.Velocity)
        vessel.Nodes[0].FlowRate = abs(vessel.Nodes[0].FlowRate)
        vessel.Nodes[-1].FlowRate = abs(vessel.Nodes[-1].FlowRate)
        vessel.Nodes[0].Start = -1
        vessel.Nodes[-1].Start = 1
        maxC = max(vessel.Velocity * dt / vessel.GridSize, maxC)

    print("\tMax C: %f" % maxC)
    dt = dt / (maxC / 0.9)
    dt = 0.001
    print("\tSetting dt to %f" % dt)

    Nodes = []
    Nnodes = 0
    if inlet_section:
        # extend the first vessel to include the concentration function in the vector.
        Nnodes = 200
        Nodes = [Node.Node() for _ in range(0, Nnodes)]
        [node.SetLengthAlongVessel(i * patient.Topology.Vessels[0].GridSize) for i, node in enumerate(Nodes)]
        patient.Topology.Vessels[0].Nodes = Nodes + patient.Topology.Vessels[0].Nodes

        for index, node in enumerate(patient.Topology.Vessels[0].Nodes[:-1]):
            node.AddConnection(patient.Topology.Vessels[0].Nodes[index + 1])
            patient.Topology.Vessels[0].Nodes[index + 1].AddConnection(node)

        # update inlet node
        patient.Topology.InletNodes[0] = (Nodes[0], patient.Topology.InletNodes[0][1])

    mtx, inputvector, nodesetlist = contrastMatrixUpwind(patient, dt)

    Initialsolution = inputvector
    patient.Results.ContrastTime = []
    patient.Results.Contrast = []

    BoundaryUpdateMat = boundaryUpdateMatrix(patient, len(nodesetlist))
    mtx = BoundaryUpdateMat.dot(mtx)  # update matrix to include bifucation conditions

    if inlet_section:
        # update to old inlet nodes
        print("using inlet section")
        patient.Topology.InletNodes[0] = (patient.Topology.Nodes[0], patient.Topology.InletNodes[0][1])
        patient.Topology.Vessels[0].Nodes = patient.Topology.Vessels[0].Nodes[Nnodes:]

        # initial condition
        for node in Nodes:
            pos = abs(node.LengthAlongVessel - Nnodes * patient.Topology.Vessels[0].GridSize)
            velocity = patient.Topology.Vessels[0].Velocity
            time = 1
            dx = velocity * time
            Initialsolution[node.ssindex] = 10 * np.exp(-1 * (pos - dx) * (pos - dx) / (250))  # gaussian pulse
            # Initialsolution[node.ssindex] = 1  # front
            # Initialsolution[node.ssindex] = np.heaviside(dx - pos, 1)  # block, time here indicates duration of the block.

    for node in patient.Topology.BifurcationNodes:
        node.Contrast = 0

    # simulation
    start_time = datetime.now()
    time = 0.0
    maxiter = math.ceil(simulationtime / dt) + 2
    contrastFiles = number_output_files
    iterstep = math.floor(maxiter / contrastFiles)

    print(f"\tSimulation time: {time}, Output number: 1/{contrastFiles}")
    for node, contrast in zip(nodesetlist, Initialsolution):
        node.Contrast = contrast

    ContrastTime = [iter * dt for iter in range(0, maxiter) if iter % iterstep == 0]
    result = [node.Contrast for node in patient.Topology.Nodes]

    if hdf5DataFile is not None:
        hdf5DataFile.create_dataset('Contrast Concentration', (len(ContrastTime), len(result)), compression="gzip",
                                    compression_opts=9, chunks=True)
        hdf5DataFile['Contrast Concentration'][0, :] = result

    writeIter = 0
    peak_times = [np.inf for _ in nodesetlist]

    # todo disable contrast after clot?
    if clot_contrast_disable:
        print("Disabled contrast after clot vessel!")
        disabled_nodes = []
        for index, node in enumerate(nodesetlist):
            if node in patient.Topology.Vessels[-1].Nodes[1:-1]: # except first and last
                disabled_nodes.append(index)

    for nt in range(1, maxiter):
        # update initial condition if not using inlet_section
        if not inlet_section:
            node = patient.Topology.Vessels[0].Nodes[0]
            velocity = patient.Topology.Vessels[0].Velocity
            Initialsolution[node.ssindex] = 10 * np.exp(
                -1 * (velocity * time - velocity * 1) * (velocity * time - velocity * 1) / (250))  # gaussian pulse
            # Initialsolution[node.ssindex] = np.heaviside(velocity*1 - velocity*time, 1) # block, time (1) here indicates duration of the block.
        next_step = mtx.dot(Initialsolution)
        # todo disable contrast after clot?
        # print("Disabled contrast after clot vessel!")
        # for index, node in enumerate(nodesetlist):
        #     if node in patient.Topology.Vessels[-1].Nodes:
        #         next_step[index] = 0.0
        if clot_contrast_disable:
            # for index in disabled_nodes:
            #     next_step[index] = 0.0
            next_step[disabled_nodes] = 0.0

        time = nt * dt
        # peak_times = np.where(((next_step - Initialsolution) > 0.0) and next_step > 0.01, time, peak_times)
        peak_times = np.where((next_step - Initialsolution) > 0.0, time, peak_times)
        Initialsolution = next_step

        if nt % iterstep == 0 and hdf5DataFile is not None:
            print(f"\tSimulation time: {time}")
            for node, contrast in zip(nodesetlist, Initialsolution):
                node.Contrast = contrast
            for node in patient.Topology.BifurcationNodes:
                node.Contrast = node.ContrastNode.Contrast

            writeIter += 1
            hdf5DataFile['Contrast Concentration'][writeIter, :] = [node.Contrast for node in patient.Topology.Nodes]

    # map peak time to bifurcation nodes
    for node, peak_time in zip(nodesetlist, peak_times):
        node.peak_time = peak_time
    for node in patient.Topology.BifurcationNodes:
        node.peak_time = node.ContrastNode.peak_time

    if hdf5DataFile is not None:
        hdf5DataFile.create_dataset('Contrast Time', data=ContrastTime, compression="gzip", compression_opts=9)

    time_elapsed = datetime.now() - start_time
    print('\tComputation time (hh:mm:ss.ms) {}'.format(time_elapsed))


def writeContrastPAnodes(patient, hdf5DataFile):
    """
    Write the results of the contrast simulation to file.
    Output is written to ResultsContrastPA.pvd

    Parameters
    ----------
    patient : patient object
        patient object with simulation results.
    """
    print("Writing PA Contrast Simulation files.")
    contrastConcentration = hdf5DataFile['Contrast Concentration']
    time = hdf5DataFile['Contrast Time']

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(patient.Folders.ModellingFolder + "DualPA.vtp")
    reader.Update()
    data = reader.GetOutput()
    outputfolder = patient.Folders.ModellingFolder + "TimeSeriesContrastPA/"
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    for timeindex, timepoint in enumerate(time):
        Contrast = vtk.vtkFloatArray()
        Contrast.SetNumberOfComponents(1)
        Contrast.SetName("Contrast Concentration")

        for node in patient.Topology.PAnodes:
            Contrast.InsertNextValue(contrastConcentration[timeindex, node.Number])

        writer = vtk.vtkXMLPolyDataWriter()
        filename = outputfolder + "Contrast" + str(timeindex) + ".vtp"
        writer.SetFileName(filename)
        # data.clear_arrays()
        data.GetCellData().AddArray(Contrast)
        writer.SetInputData(data)
        writer.Write()

    with open(patient.Folders.ModellingFolder + "ResultsContrastPA.pvd", "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write(
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
        f.write("<Collection>\n")
        for timeindex, timepoint in enumerate(time):
            name = "TimeSeriesContrastPA/Contrast" + str(timeindex) + ".vtp"
            time = str(timepoint)
            f.write("<DataSet timestep=\"" + time + "\" group=\"\" part=\"0\" file=\"" + name + "\"/>\n")
        f.write("</Collection>\n")
        f.write("</VTKFile>")


def writeContrastPAnodesSurface(patient, hdf5DataFile):
    """
    Write the results of the contrast simulation to file.
    Output is written to ResultsContrastPASurface.pvd

    Parameters
    ----------
    patient : patient object
        patient object with simulation results.
    """
    print("Writing PA Surface Contrast Simulation files.")
    contrastConcentration = hdf5DataFile['Contrast Concentration']
    time = hdf5DataFile['Contrast Time']

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(patient.Folders.ModellingFolder + "PA.vtp")
    reader.Update()
    data = reader.GetOutput()
    outputfolder = patient.Folders.ModellingFolder + "TimeSeriesContrastPASurface/"
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    for timeindex, timepoint in enumerate(time):
        Contrast = vtk.vtkFloatArray()
        Contrast.SetNumberOfComponents(1)
        Contrast.SetName("Contrast Concentration")

        for node in patient.Topology.PAnodes:
            Contrast.InsertNextValue(contrastConcentration[timeindex, node.Number])

        writer = vtk.vtkXMLPolyDataWriter()
        filename = outputfolder + "Contrast" + str(timeindex) + ".vtp"
        writer.SetFileName(filename)
        # data.clear_arrays()
        data.GetPointData().AddArray(Contrast)
        writer.SetInputData(data)
        writer.Write()

    with open(patient.Folders.ModellingFolder + "ResultsContrastPASurface.pvd", "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write(
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
        f.write("<Collection>\n")
        for timeindex, timepoint in enumerate(time):
            name = "TimeSeriesContrastPASurface/Contrast" + str(timeindex) + ".vtp"
            time = str(timepoint)
            f.write("<DataSet timestep=\"" + time + "\" group=\"\" part=\"0\" file=\"" + name + "\"/>\n")
        f.write("</Collection>\n")
        f.write("</VTKFile>")


def writeContrastVessels(patient, hdf5DataFile):
    """
    Write the results of the contrast simulation to file.
    Output is written to ResultsContrast.pvd

    Parameters
    ----------
    patient : patient object
        patient object with simulation results.
    """
    print("Writing Contrast Simulation files.")
    contrastConcentration = hdf5DataFile['Contrast Concentration']
    time = hdf5DataFile['Contrast Time']

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(patient.Folders.ModellingFolder + "Topology.vtp")
    reader.Update()
    data = reader.GetOutput()
    outputfolder = patient.Folders.ModellingFolder + "TimeSeriesContrast/"
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    for timeindex, timepoint in enumerate(time):
        Contrast = vtk.vtkFloatArray()
        Contrast.SetNumberOfComponents(1)
        Contrast.SetName("Contrast Concentration")

        for nodeindex in range(0, len(contrastConcentration[timeindex, :])):
            Contrast.InsertNextValue(contrastConcentration[timeindex, nodeindex])

        writer = vtk.vtkXMLPolyDataWriter()
        filename = outputfolder + "Contrast" + str(timeindex) + ".vtp"
        writer.SetFileName(filename)
        # data.clear_arrays()
        data.GetPointData().AddArray(Contrast)
        writer.SetInputData(data)
        writer.Write()

    with open(patient.Folders.ModellingFolder + "ResultsContrast.pvd", "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write(
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
        f.write("<Collection>\n")
        for timeindex, timepoint in enumerate(time):
            name = "TimeSeriesContrast/Contrast" + str(timeindex) + ".vtp"
            time = str(timepoint)
            f.write("<DataSet timestep=\"" + time + "\" group=\"\" part=\"0\" file=\"" + name + "\"/>\n")
        f.write("</Collection>\n")
        f.write("</VTKFile>")


def run_contrast_model(patient_folder, correctness=True, duration=30, number_output_files=100, resultfile="Results.dyn",
             hdf5_file='ContrastData.h5', output_time_series=True, inlet_section=False):
    """
    Load patient files and calculate the time delay through the patient network.
    The velocity used is the mean velocity of each edge.
    A timedelay.csv is created and the delays are added to the topology file.

    correctness: True if direction at the outlets is the same as the internal nodes.
    False if direction is opposite that of the internal nodes.
    True if steady model has used, False for pulsatile model.

    Parameters
    ----------
    patient_folder : String
         location of patient folder
    correctness : Boolean
        Does the model need to update the flow data at the outlets?
        Pulsatile model output requires correction.
    duration : float
        duration of the contrast simulation

    Returns
    -------
    patient: patient object
        patient object with results.

    """
    patient = Patient.Patient(patient_folder)
    patient.LoadBFSimFiles()
    patient.Topology.TopologyToGraph()
    patient.LoadResults(resultfile, correct=correctness)
    patient.GetMeanResults()

    for node, flowrate in zip(patient.Topology.Nodes, patient.Results.MeanVolumeFlowRatePerNode[-1]):
        node.FlowRate = flowrate

    for vessel, data in zip(patient.Topology.Vessels, patient.Results.MeanResults):
        vessel.Velocity = data[1][3] * 1e3  # mm/s

    ContrastGraphModel.CalculateTimeDelay(patient)
    ContrastGraphModel.WriteTimeDelays(patient)
    ContrastGraphModel.AddTimeDelaysToTopologyFile(patient)

    if patient.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
            patient.Folders.ModellingFolder + "PAmapping.csv"):
        Collaterals.ImportPAnodes(patient)

    if hdf5_file is not None:
        hdf5DataFile = h5py.File(patient.Folders.ModellingFolder + hdf5_file, 'w')
    else:
        hdf5DataFile = None
    contrastSimulationUpwind(patient, simulationtime=duration, hdf5DataFile=hdf5DataFile,
                             number_output_files=number_output_files, inlet_section=inlet_section)

    if output_time_series:
        writeContrastVessels(patient, hdf5DataFile)
    if hdf5_file is not None:
        contrast_clot_ends(patient, hdf5DataFile)

    peak_times = [node.peak_time for node in patient.Topology.Nodes]
    if hdf5_file is not None:
        hdf5DataFile.create_dataset("Peak Times", data=peak_times, compression="gzip", compression_opts=9)
    patient.Results.peak_times = peak_times

    GeneralFunctions.AddArrayToFile(patient.Folders.ModellingFolder + "Topology.vtp", patient.Results.peak_times,
                                    "Peak Times (s)",
                                    False)

    if patient.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
            patient.Folders.ModellingFolder + "PAmapping.csv"):
        if output_time_series:
            writeContrastPAnodes(patient, hdf5DataFile)
            writeContrastPAnodesSurface(patient, hdf5DataFile)
        peak_times = [node.peak_time for node in patient.Topology.PAnodes]
        GeneralFunctions.AddArrayToFile(patient.Folders.ModellingFolder + "DualPA.vtp", peak_times, "Peak time (s)",
                                        True)
        GeneralFunctions.AddArrayToFile(patient.Folders.ModellingFolder + "PA.vtp", peak_times, "Peak time (s)", False)
    if hdf5_file is not None:
        hdf5DataFile.close()
    return patient


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')

    # location of the patient data
    patient_folder = arguments["<patient_folder>"]

    # True if direction at the outlets is the same as the internal nodes.
    # False if direction is opposite that of the internal nodes.
    # True if steady model has used, False for pulsatile model.
    # todo remove this input, setting located in modellingparamters
    resultcorrect = arguments["<correct>"]
    resultcorrect = True if resultcorrect == "True" else False

    duration = float(arguments["<duration>"])

    start_time = datetime.now()

    # Run the contrast model with the data in patient_folder.
    run_contrast_model(patient_folder, correctness=resultcorrect, duration=duration, number_output_files=100,
             resultfile="Results.dyn", hdf5_file='ContrastData.h5', output_time_series=True, inlet_section=False)

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
