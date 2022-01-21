#!/usr/bin/env python
# coding: utf-8
"""
Contrast delay calculator based on graphs.
Dijkstra's algorithm is used to determine the shortest path.

Usage:
  ContrastGraphModel.py <patient_folder> <correct>
  ContrastGraphModel.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from datetime import datetime
from itertools import islice

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.sparse
import vtk
from scipy.signal import find_peaks

from Blood_Flow_1D import GeneralFunctions, Patient, docopt


def CalculateTimeDelay(patient):
    """
    Calculate timedelay from the inlet at node 0 to every other node in the network.
    The delay per edge is calculated as dx/v where v is the mean velocity and dx is the separation.
    :param patient: patient object
    :return: None.
    """
    print("Calculating the time delays in the patient network using shortest path.")
    # for each edge in the graph, get the mean of their velocity and their separation as the time delay
    for edge in patient.Topology.Graph.edges:
        node1 = edge[0]
        node2 = edge[1]
        v1 = patient.Results.MeanVelocityPerNode[-1][node1.Number]
        v2 = patient.Results.MeanVelocityPerNode[-1][node2.Number]

        meanv = (v1 + v2) / 2
        dx = patient.Topology.Graph[node1][node2]['weight'] * 1e-3

        if meanv == 0:
            continue
        t = dx / meanv

        if t < 0:
            patient.Topology.Graph[node2][node1]['timedelay'] = abs(t)
        elif t > 0:
            patient.Topology.Graph[node1][node2]['timedelay'] = t
        else:
            patient.Topology.Graph[node2][node1]['timedelay'] = 0
            patient.Topology.Graph[node1][node2]['timedelay'] = 0

    # calculate time delay using dijkstra algorithm starting at the inlet
    length, path = nx.single_source_dijkstra(patient.Topology.Graph, patient.Topology.Nodes[0], weight='timedelay')

    # store timedelay at the nodes
    for node in patient.Topology.Nodes:
        if node in length:
            node.timedelay = length[node]
        else:
            node.timedelay = float('nan')

    # Implement contrast delay measurement per vessel and relative time delay over the vessel
    for vessel in patient.Topology.Vessels:
        timedelay = [node.timedelay for node in vessel.Nodes]
        relativetimedelay = [t - timedelay[0] for t in timedelay]
        vessel.timedelay = timedelay
        vessel.relativetimedelay = relativetimedelay
        for node, reltime in zip(vessel.Nodes, relativetimedelay):
            node.relativetimedelay = reltime

    # relative time delay for a bifurcation node is not defined.
    for node in patient.Topology.BifurcationNodes:
        node.relativetimedelay = 0

    # store timedelays at the results object
    patient.Results.timedelay = [vessel.timedelay for vessel in patient.Topology.Vessels]
    patient.Results.relativetimedelay = [vessel.relativetimedelay for vessel in patient.Topology.Vessels]


def CalculatePerfusionTerritories(patient, PAfile):
    """
    Using dijkstra's algorithm, calculate downstream nodes and the perfusion territories.
    :param patient: patient object
    :param PAfile: PAfile to write the perfusion territories to.
    :return: Nothing
    """
    print("Calculating perfusion territories.")
    vessels = [patient.Topology.Vessels[23]]
    for vessel in vessels:
        length, path = nx.single_source_dijkstra_path(patient.Topology.Graph, vessel.Nodes[0], weight='timedelay')
        nodes = list(length.keys())
        outletnodes = [node for node in nodes if node in patient.Topology.PAnodes]
        regions = [patient.Topology.PAnodes.index(node) for node in outletnodes]
        array = np.zeros(len(patient.Topology.PAnodes))
        for r in regions:
            array[r] = 1
        GeneralFunctions.AddArrayToFile(PAfile, array, f"Perfusion Territory of vessel: {vessel.ID} ", False)


def WriteTimeDelays(patient, file="GraphTimeDelay.csv"):
    """
    Save the time delays in a csv file.
    :param patient: patient object
    :param file: timedelay file name
    :return: None.
    """
    filename = patient.Folders.ModellingFolder + file
    print("Writing Time Delays to %s" % filename)
    with open(filename, "w") as f:
        f.write("VesselName,LengthAlongVessel,TimeDelay,RelativeTimeDelay\n")
        for vessel in patient.Topology.Vessels:
            f.write("\"%s\"," % vessel.Name)
            lengthalongvessel = ",".join([str(node.LengthAlongVessel) for node in vessel.Nodes])
            f.write("\"%s\"," % lengthalongvessel)
            time = ",".join([str(i) for i in vessel.timedelay])
            f.write("\"%s\"," % time)
            relativetime = ",".join([str(i) for i in vessel.relativetimedelay])
            f.write("\"%s\"," % relativetime)
            f.write("\n")


def AddTimeDelaysToTopologyFile(patient, file="Topology.vtp"):
    """
    Add time delays to the patient topology file.
    :param patient: patient object
    :param file: topology file name
    :return: None.
    """
    # Write time delay to file
    file = patient.Folders.ModellingFolder + file
    print("Writing Time Delays to %s" % file)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file)
    reader.Update()
    data = reader.GetOutput()

    name = "Time Delay (graph)"
    resultdata = vtk.vtkFloatArray()
    resultdata.SetNumberOfComponents(1)
    resultdata.SetName(name)
    [resultdata.InsertNextValue(node.timedelay) for node in patient.Topology.Nodes]
    data.GetPointData().AddArray(resultdata)

    name = "Relative Time Delay (graph)"
    resultdata = vtk.vtkFloatArray()
    resultdata.SetNumberOfComponents(1)
    resultdata.SetName(name)
    [resultdata.InsertNextValue(node.relativetimedelay) for node in patient.Topology.Nodes]
    data.GetPointData().AddArray(resultdata)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(data)
    writer.Write()


def BoundaryUpdateMatrix(patient, numberNodes):
    BoundaryUpdateMat = scipy.sparse.identity(numberNodes, dtype='float64', format="lil")
    # label nodes if they belong to the distal or proximal part of the vessel

    for bifnode in patient.Topology.BifurcationNodes:
        # get total inflow into bif node
        inflow = sum([abs(node.FlowRate) for node in bifnode.Connections if node.Start * node.FlowRate > 0])
        bifnode.inflow = inflow
        bifin = [node for node in bifnode.Connections if node.Start * node.FlowRate > 0]
        bifout = [node for node in bifnode.Connections if node.Start * node.FlowRate <= 0]

        if len(bifout) > 0:
            bifnode.ContrastNode = bifout[0]  # store node for easy access later.
        else:
            print(inflow)
            bifnode.ContrastNode = bifin[0]

        for nodeout in bifout:
            for nodein in bifin:
                BoundaryUpdateMat[nodeout.ssindex, nodein.ssindex] = nodein.FlowRate / inflow

        # phi = sum([nextsolution[node.ssindex] * abs(node.FlowRate) / inflow for node in bifin])
        # for node in bifout:
        # Initialsolution[node.ssindex] = phi
        # bifnode.Contrast = phi
    return BoundaryUpdateMat


def ContrastGraphModel(patient_folder, correctness=True):
    """
    Load patient files and calculate the time delay through the patient network.
    The velocity used is the mean velocity of each edge.
    A timedelay.csv is created and the delays are added to the topology file.

    correctness: True if direction at the outlets is the same as the internal nodes.
    False if direction is opposite that of the internal nodes.
    True if steady model has used, False for pulsatile model.
    :param patient_folder: location of patient folder
    :param correctness: Default:True
    :return: None.
    """
    patient = Patient.Patient(patient_folder)
    patient.LoadBFSimFiles()
    patient.Topology.TopologyToGraph()
    # patient.LoadResults("Results.dyn", correct=correctness)
    patient.LoadResults("ResultsClot.dyn", correct=correctness)
    patient.GetMeanResults()
    #
    CalculateTimeDelay(patient)
    WriteTimeDelays(patient)
    AddTimeDelaysToTopologyFile(patient)

    ContrastPathSum(patient)

    return patient


def ContrastPathSum(patient):
    """
    Calculate the expected contrast profile based on the sum of paths.
    Results are saved in the simulation folder.

    Parameters
    ----------
    patient : patient object
        patient object containing all data
    """
    print("Calculating the path sum of contrast.")
    # calculate contrast profile

    print("\tInitiation.")
    for node, flowrate in zip(patient.Topology.Nodes, patient.Results.MeanVolumeFlowRatePerNode[-1]):
        node.FlowRate = flowrate

    # step one: define nodes A and B
    NodeA = patient.Topology.Nodes[0]

    # find the vessel(s) with a clot
    patient.Topology.NodeVesselDict()
    clotvessels = set()
    clotnodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    for clotnode in clotnodes:
        vessel = patient.Topology.NodeDict[clotnode]
        clotvessels.add(vessel)

    # NodeB = [vessel.Nodes[-1] for vessel in clotvessels][0]  # one node for now
    NodeB1 = [vessel.GetProximalBifurcation() for vessel in clotvessels][0]
    NodeB2 = [vessel.GetDistalBifurcation() for vessel in clotvessels][0]

    # AllPaths = nx.all_simple_paths(patient.Topology.Graph, source=NodeA, target=NodeB, cutoff=10)  # 10 seconds cuttoff
    # MaxPeak = Finalprofile.index(max(Finalprofile[peaks]))
    Finalprofile, profiles, time, peaks = CalculateContrastProfile(patient, NodeA, NodeB1)
    Finalprofile2, profiles2, time2, peaks2 = CalculateContrastProfile(patient, NodeA, NodeB2)

    print("\tPlotting profiles.")
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1000.0 / float(DPI), 1000.0 / float(DPI))

    axis1 = fig.add_subplot()
    axis1.set_title("Time Delay over a clot", fontsize=30)
    axis1.set_xlabel("Time (s)", fontsize=30)
    axis1.set_ylabel("Contrast ()", fontsize=30)
    axis1.xaxis.set_tick_params(labelsize=25)
    axis1.yaxis.set_tick_params(labelsize=25)

    # axis1.set_ylim([0,10])
    # axis1.set_xscale('log')
    axis1.grid()
    axis1.plot(time, Finalprofile)
    axis1.plot(time[peaks], Finalprofile[peaks], "x")
    axis1.plot(time2, Finalprofile2)
    axis1.plot(time2[peaks2], Finalprofile2[peaks2], "x")
    # axis1.plot(time[MaxPeak], Finalprofile[MaxPeak], "*")

    for path in profiles:
        axis1.plot(time, path, color='green', linewidth=4.0, alpha=0.5)

    for path in profiles2:
        axis1.plot(time, path, color='blue', linewidth=4.0, alpha=0.5)

    fig.canvas.draw_idle()
    fig.tight_layout()

    # plt.show()
    # plt.close(fig)
    print("\tSaving figure.")
    fig.savefig(patient.Folders.ModellingFolder + "ContrastProfile.png", dpi=72)

    # separate plots
    print("\tPlotting profiles.")
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1000.0 / float(DPI), 1000.0 / float(DPI))

    axis1 = fig.add_subplot()
    axis1.set_title("Time Delay over a clot", fontsize=30)
    axis1.set_xlabel("Time (s)", fontsize=30)
    axis1.set_ylabel("Contrast ()", fontsize=30)
    axis1.xaxis.set_tick_params(labelsize=25)
    axis1.yaxis.set_tick_params(labelsize=25)

    # axis1.set_ylim([0,10])
    # axis1.set_xscale('log')
    axis1.grid()
    axis1.plot(time, Finalprofile)
    axis1.plot(time[peaks], Finalprofile[peaks], "x")
    # axis1.plot(time[MaxPeak], Finalprofile[MaxPeak], "*")

    for path in profiles:
        axis1.plot(time, path, color='green', linewidth=4.0, alpha=0.5)

    fig.canvas.draw_idle()
    fig.tight_layout()

    # plt.show()
    # plt.close(fig)
    print("\tSaving figure.")
    fig.savefig(patient.Folders.ModellingFolder + "ContrastProfile1.png", dpi=72)

    print("\tPlotting profiles.")
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1000.0 / float(DPI), 1000.0 / float(DPI))

    axis1 = fig.add_subplot()
    axis1.set_title("Time Delay over a clot", fontsize=30)
    axis1.set_xlabel("Time (s)", fontsize=30)
    axis1.set_ylabel("Contrast ()", fontsize=30)
    axis1.xaxis.set_tick_params(labelsize=25)
    axis1.yaxis.set_tick_params(labelsize=25)

    # axis1.set_ylim([0,10])
    # axis1.set_xscale('log')
    axis1.grid()
    axis1.plot(time2, Finalprofile2)
    axis1.plot(time2[peaks2], Finalprofile[peaks2], "x")
    # axis1.plot(time[MaxPeak], Finalprofile[MaxPeak], "*")

    for path in profiles2:
        axis1.plot(time, path, color='green', linewidth=4.0, alpha=0.5)

    fig.canvas.draw_idle()
    fig.tight_layout()

    # plt.show()
    # plt.close(fig)
    print("\tSaving figure.")
    fig.savefig(patient.Folders.ModellingFolder + "ContrastProfile2.png", dpi=72)

    with open(patient.Folders.ModellingFolder + "ContrastPathDelay.txt", 'w') as f:
        timedelayPeaks = time2[peaks2[0]] - time[peaks[0]]
        f.write("%f" % timedelayPeaks)
    plt.close("all")


def CalculateContrastProfile(patient, NodeA, NodeB):
    """
    Calculate the contrast profile at Node B.
    Inlet node is NodeA.

    Parameters
    ----------
    patient : patient object
        object containing patient data
    NodeA : Node
        Initial node
    NodeB : Node
        Target node

    Returns
    -------
    Finalprofile: list of floats
        The sum of all individual profiles given in profiles.
    profiles : list of list of floats
        The shortest paths between NodeA and NodeB
    time : list of floats
        Time points belonging to the profiles
    peaks : list of integers
        List of indexes that belong to peaks in Finalprofile
    """
    print("Path finding.")

    def k_shortest_paths(G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    AllPaths = k_shortest_paths(patient.Topology.Graph, source=NodeA, target=NodeB, k=500, weight='timedelay')

    # for path in AllPaths:
    #     print(path)

    print("\tParameter calculation.")
    VesselsInPath = set()
    BifurcationsInPath = set()
    for path in AllPaths:
        for node in path:
            if node in patient.Topology.NodeDict:
                VesselsInPath.add(patient.Topology.NodeDict[node])
            else:
                BifurcationsInPath.add(node)

    FractionsDict = dict()  # [vessel] = 0.5
    FractionsDict[patient.Topology.NodeDict[NodeA]] = 1  # initial vessel
    # DelayDict =

    # label nodes if they belong to the distal or proximal part of the vessel
    for vessel in patient.Topology.Vessels:
        vessel.Nodes[0].Start = -1
        vessel.Nodes[-1].Start = 1

    for bifnode in BifurcationsInPath:
        # get total inflow into bif node
        inflow = sum([abs(node.FlowRate) for node in bifnode.Connections if node.Start * node.FlowRate > 0])
        bifnode.inflow = inflow
        for node in bifnode.Connections:
            if node.Start * node.FlowRate < 0:
                fraction = abs(node.FlowRate) / inflow
                FractionsDict[patient.Topology.NodeDict[node]] = fraction

    AllFractions = []
    AllTimeDelays = []
    #
    # # does networkx find all paths inc cycles?
    # for path in AllPaths:
    #     uniquenodes = set(path)
    #     if len(path) != len(uniquenodes):
    #         print("duplicate nodes found!")

    # find cycles
    try:
        a = list(nx.find_cycle(patient.Topology.Graph, source=NodeA, orientation='original'))
        print(a)
    except:
        print("\tNo cycles found.")
        pass

    # Calculate fractions and timedelays for all paths
    for path in AllPaths:
        PathVessels = set()
        for node in path:
            if node in patient.Topology.NodeDict:
                PathVessels.add(patient.Topology.NodeDict[node])

        TotalFraction = np.prod([FractionsDict[vessel] for vessel in PathVessels])
        TotalTime = np.sum([abs(vessel.timedelay[0] - vessel.timedelay[-1]) for vessel in PathVessels])
        AllFractions.append(TotalFraction)
        AllTimeDelays.append(TotalTime)
        # print("Path:")
        # print(TotalFraction)
        # print(TotalTime)

    print("\tContrast calculation.")
    # Calculating the final contrast profile
    time = np.linspace(0, 10, 200)
    profiles = np.array(
        [fraction * InletFunction(time - delay) for fraction, delay in zip(AllFractions, AllTimeDelays)])

    Finalprofile = profiles.sum(axis=0)
    peaks, _ = find_peaks(Finalprofile)
    return Finalprofile, profiles, time, peaks


def InletFunction(time):
    """
    Contrast inlet function
    Shifted Gaussian with the peak at t=0.15s

    Parameters
    ----------
    time : float
        imput time parameter

    Returns
    -------
    result : float
        Value at the imput time

    """
    return 1 * np.exp(-1 * (time - 0.15) * (time - 0.15) / (0.1))


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    patient_folder = arguments["<patient_folder>"]

    # True if direction at the outlets is the same as the internal nodes.
    # False if direction is opposite that of the internal nodes.
    # True if steady model has used, False for pulsatile model.
    resultcorrect = arguments["<correct>"]
    resultcorrect = True if resultcorrect == "True" else False

    start_time = datetime.now()
    # Run the contrast model with the data in patient_folder.
    ContrastGraphModel(patient_folder, resultcorrect)
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
