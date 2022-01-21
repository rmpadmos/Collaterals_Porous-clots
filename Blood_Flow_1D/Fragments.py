#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Fragment simulation

Requires a file that specifies the fragmentation distribution in the patient folder: micro_thrombi_distribution.csv
Optional file is the Results.dyn to be used to the micro-thrombi shower.

Output in bf_sim:
	Large_fragments.txt
	Micro-thrombi.csv
	Additional "Clot" array in Topology.vtp

optional
	Matlibplot figure of occlusion probability for all nodes:  "ClotSize_Probability.png"
	Occlusion probability per node: ClotSize_Probability.csv
	Occlusion probability per PAnode is using detailed collateral model: ClotSize_Probability_PAnodes.csv
	Folder with occlusion probabilities for various clot sizes: FragmentsTopologySeries/*

Usage:
  Fragments.py <patient_folder> [<file>]
  Fragments.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
import vtk
from scipy.interpolate import interp1d
from scipy.sparse.linalg import inv

from Blood_Flow_1D import Collaterals, GeneralFunctions, Patient, docopt, transcript, CollateralsSimulation


def patient_loading(patient_folder, results_file):
    patient_object = Patient.Patient(patient_folder)
    patient_object.LoadBFSimFiles()
    patient_object.LoadModelParameters("Model_parameters.txt")
    patient_object.LoadClusteringMapping(patient_object.Folders.ModellingFolder + "Clusters.csv")
    patient_object.LoadPositions()
    patient_object.LoadResults(file=results_file)
    return patient_object


    # # Load files
    # patient_object = Patient.Patient(patient_folder)
    # patient_object.LoadBFSimFiles()
    # # patient_object.LoadClusteringMapping(patient_object.Folders.ModellingFolder + "Clusters.csv")
    # # patient_object.LoadSurfaceMapping()
    # patient_object.LoadPositions()
    # # patient_object.Perfusion.PrimalGraph.LoadSurface(patient.Folders.ModellingFolder + "Clustering.vtp")
    # # patient_object.Perfusion.SetDualGraph(method="vertices")
    # # patient_object.Perfusion.DualGraph.NodeColour = patient.Perfusion.PrimalGraph.map
    #
    # # patient_object.Initiate1DSteadyStateModel()
    # # if patient_object.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
    # #         patient_object.Folders.ModellingFolder + "PAmapping.csv"):
    # #     Collaterals.ImportPAnodes(patient_object)
    # #     # if PAnode are present, run the model with the autoregulation method.
    # #
    # #     CollateralsSimulation.collateral_simulation(patient_object, clot_active=False)
    # # else:
    # #     patient_object.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=False, PressureInlets=True,
    # #                                          FlowRateOutlets=False, coarseCollaterals=False,
    # #                                          frictionconstant=patient_object.ModelParameters["FRICTION_C"])
    #
    # # load already computed solution
    #
    #
    # return patient_object


def fragment_simulation(patient_folder_location, sizes, results_file, plot_example=False,save_optional_results=False):
    """
    The goal is to get a probability curve per node (segment) of a clot getting stuck (as a function of radius).

    Load patient network, simulate healthy flow, track fragments.
    Create a figure showing the probability function of all nodes.
    Outputs CSV files for nodes/PAnodes and the probability of blockages per clot size.

    Parameters
    ----------
    sizes: list of fragment sizes.
    plot_example: boolean to plot probability for every node in the system
    patient_folder_location: patient folder location

    Returns
    -------
    """
    start_time = datetime.now()
    transcript.start(patient_folder_location + 'logfile.log')

    patient_object = patient_loading(patient_folder_location, results_file)

    # collect all probability dictionaries and build probability functions per node (as function of clot radius)
    fragment_locations = [probability_map(patient_object, size) for size in sizes]
    print("Probability computed")
    # for nodes not present in the location dictionary, the probability is zero.
    # for each node, create a probability array
    for node in patient_object.Topology.Nodes:
        node.FragmentProbability = []

    for size, fragment_prob in zip(sizes, fragment_locations):
        for node in patient_object.Topology.Nodes:
            try:
                prob = fragment_prob[node]
            except KeyError:
                prob = 0
            node.FragmentProbability.append(prob)

    # interpolate functions based on the sampling of fragment sizes.
    for node in patient_object.Topology.Nodes:
        node.FragmentProbabilityFunction = interp1d(sizes, node.FragmentProbability, kind='nearest')

    for bif_node in patient_object.Topology.BifurcationNodes:
        for node in bif_node.Connections:
            node.FragmentProbabilityFunction = bif_node.FragmentProbabilityFunction
            node.FragmentProbability = bif_node.FragmentProbability

    # plot example (all nodes in the network)
    if plot_example:
        sizes_new = np.linspace(min(sizes), max(sizes), num=1000, endpoint=True)

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 10)
        for node in patient_object.Topology.Nodes:
            y = node.FragmentProbability
            f = node.FragmentProbabilityFunction
            plt.plot(sizes, y, 'o', sizes_new, f(sizes_new), '-', linewidth=3, markersize=5)

        ax.set_title("Probability vs clot size", fontsize=20)
        ax.set_xlabel("Clot Size [mm]", fontsize=20)
        ax.set_ylabel("Probability", fontsize=20)
        ax.tick_params(labelsize=20)
        ax.grid()
        plt.tight_layout()
        # plt.show()
        fig.savefig(patient_object.Folders.ModellingFolder + "ClotSize_Probability")
        print("Figure done")

    ################### save results
    if save_optional_results:
        filename = patient_object.Folders.ModellingFolder + "ClotSize_Probability.csv"
        with open(filename, "w") as f:
            f.write("NodeNumber,ClotSizes,Probabilities\n")
            clot_sizes_string = ",".join([str(i) for i in sizes])
            for node in patient_object.Topology.Nodes:
                probability_string = ",".join([str(i) for i in node.FragmentProbability])
                f.write("%f,\"%s\",\"%s\"\n" % (node.Number, clot_sizes_string, probability_string))

        # Output affected PA nodes
        if patient_object.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
                patient_object.Folders.ModellingFolder + "PAmapping.csv"):
            filename = patient_object.Folders.ModellingFolder + "ClotSize_Probability_PAnodes.csv"
            with open(filename, "w") as f:
                f.write("NodeNumber,ClotSizes,Probabilities\n")
                clot_sizes_string = ",".join([str(i) for i in sizes])
                for node in patient_object.Topology.PAnodes:
                    if sum(node.FragmentProbability) > 0:
                        f.write("%f,\"%s\",\"%s\"\n" % (
                            node.Number, clot_sizes_string, ",".join([str(i) for i in node.FragmentProbability])))

        # write series
        topology_file = patient_object.Folders.ModellingFolder + "Topology.vtp"
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(topology_file)
        reader.Update()
        data = reader.GetOutput()

        folder_fragments = "FragmentsTopologySeries/"
        output_folder = patient_object.Folders.ModellingFolder + folder_fragments
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for index, size in enumerate(sizes):
            probability = vtk.vtkFloatArray()
            probability.SetNumberOfComponents(1)
            probability.SetName("Fragment Probability")

            for node in patient_object.Topology.Nodes:
                probability.InsertNextValue(node.FragmentProbability[index])

            writer = vtk.vtkXMLPolyDataWriter()
            filename = output_folder + "Fragment" + str(size) + ".vtp"
            writer.SetFileName(filename)
            data.GetPointData().AddArray(probability)
            writer.SetInputData(data)
            writer.Write()

        with open(patient_object.Folders.ModellingFolder + folder_fragments + "FragmentsTopology.pvd", "w") as f:
            f.write("<?xml version=\"1.0\"?>\n")
            f.write(
                "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
            f.write("<Collection>\n")
            for index, size in enumerate(sizes):
                name = folder_fragments + "Fragment" + str(size) + ".vtp"
                time = str(size)
                f.write("<DataSet timestep=\"" + time + "\" group=\"\" part=\"0\" file=\"" + name + "\"/>\n")
            f.write("</Collection>\n")
            f.write("</VTKFile>")

        # map PAnodes probability to surface file
        if patient_object.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
                patient_object.Folders.ModellingFolder + "PAmapping.csv"):
            surface_file = patient_object.Folders.ModellingFolder + "PA.vtp"

            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(surface_file)
            reader.Update()
            data = reader.GetOutput()
            output_folder = patient_object.Folders.ModellingFolder + "FragmentsSurfaceSeries/"
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            for index, size in enumerate(sizes):
                probability = vtk.vtkFloatArray()
                probability.SetNumberOfComponents(1)
                probability.SetName("Fragment Probability")

                for node in patient_object.Topology.PAnodes:
                    probability.InsertNextValue(node.FragmentProbability[index])

                writer = vtk.vtkXMLPolyDataWriter()
                filename = output_folder + "Fragment" + str(size) + ".vtp"
                writer.SetFileName(filename)
                data.GetPointData().AddArray(probability)
                # data.GetCellData().AddArray(probability)
                writer.SetInputData(data)
                writer.Write()

            with open(patient_object.Folders.ModellingFolder + "FragmentsSurface.pvd", "w") as f:
                f.write("<?xml version=\"1.0\"?>\n")
                f.write(
                    "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
                f.write("<Collection>\n")
                for index, size in enumerate(sizes):
                    name = "FragmentsSurfaceSeries/Fragment" + str(size) + ".vtp"
                    time = str(size)
                    f.write("<DataSet timestep=\"" + time + "\" group=\"\" part=\"0\" file=\"" + name + "\"/>\n")
                f.write("</Collection>\n")
                f.write("</VTKFile>")

    # Fragment tracking done
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    transcript.stop()
    return patient_object


def probability_map(patient_object, radius):
    """
    Return a dict of nodes with non-zero probability of blockage for the given clot radius.
    Parameters
    ----------
    patient_object: patient object
    radius: clot radius

    Returns
    -------
    dictionary of nodes and the probability of a clot fragment getting stuck at the node.
    """
    print("Calculating Probability")
    # transform network to graph
    network = nx.DiGraph()

    # currently only one clot
    clot_node = patient_object.Topology.Clots[0][0][0]
    # clot_node = patient.Topology.InletNodes[0][0]  # inlet node, all other nodes reachable!

    # calculate probability at bifurcations
    for vessel in patient_object.Topology.Vessels:
        vessel.Nodes[0].Start = -1
        vessel.Nodes[-1].Start = 1

    for bif_node in patient_object.Topology.BifurcationNodes:
        # get total inflow into bif node
        inflow = sum([abs(node.FlowRate) for node in bif_node.Connections if node.Start * node.FlowRate > 0])
        bif_node.inflow = inflow

        bif_in = [node for node in bif_node.Connections if node.Start * node.FlowRate >= 0]
        bif_out = [node for node in bif_node.Connections if node.Start * node.FlowRate < 0]

        blocked_flow = sum([abs(node.FlowRate) for node in bif_node.Connections if
                            node.Start * node.FlowRate < 0 and node.Radius < radius])

        for node_in in bif_in:
            node_in.P = 1
            network.add_edge(node_in, bif_node, direction=0)

        # use relative flow rates as probability
        for node_out in bif_out:
            if node_out.Radius < radius or inflow == blocked_flow:
                node_out.P = 0
            else:
                node_out.P = abs(node_out.FlowRate) / (inflow - blocked_flow)
                prob = abs(-np.log(node_out.P))
                network.add_edge(bif_node, node_out, direction=prob)

    # add vessels to graph
    for vessel in patient_object.Topology.Vessels:
        for i in range(1, len(vessel.Nodes)):
            node1 = vessel.Nodes[i - 1]
            node2 = vessel.Nodes[i]
            p1 = node1.Pressure
            p2 = node2.Pressure
            dp = p1 - p2
            if dp == 0 or node2.Radius < radius:
                continue
            # exclude vessels (segments) that have a radius smaller than the clot fragment
            if dp > 0:
                # probability == 1 -> -log(1)=0
                # relative flow so P=1
                network.add_edge(node1, node2, direction=0)
            elif dp < 0:
                network.add_edge(node2, node1, direction=0)

    # set relative weights as the fractional flow (log sum and transform back)
    length, _ = nx.single_source_dijkstra(network, clot_node, weight='direction')
    probability = {k: np.exp(-v) for k, v in length.items()}

    # get end_nodes in the network, only those included in the (sub)network (reachable nodes)
    end_nodes = [x for x in network.nodes() if network.out_degree(x) == 0 and network.in_degree(x) >= 1]

    print(f"\tnumber of end_nodes:{len(end_nodes)}")
    # get probability at end_nodes in the network
    fragment_locations = {}
    for node in end_nodes:
        try:
            fragment_locations[node] = probability[node]
        except KeyError:
            pass
    #
    total_probability = sum([i for _, i in fragment_locations.items()])
    print(f"\tDijkstra's method, total probability:{total_probability}")
    if abs(total_probability-1) < 1e-6:
        return fragment_locations  # if sum p=1?

    print("\tAbsorbing Markov Chain")
    print(f"\033[91m\tWarning: This can require more than 40GB of RAM.\033[m")
    # speed up code by only considering reachable nodes
    # get reachable nodes
    downstream_nodes = nx.single_source_dijkstra(network, clot_node, weight='direction')
    reachable_vessel_nodes = list(downstream_nodes[0].keys())
    # reachable_vessel_nodes = list(network.nodes) # all nodes
    reachable_end_nodes = [node for node in end_nodes if node in reachable_vessel_nodes]
    [reachable_vessel_nodes.remove(x) for x in reachable_end_nodes]  # transient nodes

    number_transient = len(reachable_vessel_nodes)
    number_ends = len(reachable_end_nodes)
    print(f"\tNumber of reachable end_nodes:{number_ends}")
    print(f"\tNumber of reachable transient nodes:{number_transient}")

    print("\tCalculating Q")
    # Q = scipy.sparse.lil_matrix((number_transient, number_transient))

    # build matrix from list
    # Q_data = []
    Q_row_ind = []
    Q_col_ind = []

    for vessel in patient_object.Topology.Vessels:
        for i in range(1, len(vessel.Nodes)):
            node1 = vessel.Nodes[i - 1]
            node2 = vessel.Nodes[i]
            p1 = node1.Pressure
            p2 = node2.Pressure
            dp = p1 - p2
            # exclude vessels (segments) that have a radius smaller than the clot fragment
            if dp == 0 or node2.Radius < radius:
                continue
            if node1 in reachable_vessel_nodes and node2 in reachable_vessel_nodes:
                if dp > 0:
                    # Q[reachable_vessel_nodes.index(node1), reachable_vessel_nodes.index(node2)] = 1
                    Q_row_ind.append(reachable_vessel_nodes.index(node1))
                    Q_col_ind.append(reachable_vessel_nodes.index(node2))
                    # Q_data.append(1)
                elif dp < 0:
                    # Q[reachable_vessel_nodes.index(node2), reachable_vessel_nodes.index(node1)] = 1
                    Q_row_ind.append(reachable_vessel_nodes.index(node2))
                    Q_col_ind.append(reachable_vessel_nodes.index(node1))
                    # Q_data.append(1)

    Q_data = [1 for _ in Q_col_ind]

    for bif_node in patient_object.Topology.BifurcationNodes:
        # get total inflow into bif node
        inflow = sum([abs(node.FlowRate) for node in bif_node.Connections if node.Start * node.FlowRate > 0])
        bif_node.inflow = inflow

        bif_in = [node for node in bif_node.Connections if node.Start * node.FlowRate >= 0]
        bif_out = [node for node in bif_node.Connections if node.Start * node.FlowRate < 0]

        blocked_flow = sum([abs(node.FlowRate) for node in bif_node.Connections if
                            node.Start * node.FlowRate < 0 and node.Radius < radius])

        # use relative flow rates as probability
        for node_out in bif_out:
            if node_out.Radius < radius:
                node_out.P = 0
            else:
                node_out.P = abs(node_out.FlowRate) / (inflow - blocked_flow)
                if bif_node in reachable_vessel_nodes and node_out in reachable_vessel_nodes:
                    # Q[reachable_vessel_nodes.index(bif_node), reachable_vessel_nodes.index(node_out)] = node_out.P
                    Q_row_ind.append(reachable_vessel_nodes.index(bif_node))
                    Q_col_ind.append(reachable_vessel_nodes.index(node_out))
                    Q_data.append(node_out.P)

        for node_in in bif_in:
            node_in.P = 1
            if bif_node in reachable_vessel_nodes and node_in in reachable_vessel_nodes:
                # Q[reachable_vessel_nodes.index(node_in), reachable_vessel_nodes.index(bif_node)] = 1
                Q_row_ind.append(reachable_vessel_nodes.index(node_in))
                Q_col_ind.append(reachable_vessel_nodes.index(bif_node))
                Q_data.append(1)

    Q = scipy.sparse.csc_matrix((Q_data, (Q_row_ind, Q_col_ind)), shape=(number_transient, number_transient))

    print("\tMatrix invert: N = inv(I-Q)")
    start_time = datetime.now()
    N = inv(scipy.sparse.identity(number_transient, format="csc") - Q)
    # N = inv(scipy.sparse.identity(number_transient, format="csc") - Q.tocsc())
    time_elapsed = datetime.now() - start_time
    print('\t\tTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    print("\tCalculating R")
    # R = scipy.sparse.lil_matrix((number_transient, number_ends))
    start_time = datetime.now()
    # R_data = []
    R_row_ind = []
    R_col_ind = []

    for node in reachable_end_nodes:
        number_end_node = reachable_end_nodes.index(node)
        for con in list(node.Connections):
            if con in reachable_vessel_nodes:
                number_in = reachable_vessel_nodes.index(con)
                # R[number_in, number_end_node] = 1
                R_row_ind.append(number_in)
                R_col_ind.append(number_end_node)
                # R_data.append(1)

    R_data = [1 for _ in R_col_ind]

    R = scipy.sparse.csc_matrix((R_data, (R_row_ind, R_col_ind)), shape=(number_transient, number_ends))
    time_elapsed = datetime.now() - start_time
    print('\t\tTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    print("\tMatrix Multiplication: N*R")
    start_time = datetime.now()
    B = N * R  # matrix multiplication
    time_elapsed = datetime.now() - start_time
    print('\t\tTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    print("\tExtracting fragment probability")
    fragment_locations2 = {}
    starting_node = reachable_vessel_nodes.index(clot_node)
    for node in reachable_end_nodes:
        fragment_locations2[node] = B[starting_node, reachable_end_nodes.index(node)]

    print("\tFragment probability determined")
    print(f"\tDijkstra's method, total probability:{sum([i for _, i in fragment_locations.items()])}")
    print(f"\tAbsorbing Markov chains, total probability:{sum([i for _, i in fragment_locations2.items()])}")

    return fragment_locations2


def micro_occlusion(patient_object, thrombi_distribution, run_sim=False):
    # get fragment size
    # get probability of every node for fragment size

    # remove old clot
    patient_object.Topology.Clots = []

    node_thrombi = []
    for size, number in thrombi_distribution:
        # get all possible nodes for current fragment size
        possible_node = {}
        for node in patient_object.Topology.Nodes:
            node_probability = node.FragmentProbabilityFunction(size)
            if node_probability > 0:
                possible_node[node] = node_probability
        # randomly (weighted) pick a node to block (segment will be blocked)
        # weighted random sample with replacement
        population = list(possible_node.keys())
        weights = [possible_node[key] for key in population]
        # fragments can map to the same node, this will be a single occlusion
        selected_nodes = random.choices(population, weights=weights, k=number)
        # if bifurcation node, set upstream or downstream nodes (last two) to clot nodes
        # if vessel node, block selected and one before
        # add all clots to patient
        for node in set(selected_nodes):
            thrombi_count = selected_nodes.count(node)
            node_thrombi.append((node, size, thrombi_count))

        for node in set(selected_nodes):
            if node in patient_object.Topology.OutletNodes:
                # block outlet completely
                # todo replace with micro-occlusion model
                clot_node1 = node
                clot_node2 = list(node.Connections)[0]
                # patient_object.Topology.Clots.append([[clot_node1, clot_node2], 0, 0])
                # todo clots at outlets are micro-thrombi (smaller than these larger vessels)
            elif node in patient_object.Topology.BifurcationNodes:
                # block all upstream/downstream segments
                # get segments
                bifurcation_nodes = list(node.Connections)
                for vessel_end in bifurcation_nodes:
                    nodes = [vessel_end]
                    for con in vessel_end.Connections:
                        nodes.append(con)
                patient_object.Topology.Clots.append([nodes, 0, 0])
            else:
                # vessel node, block segment
                clot_node1 = node
                clot_node2 = list(node.Connections)[0]
                patient_object.Topology.Clots.append([[clot_node1, clot_node2], 0, 0])

    patient_object.WriteClotFile(name="Large_fragments.txt")  # save new clot nodes

    # run simulation
    if run_sim:
        patient_object.Initiate1DSteadyStateModel()
        if patient_object.ModelParameters["UsingPAfile"] == "True" and GeneralFunctions.is_non_zero_file(
                patient_object.Folders.ModellingFolder + "PAmapping.csv"):
            Collaterals.ImportPAnodes(patient_object)
            # if PAnode are present, run the model with the autoregulation method.
            CollateralsSimulation.collateral_simulation(patient_object, clot_active=True)
        else:
            patient_object.Run1DSteadyStateModel(model="Linear", tol=1e-9, clotactive=True, PressureInlets=True,
                                                 FlowRateOutlets=False, coarseCollaterals=False,
                                                 frictionconstant=patient_object.ModelParameters["FRICTION_C"])

        coupling_points_file = "ClusterFlowData.csv"
        vessel_result_file = "ResultsPerVessel.csv"
        topology_file = "Topology.vtp"

        patient_object.Results1DSteadyStateModel()
        patient_object.Export1DSteadyClusterFlowRate(file=coupling_points_file)
        patient_object.ExportMeanResults(file=vessel_result_file)
        patient_object.Results.AddResultsPerNodeToFile(patient_object.Folders.ModellingFolder + topology_file)
        patient_object.Results.AddResultsPerVesselToFile(patient_object.Folders.ModellingFolder + topology_file)

    filename = patient_object.Folders.ModellingFolder + "Micro-thrombi.csv"
    outlet_thrombi = []
    for node, size, thrombi_count in node_thrombi:
        if node in patient_object.Topology.OutletNodes:
            outlet_thrombi.append(node)
            try:
                node.micro_thrombi[size] = thrombi_count
            except: # todo update for specifc exception
                node.micro_thrombi = {size: thrombi_count}

    with open(filename, "w") as f:
        f.write("Location,Distribution\n")
        for node in set(outlet_thrombi):
            pos = node.Position
            thrombi = node.micro_thrombi
            f.write("\"%s\",\"%s\"\n" % (pos, thrombi))

    clot_nodes = [node for sublist in patient_object.Topology.Clots for node in sublist[0]] # large fragments
    clot_nodes += [node for node in set(outlet_thrombi)] # small fragments
    occlusions = [1 if node in clot_nodes else 0 for node in patient_object.Topology.Nodes]
    GeneralFunctions.AddArrayToFile(patient_object.Folders.ModellingFolder + "Topology.vtp", occlusions, "Clot", False)


def micro_thrombi(folder, result_file="Results.dyn"):
    # build probability functions
    # anything below the smallest vessel just passes through the system. Probability is then the fraction of flow compared to the start location.
    # patient_folder = "/home/raymond/Desktop/1d-blood-flow/Generated_Patients/Fragment_collaterals/"
    clot_sizes = [0, 0.1, 0.2, 0.3, 0.45, 0.7, 0.9, 1.0, 1.2, 1.3, 1.4]  # (radius) sizes used to build the probability functions (must be less than the initial vessel)
    patient = fragment_simulation(folder, clot_sizes, result_file, plot_example=False, save_optional_results=False)

    # load distribution from file
    file = patient.Folders.PatientFolder+"micro_thrombi_distribution.csv"
    data = [line.strip('\n').split(',') for line in open(file)][1:]
    micro_thrombi_distribution = [(float(size)*1e-3/2, int(number)) for number, size in data if int(number) > 0] # convert to radius in mm
    # number_thrombi = sum([number for _, number in micro_thrombi_distribution])

    # micro_thrombi_distribution = [[0.1, 100], [0.3, 10], [0.45, 3], [0.7, 1], [0.9, 2]]  # clot_size (radius), number
    # determine location and number of micro-thrombi in the network
    micro_occlusion(patient, micro_thrombi_distribution)


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    patient_folder = arguments["<patient_folder>"]
    results_file = arguments["<file>"] if arguments["<file>"] is not None else "Results.dyn"
    micro_thrombi(patient_folder, results_file)
