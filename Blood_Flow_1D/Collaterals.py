#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Functions related to the pial network model.
"""
import contextlib
import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.sparse
import scipy.spatial
import tqdm
import vtk
from mgmetis import metis

from Blood_Flow_1D import Node, Vessel, Topology, GeneralFunctions, BloodFlowEquations, Patient, Results, Perfusion


def overlapregion(patient, areaincrease):
    """
    Add triangles to the clusters. Add the closest triangle until we have increased the area with the input percentage
    Calculate the overlap by finding connected triangles
    :param patient: patient object
    :param areaincrease: percentage of increase in area
    :return: Nothing
    """
    dualgraph = patient.Perfusion.DualGraph

    for indexcp, couplingpoint in enumerate(patient.Perfusion.CouplingPoints):
        clusterids = [indexcp if i == indexcp else -1 for i in patient.Perfusion.DualGraph.NodeColour]
        newarea = 0
        while newarea < couplingpoint.Area * (areaincrease / 100):
            connectednodes = couplingpoint.SurfaceNodes

            # For the connected triangles, calculate the distance to its center.
            boundarynodes = []
            for item in connectednodes:
                linkednodes = [i for i in dualgraph.Links[item] if clusterids[i] != indexcp]
                if len(linkednodes) > 0:
                    for p in linkednodes:
                        distancetolinkednodes = GeneralFunctions.distancebetweenpoints(dualgraph.PialSurface[item],
                                                                                       dualgraph.PialSurface[p])
                        boundarynodes.append((distancetolinkednodes, p))

            # add triangles to the set, starting at the nearest one.
            sortedboundarynodes = sorted(boundarynodes, key=lambda k: k[0])
            processednodes = set()
            for element in sortedboundarynodes:
                # if the extra area is large enough, stop adding triangles, else add another one
                if newarea > couplingpoint.Area * (areaincrease / 100):
                    break
                if element[1] not in processednodes:
                    processednodes.add(element[1])
                    area = patient.Perfusion.PrimalGraph.Areas[element[1]]
                    newarea += area
                    # print(newarea)
                    couplingpoint.SurfaceNodes.append(element[1])
                    clusterids[element[1]] = indexcp
                    # print(element[0])

    oldarea = [couplingpoint.Area for indexcp, couplingpoint in enumerate(patient.Perfusion.CouplingPoints)]
    newarea = [sum([patient.Perfusion.PrimalGraph.Areas[i] for i in cp.SurfaceNodes]) for cp in
               patient.Perfusion.CouplingPoints]
    for index, area in enumerate(oldarea):
        extraregion = newarea[index] / area - 1
        print(str(area) + ":" + str(newarea[index]) + " Increased %f:" % extraregion)

    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        cp.Area = sum([patient.Perfusion.PrimalGraph.Areas[i] for i in cp.SurfaceNodes])


def ExportOverlappingRegions(patient):
    """
    Export the overlapping regions to a file.
    :param patient: patient object
    :return: Regions.vtp and SurfaceNodesMapping.csv
    """
    print("Exporting regions.")
    nodes = vtk.vtkPoints()
    graph = patient.Perfusion.PrimalGraph
    for index, item in enumerate(graph.PialSurface):
        nodes.InsertNextPoint(item)

    Colour = vtk.vtkFloatArray()
    Colour.SetNumberOfComponents(1)
    Colour.SetName("Colour")
    for colour in graph.PolygonColour:
        Colour.InsertNextValue(colour)

    VesselsPolyData = vtk.vtkPolyData()
    VesselsPolyData.SetPoints(nodes)
    VesselsPolyData.SetPolys(graph.Polygons)
    VesselsPolyData.GetCellData().AddArray(Colour)

    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        Colour = vtk.vtkIntArray()
        Colour.SetName("Region %d" % index)
        for index2, colour in enumerate(graph.PolygonColour):
            if index2 in cp.SurfaceNodes:
                if index == colour:
                    Colour.InsertNextValue(1)
                else:
                    Colour.InsertNextValue(2)
            else:
                Colour.InsertNextValue(0)
        VesselsPolyData.GetCellData().AddArray(Colour)

    file = patient.Folders.ModellingFolder + "Regions.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    print("Writing file: %s" % file)
    writer.SetFileName(file)
    writer.SetInputData(VesselsPolyData)
    writer.Write()

    trianglesmapping = [[] for i in patient.Perfusion.PrimalGraph.Triangles]
    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        for triangle in cp.SurfaceNodes:
            trianglesmapping[triangle].append(index)

    file = patient.Folders.ModellingFolder + "SurfaceNodesMapping.csv"
    print("Writing surface node mapping to file: %s" % file)
    with open(file, 'w') as f:
        f.write("TriangleNumber,ClusterID\n")
        for index, item in enumerate(trianglesmapping):
            line = ','.join(map(str, item))
            f.write(str(index) + "," + line + "\n")


def CreatePialNetwork(surfacefile, separation="majorregions", collateralscore=1):
    """
    Determine the dual graph as the pial network and set regions based on clustering
    :param surfacefile: The primary graph to use as a template for the pial network.
    :param separation: Method to determine where the collateral vessels are. Default: majorregions, (other option:clusters)
    :param collateralscore: probability to include any particular collateral vessel in the pial network.
    :return: perfusion object, node positions, edges, collateral edges
    """
    patienttemp = Perfusion.Perfusion()
    patienttemp.PrimalGraph.LoadSurface(surfacefile)
    patienttemp.SetDualGraph(method="edges")
    patienttemp.DualGraph.File = "PialNetworkCluster.vtp"
    patienttemp.DualGraph.NodeColour = patienttemp.PrimalGraph.PolygonColour

    nodepos = patienttemp.DualGraph.PialSurface
    links = patienttemp.DualGraph.Links
    edges = []
    for index, link in enumerate(links):
        for linkednode in link:
            if index < linkednode:
                edges.append((index, linkednode))

    # separation based on regions or clusters
    if separation == "majorregions":
        identification = patienttemp.PrimalGraph.map
    elif separation == "clusters":
        identification = patienttemp.PrimalGraph.PolygonColour
    else:
        identification = np.zeros(len(patienttemp.DualGraph.PialSurface))

    newedges = []
    colleratalEdges = []
    edgeskept = 0
    for edge in edges:
        c1 = identification[edge[0]]
        c2 = identification[edge[1]]
        if c1 == c2:
            newedges.append(edge)
        elif random.random() < collateralscore:
            # add the edge with a certain change based on the collateral score
            edgeskept += 1
            newedges.append(edge)
            colleratalEdges.append(edge)

    print(f"Number of vessels before removal: {len(edges)}")
    edges = newedges
    print(f"Number of vessels after removal: {len(edges)}")
    print(f"Number of collateral vessels: {edgeskept}")

    return patienttemp, nodepos, edges, colleratalEdges


def GenerateVessels(patient, nodepositions, edges, colleratalEdges, PialnetworkRadius=0.2, Connecting="Tree"):
    """
    Generate simple vessels for the pial network. Every vertix is a node and every edge is a vessel.
    :param patient: patient object
    :param nodepositions: 3d positions of the pial network nodes
    :param edges: edges of the pial network
    :param colleratalEdges: edged that belong to the collateral system
    :param PialnetworkRadius: radius of every vessel of the pial network
    :return: list of collateral vessels, pial network nodes, pial network vessels and nodes that belong to the edges.
    """
    # we have lists of positions and edges
    # we want to make these into nodes and vessels

    # it seems some nodes have no connections?
    pialnetworknodes = [Node.Node() for _ in nodepositions]
    for pos, node in zip(nodepositions, pialnetworknodes):
        node.SetPosition(pos)
        node.SetRadius(PialnetworkRadius)

    nodestoInclude = set()

    collateralvessels = []
    pialnetworkvessels = [Vessel.Vessel() for _ in edges]
    for vessel, edge in zip(pialnetworkvessels, edges):
        vessel.SetType(4)
        vessel.SetNodes([pialnetworknodes[edge[0]], pialnetworknodes[edge[1]]])
        pialnetworknodes[edge[0]].AddConnection(pialnetworknodes[edge[1]])
        pialnetworknodes[edge[1]].AddConnection(pialnetworknodes[edge[0]])
        nodestoInclude.add(pialnetworknodes[edge[0]])
        nodestoInclude.add(pialnetworknodes[edge[1]])
        if edge in colleratalEdges:
            collateralvessels.append(vessel)

    # Add vessel name and ID for the pial network
    [vessel.SetName(f"PialNetworkVessel {index}") for index, vessel in enumerate(pialnetworkvessels)]
    lastID = patient.Topology.Vessels[-1].ID
    [vessel.SetID(lastID + 1 + index) for index, vessel in enumerate(pialnetworkvessels)]

    if patient.Topology.NumberOfPenetratingArteries > 0:
        patient.Topology.PAnodes = pialnetworknodes[-patient.Topology.NumberOfPenetratingArteries:]
    else:
        patient.Topology.PAnodes = []
    print(f"Number of penetrating arteries: {len(patient.Topology.PAnodes)}")

    if Connecting == "CouplingPoints":
        # connect pial network to coupling points!
        for cp in patient.Perfusion.CouplingPoints:
            node = cp.Node
            if pialnetworknodes[cp.PialSurfacePointID] in nodestoInclude:
                node.AddConnection(pialnetworknodes[cp.PialSurfacePointID])
                pialnetworknodes[cp.PialSurfacePointID].AddConnection(node)
            else:
                for othernode in node.SurfaceNodes:
                    if pialnetworknodes[othernode] in nodestoInclude:
                        node.AddConnection(pialnetworknodes[othernode])
                        pialnetworknodes[othernode].AddConnection(node)
                        break

    if Connecting == "Tree":
        # connect tree and pial network
        for tree in patient.Trees:
            for node in tree.EndNodes:
                if pialnetworknodes[node.SurfaceMappingNode] in nodestoInclude:
                    node.AddConnection(pialnetworknodes[node.SurfaceMappingNode])
                    pialnetworknodes[node.SurfaceMappingNode].AddConnection(node)
                else:
                    for othernode in node.SurfaceMappingTriangles:
                        if pialnetworknodes[othernode] in nodestoInclude:
                            node.AddConnection(pialnetworknodes[othernode])
                            pialnetworknodes[othernode].AddConnection(node)
                            break

    # only add the nodes that we need
    pialnetworknodes = list(nodestoInclude)
    patient.Topology.Nodes.extend(pialnetworknodes)
    patient.Topology.Vessels.extend(pialnetworkvessels)
    patient.Topology.OutletNodes.extend(patient.Topology.PAnodes)

    for cp in patient.Perfusion.CouplingPoints:
        patient.Topology.OutletNodes.remove(cp.Node)

    for index, node in enumerate(patient.Topology.PAnodes):
        node.PAnumber = index

    for vessel in collateralvessels:
        vessel.SetType(5)

    return collateralvessels, pialnetworknodes, pialnetworkvessels


def SetPAOutlets(patient, brainperfusionestimate=12.5e-6, pialsurfacepressure=7333, venouspressure=2500):
    """
    Calculate the resistance for the penerating arteries and assign it.
    :param patient: patient object
    :param brainperfusionestimate: estimate of the volume flow rate to the entire brain (through the PAs)
    :param pialsurfacepressure: pressure at the pial surface in pa.
    :param venouspressure: venous pressure at the start of the venous system.s
    :return: None.
    """
    pressurediff = pialsurfacepressure - venouspressure
    if patient.Topology.NumberOfPenetratingArteries > 0:
        ResistancePA = pressurediff / (brainperfusionestimate / patient.Topology.NumberOfPenetratingArteries)
        print(f"Resistance of a single PA: {ResistancePA}")
    else:
        return 0

    for index, node in enumerate(patient.Topology.PAnodes):
        #     node.InputPressure = 7333 # option to use constant pressure
        node.R1 = 0
        node.R2 = ResistancePA
        node.C = 0


def PAfromFile(file, nodes):
    """
    Load a surface file and use the vertices as the location of the penetrating arteries.
    Map the penetrating arteries to the nearest pial network node.
    :param file: surface file
    :param nodes: nodes of the pial network
    :return: list of penetrating artery position, edges between the PAs and the nodes.
    """
    patienttemp = Perfusion.Perfusion()
    patienttemp.PrimalGraph.LoadSurface(file)
    penetratingartery = patienttemp.PrimalGraph.PialSurface

    KDTree = scipy.spatial.KDTree(nodes)

    connectionsPA = []
    for index in range(0, len(penetratingartery)):
        number = index + len(nodes)
        _, MinDistanceIndex = KDTree.query(penetratingartery[index], k=1)
        connectionsPA.append((MinDistanceIndex, number))

    return penetratingartery, connectionsPA


def GeneratePAs(graph, n=100000):
    """
    Generate penetrating arteries on a surface uniformly.
    :param graph: Graph object, surface used to generate the penetrating arteries.
    :param n: number of penetrating arteries to generate.
    :return: position of the penetrating artery, connections to the nearest triangle of the surface.
    """
    weights = np.array(graph.Areas)
    prob = weights / weights.sum()
    sample = np.random.choice(range(0, len(graph.Triangles)), n, replace=True, p=prob)
    # sample = np.random.choice(range(0,10000,100), n, replace=True)

    uniformdistribution = np.random.uniform(0, 1, (len(sample), 2))

    penetratingartery = []  # list of positions within a triangle
    for index, sampledtriagle in enumerate(sample):
        triangle = graph.Triangles[sampledtriagle]
        v1 = graph.PialSurface[triangle[1]] - graph.PialSurface[
            triangle[0]]
        v2 = graph.PialSurface[triangle[2]] - graph.PialSurface[
            triangle[0]]

        if uniformdistribution[index][0] > 0 and uniformdistribution[index][1] > 0 and uniformdistribution[index][
            0] + uniformdistribution[index][1] < 1:
            x = graph.PialSurface[triangle[0]] + uniformdistribution[index][0] * v1 + \
                uniformdistribution[index][1] * v2
        else:
            x = graph.PialSurface[triangle[0]] + (1 - uniformdistribution[index][0]) * v1 + \
                (1 - uniformdistribution[index][1]) * v2

        penetratingartery.append(x)

    # save the connecting vessels
    connectionsPA = []
    for index, sampledtriagle in enumerate(sample):
        numb = index + len(graph.Triangles)  # number of the generated point
        # sampled triangle is the node that connects to the generated point
        connectionsPA.append((sampledtriagle, numb))

    return penetratingartery, connectionsPA


def SplitRegionInTwo(points, PialDMat, debug=False):
    """
    Split a set of points in two based on distance.
    :param points: points to divide
    :param PialDMat: distance matrix
    :param debug: Boolean to set debu mode. Debug mode randomly splits the region in two.
    :return: Region1, Region2
    """
    print("Splitting region in two.")
    RootsN = 2
    PointsN = len(points)
    NRootsLabels = [np.floor(PointsN / 2), np.ceil(PointsN / 2)]
    regionpoints = points
    roots = np.random.choice(points, 2, replace=False)
    sampledict = {sample: index for index, sample in enumerate(points)}

    if debug:
        region1 = np.random.choice(points, int(NRootsLabels[0]), replace=False)
        region2 = [point for point in points if point not in region1]
        return region1, region2

    # DistanceMat = dualgraph.CalculateDistanceMatRegion("", roots, regionpoints)
    # DistanceMat2 = PialDMat[[sampledict[root] for root in roots],:]
    iter = 0
    result = []
    while 1:
        DistanceMat = PialDMat[[sampledict[root] for root in roots], :]

        flat_list = [item for sublist in DistanceMat for item in sublist]
        AssignedLabelsN = [0 for _ in range(0, RootsN)]
        Labels = [-1 for _ in range(0, PointsN)]

        sortedindex = sorted(range(len(flat_list)), key=lambda k: flat_list[k])

        for index in sortedindex:
            ClusterIndex = index // PointsN
            PointIndex = index % PointsN

            if AssignedLabelsN[ClusterIndex] < NRootsLabels[ClusterIndex] and Labels[PointIndex] == -1:
                # if Labels[PointIndex] == -1:
                Labels[PointIndex] = ClusterIndex
                AssignedLabelsN[ClusterIndex] += 1

        swapiter = 0
        while 1:
            swaplist = []
            for point in range(0, PointsN):
                possiblelabels = DistanceMat[:, point]
                sortedlabels = sorted(range(len(possiblelabels)), key=lambda k: possiblelabels[k])
                currentlabel = Labels[point]
                if currentlabel != sortedlabels[0]:
                    swaplist.append(point)
            changecounter = 0
            for swap in swaplist:
                differenceafterswap = [(DistanceMat[Labels[swap], swap] + DistanceMat[Labels[i], i])
                                       - (DistanceMat[Labels[i], swap] + DistanceMat[Labels[swap], i])
                                       for i in range(0, PointsN)]
                differencebyindex = max(range(len(differenceafterswap)), key=lambda k: differenceafterswap[k])
                if differenceafterswap[differencebyindex] > 0:
                    Labels[differencebyindex], Labels[swap] = Labels[swap], Labels[differencebyindex]
                    changecounter += 1
            print("Total changes: %d" % changecounter)
            swapiter += 1
            if changecounter == 0 or swapiter == 10:
                break

        stopcrit = 0

        print("Recalculating roots.")
        clusterelements = [[i for i in range(0, len(Labels)) if Labels[i] == element] for element
                           in range(0, RootsN)]

        for index, cluster in enumerate(clusterelements):
            # if method == "euclidean":
            # newpos = np.mean([dualgraph.PialSurface[regionpoints[i]] for i in cluster], axis=0)
            # diff = distancebetweenpoints(roots[index], newpos)
            # stopcrit += diff
            # roots[index] = newpos

            distancematforcluster = np.empty([len(cluster), len(cluster)])
            for num, point in enumerate(cluster):
                for num2, point2 in enumerate(cluster):
                    distancematforcluster[num, num2] = PialDMat[point, point2]
            print(distancematforcluster)
            distances = np.amax(distancematforcluster, axis=0)
            newguess = min(range(0, len(distances)), key=lambda k: distances[k])
            newguessPoint = cluster[newguess]
            # new guess for root node is the Jardan Center of the cluster
            # calculate the change in position of the root
            # newpos = PialSurface[regionpoints[newguessPoint]]
            # oldpos = PialSurface[roots[index]]
            # diff = GeneralFunctions.distancebetweenpoints(oldpos, newpos)
            diff = abs(regionpoints[newguessPoint] - roots[index])
            stopcrit += diff
            roots[index] = regionpoints[newguessPoint]
            # print(roots[index])
        print("Root displacement: %f" % stopcrit)
        iter += 1
        # print(rootpositions)
        if stopcrit == 0 or iter == 20:
            result = Labels
            break

        # DistanceMat = dualgraph.CalculateDistanceMatRegion("", roots, regionpoints)

    region1 = [points[i] for i in range(0, len(result)) if result[i] == 0]
    region2 = [points[i] for i in range(0, len(result)) if result[i] == 1]
    return region1, region2


def TreeToSurfaceMapping(patient, distancemat, debug=False, method="dijkstra"):
    """
    Map bifurcating trees to the pial surface by iterative division of the regions.
    Updated method at TreeToSurfaceMappingMetis()
    :param patient: patient object
    :param distancemat: distance matrix
    :param debug: dBoolean to set debu mode. Debug mode randomly splits the region in two.
    :param method: distance method.
    :return: Nothing
    """
    print("Mapping trees to the surface.")
    # Connect the bifurcating trees to the pial network
    patient.Perfusion.DualGraph.CalculateScipyGraph()
    # graph = patient.Perfusion.DualGraph.ScipyGraph

    samples = [cp.SurfaceNodes for cp in patient.Perfusion.CouplingPoints]
    print("Computing distance matrices.")

    if method == "Euclidean" or method == "euclidean":
        pbar = tqdm.tqdm(total=len(samples))

        def pialmat2(sample):
            rootspos = np.array([patient.Perfusion.DualGraph.PialSurface[point] for point in sample])
            mat = scipy.spatial.distance.cdist(rootspos, rootspos, 'euclidean')
            pbar.update(1)
            return mat

        mat = [pialmat2(i) for i in samples]
        pbar.close()
    else:
        if GeneralFunctions.is_non_zero_file(distancemat):
            print("Loading Distance Matrix.")
            mat = np.memmap(distancemat, dtype='float32', mode='r',
                            shape=(len(patient.Perfusion.DualGraph.PialSurface),
                                   len(patient.Perfusion.DualGraph.PialSurface)))
        else:
            mat = np.memmap(distancemat, dtype='float32', mode='w+',
                            shape=(len(patient.Perfusion.DualGraph.MajorVesselID),
                                   len(patient.Perfusion.DualGraph.MajorVesselID)))
            print("Allocated diskspace for the matrix.")
            # CompleteDijkstraMap = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False)
            pbar = tqdm.tqdm(total=len(patient.Perfusion.DualGraph.MajorVesselID))
            for i in range(0, len(patient.Perfusion.DualGraph.MajorVesselID)):
                mat[:, i] = scipy.sparse.csgraph.dijkstra(patient.Perfusion.DualGraph.ScipyGraph, directed=False,
                                                          indices=i)
                pbar.update(1)
            pbar.close()

    print("Calculating splits of the clusters iteratively.")
    # Serial code
    # splits = []
    # for index, cp in enumerate(patient.Perfusion.CouplingPoints):
    #     print("Region %d of %d" % (index + 1, len(patient.Perfusion.CouplingPoints)))
    #     # numbersplits = int(
    #     #     np.log2(2 * (int(np.ceil(patient.Perfusion.CouplingPoints[index].NumberOfPialPoints / 2)))))
    #     treeoutlets = len(patient.Trees[index].EndNodes)
    #     numbersplits = int(
    #         np.log2(2 * (int(np.ceil(treeoutlets / 2)))))
    #
    #     samplednodes = patient.Perfusion.CouplingPoints[index].SurfaceNodes
    #     # samplednodes = patient.Perfusion.CouplingPoints[index].Pialpoints
    #     sampledict = {sample: index for index, sample in enumerate(samplednodes)}
    #     split = [[samplednodes]]
    #     if numbersplits > 0:
    #         # PialDMat = scipy.sparse.csgraph.dijkstra(patient.Perfusion.DualGraph.ScipyGraph, directed=False,
    #         #                                          indices=samplednodes)
    #         # PialDMat = mat[index]
    #         for i in range(0, numbersplits):
    #             currentsplit = []
    #             for region in split[-1]:
    #                 regionids = [sampledict[i] for i in region]
    #                 with contextlib.redirect_stdout(None):
    #                     region1, region2 = patient.Perfusion.SplitRegionInTwo(patient.Perfusion.DualGraph, region,
    #                                                                           mat[index][:, regionids])
    #                 # print(region1)
    #                 # print(region2)
    #                 currentsplit.append(region1)
    #                 currentsplit.append(region2)
    #             split.append(currentsplit)
    #     else:
    #         split.append(samplednodes)
    #     splits.append(split)

    # multiprocessing
    # pialsurface = patient.Perfusion.DualGraph.PialSurface
    args = []
    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        treeoutlets = len(patient.Trees[index].EndNodes)
        numbersplits = int(
            np.log2(2 * (int(np.ceil(treeoutlets / 2)))))

        samplednodes = patient.Perfusion.CouplingPoints[index].SurfaceNodes
        # samplednodes = patient.Perfusion.CouplingPoints[index].Pialpoints
        # sampledict = {sample: index for index, sample in enumerate(samplednodes)}

        arg = (numbersplits, samplednodes, mat[index])
        args.append(arg)

    def splittingregions(args):
        numbersplits, samplednodes, mat = args
        sampledict = {sample: index for index, sample in enumerate(samplednodes)}
        split = [[samplednodes]]

        if numbersplits > 0:
            for i in range(0, numbersplits):
                currentsplit = []
                for region in split[-1]:
                    regionids = [sampledict[i] for i in region]
                    with contextlib.redirect_stdout(None):
                        region1, region2 = SplitRegionInTwo(region, mat[:, regionids], debug)

                    currentsplit.append(region1)
                    currentsplit.append(region2)
                split.append(currentsplit)
        else:
            split.append(samplednodes)
        return split

    splits = list(tqdm.tqdm(Patient.Patient.pool.imap(splittingregions, args), total=len(args)))

    # Map the surface nodes to the trees
    # maxsplits = max([len(i) for i in splits])
    # colours = [[-1 for i in patient.Perfusion.DualGraph.PialSurface] for i in range(0, maxsplits)]
    #
    # for split in splits:
    #     for index2, regions in enumerate(split):
    #         for indexregion, region in enumerate(regions):
    #             for element in region:
    #                 colours[index2][element] = indexregion

    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        for indexnode, node in enumerate(cp.Tree.EndNodes):
            possibletriangles = splits[index][-1][indexnode]
            selectedtriangle = np.random.choice(possibletriangles, 1)[0]
            pos = patient.Perfusion.DualGraph.PialSurface[selectedtriangle]
            node.SurfaceMappingNode = selectedtriangle
            node.SurfaceMapping = pos
            node.SurfaceMappingTriangles = possibletriangles

    patient.Perfusion.RegionSplits = splits


def TreeToSurfaceMappingMetis(patient):
    """
    Split the perfusion territories iteratively to map the bifurcating trees to the surface.
    The regions are split iteratively so that ends close together remain close on the surface.
    The center final region of each treeis set as the coupling point.
    :param patient: patient object
    :return: None
    """
    print("Calculating splits of the clusters iteratively.")

    args = []
    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        treeoutlets = len(patient.Trees[index].EndNodes)
        numbersplits = int(
            np.log2(2 * (int(np.ceil(treeoutlets / 2)))))

        samplednodes = patient.Perfusion.CouplingPoints[index].SurfaceNodes

        arg = (numbersplits, samplednodes)
        args.append(arg)

    # from mgmetis.enums import OPTION
    # opts = metis.get_default_options()
    # opts[OPTION.NUMBERING] = 0
    # opts[OPTION.DBGLVL] = 1
    def splittingregions(args):
        numbersplits, samplednodes = args
        split = [[samplednodes]]
        if numbersplits > 0:
            for i in range(0, numbersplits):
                currentsplit = []
                for region in split[-1]:
                    cells = [e for e in patient.Perfusion.DualGraph.Graph.subgraph(region).edges]
                    # graph needs to start at 0 or 1 for metis.
                    cells = [(region.index(e[0]), region.index(e[1])) for e in cells]
                    if len(cells) < 1:
                        print("Not enough cells to divide.\n")
                        print("Resolution of the graph is likely too low or the tree radius too small.\n")
                    areas = GeneralFunctions.slice_by_index(patient.Perfusion.DualGraph.Weights, region)
                    objval, epart, npart = metis.part_mesh_nodal(2, cells, vwgt=areas)  # , options=opts)

                    region1 = [i for i, j in zip(region, npart) if j == npart.max()]
                    region2 = [i for i, j in zip(region, npart) if j == npart.min()]

                    currentsplit.append(region1)
                    currentsplit.append(region2)
                split.append(currentsplit)
        else:
            split.append(samplednodes)
        return split

    pbar = tqdm.tqdm(total=len(args))

    def pialmat2(sample):
        mat = splittingregions(sample)
        pbar.update(1)
        return mat

    splits = [pialmat2(i) for i in args]
    pbar.close()

    # splits = list(tqdm.tqdm(Patient.Patient.pool.imap(splittingregions, args), total=len(args)))

    # Map the surface nodes to the trees
    # maxsplits = max([len(i) for i in splits])
    # colours = [[-1 for i in patient.Perfusion.DualGraph.PialSurface] for i in range(0, maxsplits)]
    #
    # for split in splits:
    #     for index2, regions in enumerate(split):
    #         for indexregion, region in enumerate(regions):
    #             for element in region:
    #                 colours[index2][element] = indexregion
    print("Mapping to nearest triangle.")
    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        for indexnode, node in enumerate(cp.Tree.EndNodes):
            possibletriangles = splits[index][-1][indexnode]

            # random triangle selection
            # selectedtriangle = np.random.choice(possibletriangles, 1)[0]

            # center triangle
            pos = np.array([patient.Perfusion.DualGraph.PialSurface[i] for i in possibletriangles])
            distancematforcluster = scipy.spatial.distance.cdist(pos, pos)
            distances = np.amax(distancematforcluster, axis=0)
            newguess = min(range(0, len(distances)), key=lambda k: distances[k])
            selectedtriangle = possibletriangles[newguess]
            # isolatednetwork = patient.Perfusion.DualGraph.Graph.subgraph(possibletriangles)
            # selectedtriangle = nx.center(isolatednetwork)[0]

            # print(selectedtriangle)

            pos = patient.Perfusion.DualGraph.PialSurface[selectedtriangle]
            node.SurfaceMappingNode = selectedtriangle
            node.SurfaceMapping = pos
            node.SurfaceMappingTriangles = possibletriangles

    patient.Perfusion.RegionSplits = splits

    print("Updating coupling points.")
    # update coupling points and surface mapping
    newcp = []
    for tree in patient.Trees:
        for node in tree.EndNodes:
            couplingpoint = Perfusion.CouplingPoint(node)
            # node.SetMajorVesselID()
            # couplingpoint.Area = surface.DualGraph.Areas[index]
            couplingpoint.SurfaceNodes = node.SurfaceMappingTriangles
            couplingpoint.NumberOfTriangles = len(node.SurfaceMappingTriangles)
            couplingpoint.PialSurfacePointID = node.SurfaceMappingNode
            # new clustering map
            for surfacenode in couplingpoint.SurfaceNodes:
                patient.Perfusion.PrimalGraph.PolygonColour[surfacenode] = len(newcp)
                patient.Perfusion.DualGraph.NodeColour[surfacenode] = len(newcp)
            newcp.append(couplingpoint)

    patient.Perfusion.CouplingPoints = newcp
    patient.Topology.UpdateTopology()


def ExportTreeEndNodePositions(patient, filename="TreeEndNodePositions.vtp"):
    """
    Export the location of the trees on the pial surface. The coupling point number (tree number) are included.
    :param patient: patient object
    :param filename: exported vtp file
    :return: Nothing
    """
    print("Exporting bifurcating tree endnode positions to file.")
    nodes = vtk.vtkPoints()
    treenumber = vtk.vtkIntArray()
    treenumber.SetNumberOfComponents(1)
    treenumber.SetName("Tree Number")

    # for index, cp in enumerate(patient.Perfusion.CouplingPoints):
    #     for indexnode, node in enumerate(cp.Tree.EndNodes):
    #         nodes.InsertNextPoint(node.SurfaceMapping)
    #         treenumber.InsertNextValue(index)

    for index, tree in enumerate(patient.Trees):
        for indexnode, node in enumerate(tree.EndNodes):
            nodes.InsertNextPoint(node.SurfaceMapping)
            treenumber.InsertNextValue(index)

    VesselsPolyData = vtk.vtkPolyData()
    VesselsPolyData.SetPoints(nodes)
    VesselsPolyData.GetPointData().AddArray(treenumber)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(patient.Folders.ModellingFolder + filename)
    writer.SetInputData(VesselsPolyData)
    writer.Write()


def WriteRegionsSplitsToVTP(patient, file="RegionSplitting.vtp"):
    """
    Export the split regions to a file.
    :param patient: patient object
    :param file: default:"RegionSplitting.vtp"
    :return: Nothing
    """
    print("Writing split regions to file: %s" % file)
    splits = patient.Perfusion.RegionSplits
    maxsplits = max([len(i) for i in splits])
    colours = [[-1 for _ in patient.Perfusion.DualGraph.PialSurface] for _ in range(0, maxsplits)]

    for split in splits:
        for index2, regions in enumerate(split):
            for indexregion, region in enumerate(regions):
                for element in region:
                    colours[index2][element] = indexregion

    for split in splits:
        for index2, regions in enumerate(split):
            # index2  # split number
            for indexregion, region in enumerate(regions):
                for element in region:
                    colours[index2][element] = indexregion

    nodes = vtk.vtkPoints()
    for index, item in enumerate(patient.Perfusion.DualGraph.PialSurface):
        nodes.InsertNextPoint(item)

    coloursvtk = [vtk.vtkIntArray() for _ in range(0, maxsplits)]
    [i.SetNumberOfComponents(1) for i in coloursvtk]
    [i.SetName("Split %d" % index) for index, i in enumerate(coloursvtk)]

    for index, col in enumerate(colours):
        for item in col:
            coloursvtk[index].InsertNextValue(item)

    finalsplit = vtk.vtkIntArray()
    finalsplit.SetNumberOfComponents(1)
    finalsplit.SetName("Split final")
    finalregions = [split[-1] for split in splits]
    flat_list = [item for sublist in finalregions for item in sublist]
    label = [-1 for _ in patient.Perfusion.DualGraph.PialSurface]
    for index, region in enumerate(flat_list):
        for id in region:
            label[id] = index
    for item in label:
        finalsplit.InsertNextValue(item)

    VesselsPolyData = vtk.vtkPolyData()
    VesselsPolyData.SetPoints(nodes)
    [VesselsPolyData.GetPointData().AddArray(i) for i in coloursvtk]
    VesselsPolyData.GetPointData().AddArray(finalsplit)

    # file = patient_folder + "RegionSplitting.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(VesselsPolyData)
    writer.Write()


def RemoveUnconnectedSections(patient):
    """
    Filter vessels outside main network.
    Note that the assumption at this time is that the inlet is node 0.
    :param patient: patient object
    :return: None.
    """
    patient.Topology.TopologyToGraph()
    # assume node 0 is the inlet node.
    length, path = nx.single_source_dijkstra(patient.Topology.Graph, patient.Topology.Nodes[0])

    NodesNotInNetwork = []
    for node in patient.Topology.Nodes:
        if node not in length:
            NodesNotInNetwork.append(node)

    for node in NodesNotInNetwork:
        if node in patient.Topology.BifurcationNodes:
            patient.Topology.BifurcationNodes.remove(node)
            # print(node)
    VesselIDsToDelete = set([node.VesselID for node in NodesNotInNetwork])

    # remap PA so that they are all included
    # if vessel contains PA node, update the other end (bif node) to a new (bif) node and update all positions etc accordingly.
    VesselsToDelete = set([vessel for vessel in patient.Topology.Vessels if vessel.ID in VesselIDsToDelete])
    # OtherBifurcationNodes = [node for node in patient.Topology.BifurcationNodes]
    KDtree = scipy.spatial.KDTree([i.Position for i in patient.Topology.BifurcationNodes])
    keepvessels = set()

    for vessel in VesselsToDelete:
        if vessel.Nodes[0] in patient.Topology.PAnodes:
            # remap
            _, MinDistanceIndexR = KDtree.query(vessel.Nodes[0].Position, k=1)
            bifnode = patient.Topology.BifurcationNodes[MinDistanceIndexR]

            # connections
            vessel.Nodes[-1].ResetConnections()
            vessel.Nodes[-1].AddConnection(bifnode)
            bifnode.AddConnection(vessel.Nodes[-1])
            vessel.Nodes[-1].AddConnection(vessel.Nodes[-2])

            # positions
            vessel.Nodes[-1].SetPosition(bifnode.Position)

            # update positions in vessel
            vessel.Nodes[-2].SetPosition([0.5 * vessel.Nodes[0].Position[0] + 0.5 * vessel.Nodes[-1].Position[0],
                                          0.5 * vessel.Nodes[0].Position[1] + 0.5 * vessel.Nodes[-1].Position[1],
                                          0.5 * vessel.Nodes[0].Position[2] + 0.5 * vessel.Nodes[-1].Position[2]])
            keepvessels.add(vessel.ID)
            [NodesNotInNetwork.remove(node) for node in vessel.Nodes]

            vessel.Length = GeneralFunctions.distancebetweenpoints(vessel.Nodes[0].Position, vessel.Nodes[-1].Position)
            vessel.Nodes[0].LengthAlongVessel = 0
            vessel.Nodes[1].LengthAlongVessel = 0.5 * vessel.Length
            vessel.Nodes[-1].LengthAlongVessel = vessel.Length

        if vessel.Nodes[-1] in patient.Topology.PAnodes:
            # remap
            _, MinDistanceIndexR = KDtree.query(vessel.Nodes[-1].Position, k=1)
            bifnode = patient.Topology.BifurcationNodes[MinDistanceIndexR]

            # connections
            vessel.Nodes[0].ResetConnections()
            vessel.Nodes[0].AddConnection(bifnode)
            bifnode.AddConnection(vessel.Nodes[0])
            vessel.Nodes[0].AddConnection(vessel.Nodes[1])

            # positions
            vessel.Nodes[0].SetPosition(bifnode.Position)

            # update positions in vessel
            vessel.Nodes[1].SetPosition([0.5 * vessel.Nodes[0].Position[0] + 0.5 * vessel.Nodes[-1].Position[0],
                                         0.5 * vessel.Nodes[0].Position[1] + 0.5 * vessel.Nodes[-1].Position[1],
                                         0.5 * vessel.Nodes[0].Position[2] + 0.5 * vessel.Nodes[-1].Position[2]])

            keepvessels.add(vessel.ID)
            [NodesNotInNetwork.remove(node) for node in vessel.Nodes]

            vessel.Length = GeneralFunctions.distancebetweenpoints(vessel.Nodes[0].Position, vessel.Nodes[-1].Position)
            vessel.Nodes[0].LengthAlongVessel = 0
            vessel.Nodes[1].LengthAlongVessel = 0.5 * vessel.Length
            vessel.Nodes[-1].LengthAlongVessel = vessel.Length

    [VesselIDsToDelete.remove(id) for id in keepvessels]

    # remove vessel from vessel list
    print(f"Number of PAnodes remapped: {len(keepvessels)}")
    print(f"Number of vessels before filter: {len(patient.Topology.Vessels)}")
    patient.Topology.Vessels = [vessel for vessel in patient.Topology.Vessels if vessel.ID not in VesselIDsToDelete]
    print(f"Number of vessels After filter: {len(patient.Topology.Vessels)}")
    # remove other nodes from lists
    print(f"Number of PAnodes before filter: {len(patient.Topology.PAnodes)}")
    patient.Topology.PAnodes = [node for node in patient.Topology.PAnodes if node not in NodesNotInNetwork]
    print(f"Number of PAnodes After filter: {len(patient.Topology.PAnodes)}")
    print(f"Number of OutletNodes before filter: {len(patient.Topology.OutletNodes)}")
    patient.Topology.OutletNodes = [node for node in patient.Topology.OutletNodes if node not in NodesNotInNetwork]
    print(f"Number of OutletNodes After filter: {len(patient.Topology.OutletNodes)}")

    patient.Topology.UpdateTopology()
    patient.Topology.TopologyToGraph()


def PialNetworkToPatientTopology(pialnetworkvessels, patient):
    """
    The pial network is defined as a set of bifurcation nodes.
    This function creates proper vessels from the network.
    :param pialnetworkvessels: vessels to be updated
    :param patient: Patient object
    :return: None.
    """
    print("Updating the pial network vessels of the patient topology.")
    setbifnodes = set()
    for vessel in pialnetworkvessels:
        # for the Pa endnodes update the lengths?
        if len(vessel.Nodes[-1].Connections) == 1:
            d = GeneralFunctions.distancebetweenpoints(vessel.Nodes[0].Position, vessel.Nodes[-1].Position)
            vessel.Nodes[-1].SetLengthAlongVessel(d)

        # insert node and place bifnode in node list
        if len(vessel.Nodes[0].Connections) >= 2:
            # node is bifnode
            newnode = Node.Node()
            newnode.SetPosition(vessel.Nodes[0].Position)
            # newnode.SetLengthAlongVessel(0)
            newnode.SetRadius(vessel.Nodes[0].Radius)
            # remove old connections
            vessel.Nodes[0].RemoveConnection(vessel.Nodes[1])
            vessel.Nodes[1].RemoveConnection(vessel.Nodes[0])
            # connect newnode to bifnode
            vessel.Nodes[0].AddConnection(newnode)
            newnode.AddConnection(vessel.Nodes[0])
            # to other node
            vessel.Nodes[1].AddConnection(newnode)
            newnode.AddConnection(vessel.Nodes[1])
            # update topology
            setbifnodes.add(vessel.Nodes[0])
            vessel.Nodes[0] = newnode
            patient.Topology.Nodes.append(newnode)

        if len(vessel.Nodes[-1].Connections) >= 2:
            # node is bifnode
            newnode = Node.Node()
            newnode.SetPosition(vessel.Nodes[-1].Position)
            d = GeneralFunctions.distancebetweenpoints(vessel.Nodes[0].Position, vessel.Nodes[-1].Position)
            newnode.SetLengthAlongVessel(d)
            newnode.SetRadius(vessel.Nodes[-1].Radius)

            # remove old connections
            vessel.Nodes[-1].RemoveConnection(vessel.Nodes[-2])
            vessel.Nodes[-2].RemoveConnection(vessel.Nodes[-1])
            # connect newnode to bifnode
            vessel.Nodes[-1].AddConnection(newnode)
            newnode.AddConnection(vessel.Nodes[-1])
            # to other node
            vessel.Nodes[-2].AddConnection(newnode)
            newnode.AddConnection(vessel.Nodes[-2])
            # update topology
            setbifnodes.add(vessel.Nodes[-1])
            vessel.Nodes[-1] = newnode
            patient.Topology.Nodes.append(newnode)

        # insert a node in the center of the vessel as the third
        newnode = Node.Node()
        pos = [(vessel.Nodes[0].Position[0] + vessel.Nodes[-1].Position[0]) / 2,
               (vessel.Nodes[0].Position[1] + vessel.Nodes[-1].Position[1]) / 2,
               (vessel.Nodes[0].Position[2] + vessel.Nodes[-1].Position[2]) / 2]
        newnode.SetPosition(pos)
        d = (vessel.Nodes[0].LengthAlongVessel + vessel.Nodes[-1].LengthAlongVessel)
        newnode.SetLengthAlongVessel(d / 2)
        r = (vessel.Nodes[0].Radius + vessel.Nodes[-1].Radius) / 2
        newnode.SetRadius(r)

        vessel.Nodes[0].SetLengthAlongVessel(0)
        vessel.Nodes[-1].SetLengthAlongVessel(d)

        vessel.Nodes[0].RemoveConnection(vessel.Nodes[-1])
        vessel.Nodes[-1].RemoveConnection(vessel.Nodes[0])

        vessel.Nodes[0].AddConnection(newnode)
        newnode.AddConnection(vessel.Nodes[0])
        vessel.Nodes[-1].AddConnection(newnode)
        newnode.AddConnection(vessel.Nodes[-1])
        patient.Topology.Nodes.append(newnode)
        vessel.Nodes.insert(1, newnode)

        vessel.SetLength(GeneralFunctions.distancebetweenpoints(vessel.Nodes[0].Position, vessel.Nodes[-1].Position))

    patient.Topology.BifurcationNodes.extend(list(setbifnodes))
    patient.Topology.UpdateTopology()
    print(f"Number of Nodes: {len(patient.Topology.Nodes)}")


def ExportSimulationResults(patient, healthy=True):
    """
    Create figures and export data from pial network simulation
    :param patient: patient object
    :param healthy: Boolean (default:True) This changes the filenames so that files are not overwritten.
    :return: ClusterFlowData.csv, ResultsPerVessel.csv, Topology.vtp, Results.dyn, FlowRateHistograms.png
    """
    if healthy:
        clusterflowratefile = "ClusterFlowData.csv"
        meanresultsfile = "ResultsPerVessel.csv"
        topologyfile = "Topology.vtp"
        resultsfile = "Results.dyn"
        flowhistogramsfile = "FlowRateHistograms.png"
    else:
        clusterflowratefile = "ClusterFlowDataClot.csv"
        meanresultsfile = "ResultsPerVesselClot.csv"
        topologyfile = "TopologyClot.vtp"
        resultsfile = "ResultsClot.dyn"
        flowhistogramsfile = "FlowRateHistogramsClot.png"

    patient.Results1DSteadyStateModel()
    patient.Export1DSteadyClusterFlowRate(file=clusterflowratefile)
    patient.ExportMeanResults(file=meanresultsfile)
    patient.Results.AddResultsPerNodeToFile(patient.Folders.ModellingFolder + topologyfile)
    patient.Results.AddResultsPerVesselToFile(patient.Folders.ModellingFolder + topologyfile)

    resultsfolder = patient.Folders.ModellingFolder
    TimePoint = Results.TimePoint(0)
    TimePoint.Flow = [node.FlowRate for node in patient.Topology.Nodes]
    TimePoint.Pressure = [node.Pressure for node in patient.Topology.Nodes]
    TimePoint.Radius = [node.Radius for node in patient.Topology.Nodes]

    TimePoint2 = Results.TimePoint(patient.ModelParameters['Beat_Duration'])
    TimePoint2.Flow = TimePoint.Flow
    TimePoint2.Pressure = TimePoint.Pressure
    TimePoint2.Radius = TimePoint.Radius

    patient.Results.TimePoints = [TimePoint, TimePoint2]
    patient.Results.ExportResults(resultsfolder + resultsfile)

    # calculate volume flow rate to the brain
    VolumeToBrain = sum([node.FlowRate for node in patient.Topology.PAnodes])
    print("Total perfusion to the brain: %f" % VolumeToBrain)
    # print some stats
    print(f"Endnodes: {len(patient.Topology.OutletNodes)}")
    print(f"Nodes: {len(patient.Topology.Nodes)}")
    print(f"PialNodes: {len(patient.Topology.PAnodes)}")

    # calculate histograms of the Flow Rate and Pressure for the PA nodes
    FlowRate = [node.FlowRate for node in patient.Topology.PAnodes]
    Pressure = [node.Pressure for node in patient.Topology.PAnodes]
    Velocity = [abs(node.Velocity) for node in patient.Topology.PAnodes]
    Radius = [node.Radius for node in patient.Topology.PAnodes]

    fig, axes = plt.subplots(1, 4)
    DPI = fig.get_dpi()
    fig.set_size_inches(4000.0 / float(DPI), 1000.0 / float(DPI))
    axes[0].hist(FlowRate, bins=100)
    axes[0].set_title("Volume Flow Rate Penetrating Arteries", fontsize=30)
    axes[0].set_xlabel("Volume Flow Rate (mL/s)", fontsize=28)
    axes[0].set_ylabel("Number (#)", fontsize=28)
    axes[0].xaxis.set_tick_params(labelsize=22)
    axes[0].yaxis.set_tick_params(labelsize=22)
    axes[0].grid()

    axes[1].hist(Pressure, bins=100)
    axes[1].set_title("Pressure Penetrating Arteries", fontsize=30)
    axes[1].set_xlabel("Pressure (pa)", fontsize=28)
    axes[1].set_ylabel("Number (#)", fontsize=28)
    axes[1].xaxis.set_tick_params(labelsize=22)
    axes[1].yaxis.set_tick_params(labelsize=22)
    axes[1].grid()

    axes[2].hist(Velocity, bins=100)
    axes[2].set_title("Velocity Penetrating Arteries", fontsize=30)
    axes[2].set_xlabel("Velocity (m/s)", fontsize=28)
    axes[2].set_ylabel("Number (#)", fontsize=28)
    axes[2].xaxis.set_tick_params(labelsize=22)
    axes[2].yaxis.set_tick_params(labelsize=22)
    axes[2].grid()

    axes[3].hist(Radius, bins=100)
    axes[3].set_title("Radius Penetrating Arteries", fontsize=30)
    axes[3].set_xlabel("Radius (mm)", fontsize=28)
    axes[3].set_ylabel("Number (#)", fontsize=28)
    axes[3].xaxis.set_tick_params(labelsize=22)
    axes[3].yaxis.set_tick_params(labelsize=22)
    axes[3].grid()

    fig.savefig(patient.Folders.ModellingFolder + flowhistogramsfile)
    # plt.show()
    plt.close("all")


def CalculatePressureDrop(patient):
    """
    Collateral measure and pressure difference across clots in the patient network.
    :param patient: patient object.
    :return: None.
    """
    # find the vessel(s) with a clot
    patient.Topology.NodeVesselDict()
    clotvessels = set()
    clotnodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    for clotnode in clotnodes:
        vessel = patient.Topology.NodeDict[clotnode]
        clotvessels.add(vessel)

    # calculate pressure drop as abs(p1-p2)
    dp = None
    for vessel in list(clotvessels):
        dp = abs(vessel.Nodes[0].Pressure - vessel.Nodes[-1].Pressure)
        vessel.PressureDrop = dp
        print(f"Pressure drop across the clot: {dp}")
    return dp


def ExportSystemResult(patient, folder, file="ContrastGraphMeasurements.csv", figname="TimedelayGraph.png",
                       figname2="Pressure.png"):
    """
    Function to output important results from the collateral simulation.
    Note: run this after the graph contrast model.
    :param patient: patient object
    :param folder: location to store the csv
    :param file: csv name
    :return: None
    """
    figuredpi = 72

    # find the vessel(s) with a clot
    patient.Topology.NodeVesselDict()
    clotvessels = set()
    clotnodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    for clotnode in clotnodes:
        vessel = patient.Topology.NodeDict[clotnode]
        clotvessels.add(vessel)

    for vessel in clotvessels:
        pos = [node.LengthAlongVessel for node in vessel.Nodes]
        timedelay = [node.relativetimedelay for node in vessel.Nodes]
        fig = plt.figure()
        DPI = fig.get_dpi()
        fig.set_size_inches(1000.0 / float(DPI), 1000.0 / float(DPI))

        axis1 = fig.add_subplot()
        axis1.set_title("Time Delay over a clot", fontsize=30)
        axis1.set_xlabel("Position (mm)", fontsize=30)
        axis1.set_ylabel("Time (s)", fontsize=30)
        axis1.xaxis.set_tick_params(labelsize=25)
        axis1.yaxis.set_tick_params(labelsize=25)

        # axis1.set_ylim([0,10])
        # axis1.set_xscale('log')
        axis1.grid()
        axis1.plot(pos, timedelay)

        fig.canvas.draw_idle()
        fig.tight_layout()
        if figname:
            fig.savefig(folder + figname, dpi=figuredpi)
        # plt.show()
        plt.close("all")

    for vessel in clotvessels:
        pos = [node.LengthAlongVessel for node in vessel.Nodes]
        pressure = [node.Pressure for node in vessel.Nodes]
        fig = plt.figure()
        DPI = fig.get_dpi()
        fig.set_size_inches(1000.0 / float(DPI), 1000.0 / float(DPI))

        axis1 = fig.add_subplot()
        axis1.set_title("Pressure over a clot", fontsize=30)
        axis1.set_xlabel("Position (mm)", fontsize=30)
        axis1.set_ylabel("Pressure (pa)", fontsize=30)
        axis1.xaxis.set_tick_params(labelsize=25)
        axis1.yaxis.set_tick_params(labelsize=25)

        # axis1.set_ylim([0, 10])
        # axis1.set_xscale('log')
        axis1.grid()
        axis1.plot(pos, pressure)

        fig.canvas.draw_idle()
        fig.tight_layout()
        if figname:
            fig.savefig(folder + figname2, dpi=figuredpi)
        # plt.show()
        plt.close("all")

    print("Export time delay results.")
    with open(folder + file, 'w') as f:
        f.write(
            "VesselName,Graph Time Delay (s),Pressure Difference (pa),Pressure Difference (mmHg)\n")
        for vessel in clotvessels:
            f.write("%s,%f,%f,%f\n" % (
                vessel.Name, vessel.Nodes[-1].timedelay - vessel.Nodes[0].timedelay, vessel.PressureDrop,
                vessel.PressureDrop * 0.007500617))


def ExportResultsToPAfile(patient, file, healthy=True, cellArray=False):
    """
    Map results to the PA file.
    :param patient: patient object
    :param file: PA file
    :param healthy: bool (default:True). Choose between healthy and clot, this affects the name of the arrays added.
    :return: None.
    """
    if healthy:
        namepressure = "Pressure Healthy"
        nameflow = "Flow Rate Healthy"
        namevelocity = "Velocity Healthy"
    else:
        namepressure = "Pressure Clot"
        nameflow = "Flow Rate Clot"
        namevelocity = "Velocity Clot"

    pressure = np.zeros(patient.Topology.NumberOfPenetratingArteries)
    for node in patient.Topology.PAnodes:
        pressure[node.PAnumber] = node.Pressure
    GeneralFunctions.AddArrayToFile(file, pressure, namepressure, cellArray)

    flowrate = np.zeros(patient.Topology.NumberOfPenetratingArteries)
    for node in patient.Topology.PAnodes:
        flowrate[node.PAnumber] = node.FlowRate
    GeneralFunctions.AddArrayToFile(file, flowrate, nameflow, cellArray)

    Velocity = np.zeros(patient.Topology.NumberOfPenetratingArteries)
    for node in patient.Topology.PAnodes:
        Velocity[node.PAnumber] = node.Velocity
    GeneralFunctions.AddArrayToFile(file, Velocity, namevelocity, cellArray)


def ExportPAnodes(patient):
    """
    Exporting the order of the PAnodes.
    This file is required to easily map the surface files with the topology files.

    Parameters
    ----------
    patient : patient object
        patient object with data.
    """
    print("Exporting order of PANodes.")
    file = "PAmapping.csv"
    with open(patient.Folders.ModellingFolder + file, 'w') as f:
        f.write("PANode,NodeID,\n")
        for index, node in enumerate(patient.Topology.PAnodes):
            f.write("%d,%d\n" % (index, node.Number))


def ImportPAnodes(patient):
    """
    Import the order of the PANodes.
    The first column is the number of the cell or point in the surface files.
    The second column is the node number in the topology file.

    Parameters
    ----------
    patient : patient object
    """
    print("Importing order of PANodes.")
    datafile = patient.Folders.ModellingFolder + "PAmapping.csv"
    nodeorder = [i.strip('\n').split(',') for i in open(datafile)][1:]
    patient.Topology.PAnodes = [patient.Topology.Nodes[int(number[1])] for number in nodeorder]
    patient.Topology.NumberOfPenetratingArteries = len(patient.Topology.PAnodes)
    for index, node in enumerate(patient.Topology.PAnodes):
        node.PAnumber = index


def GenerateTreesFromResistance(patient):
    print("Generating bifurcating arterial trees at the outlet nodes.")
    ### run perfusion model once to obtain resistance values
    file = patient.Folders.PatientFolder + "Model_values_Healthy.csv"
    data = [line.strip('\n').split(',') for line in open(file)][1:]
    for index, line in enumerate(data):
        patient.Perfusion.CouplingPoints[index].Node.TreeResistance = float(line[1])
        patient.Perfusion.CouplingPoints[index].Node.OutPressure = float(line[4])

    ### generate trees based on resistance
    def GenerateTreeFromResistance(self, node):
        # estimate parameters
        N, LR = EstimateParameters(self, node)
        print(f"Tree: Generaterions:{N},LR:{LR}")

        tree = Topology.Tree(node)
        tree.NumberOfGenerations = N
        tree.LengthRadiusRatio = LR

        GenerateTree(tree)
        self.Trees.append(tree)
        return tree

    def EstimateParameters(patient, node):
        # see Olufsen1999
        Z = node.TreeResistance
        r = node.Radius * 1e-3

        mu = patient.ModelParameters["BLOOD_VISC"]
        # mu = 0.0035

        alpha = np.power(0.5, 1 / 3)
        initial_lam = 250
        f = 8  # or 8 if laminar

        c1 = f * mu * initial_lam / (np.pi * np.power(r, 3))
        c3 = (1 / (2 * np.power(alpha, 3))) - 1

        topnumber = (Z * c3 / c1) + 1
        base = (1 / (2 * np.power(alpha, 3)))
        exponent = np.log(topnumber) / np.log(base)
        numbergens = max(1, int(round(exponent)))

        if numbergens == 0:
            # no tree
            return numbergens, initial_lam

        # update lamda to acount for the rounding of numbergens
        c2 = np.power((1 / (2 * np.power(alpha, 3))), numbergens) - 1
        guesslam = (Z / (f * mu / (np.pi * np.power(r, 3)))) * c3 / c2

        return numbergens, guesslam

    def GenerateTree(self):
        """
         Generate a bifurcating tree at the endnodes of the tree.

        Parameters
        ----------
        self

        Returns
        -------
        """
        currentgen = 0
        while 1:
            Newendnodes = []
            for endnode in self.EndNodes:
                if currentgen < self.NumberOfGenerations or currentgen == 0:
                    # calculate node properties
                    r1 = BloodFlowEquations.murraylaw(endnode.Radius)
                    r2 = BloodFlowEquations.murraylaw(endnode.Radius)

                    l1 = BloodFlowEquations.RadiusToLength(r1, ratio=self.LengthRadiusRatio)
                    l2 = BloodFlowEquations.RadiusToLength(r2, ratio=self.LengthRadiusRatio)

                    direction = [-1 * i for i in endnode.DirectionVector]

                    angle = math.atan2(direction[1], direction[0])
                    theta = math.atan(0.75) + angle
                    theta2 = math.atan(-0.75) + angle

                    # 90 degree angle
                    # theta = math.atan(1) + angle
                    # theta2 = math.atan(-1) + angle

                    posvec = [l1 * math.cos(theta),
                              l1 * math.sin(theta),
                              0]
                    posvec2 = [l2 * math.cos(theta2),
                               l2 * math.sin(theta2),
                               0]

                    pos1end = [endnode.Position[0] + posvec[0],
                               endnode.Position[1] + posvec[1],
                               endnode.Position[2] + posvec[2]]
                    pos2end = [endnode.Position[0] + posvec2[0],
                               endnode.Position[1] + posvec2[1],
                               endnode.Position[2] + posvec2[2]]

                    pos1mid = [endnode.Position[0] + 0.5 * posvec[0],
                               endnode.Position[1] + 0.5 * posvec[1],
                               endnode.Position[2] + 0.5 * posvec[2]]
                    pos2mid = [endnode.Position[0] + 0.5 * posvec2[0],
                               endnode.Position[1] + 0.5 * posvec2[1],
                               endnode.Position[2] + 0.5 * posvec2[2]]

                    # create new nodes
                    bifurcationnode = Node.Node()
                    bifurcationnode.SetPosition(endnode.Position)
                    bifurcationnode.SetMajorVesselID(endnode.MajorVesselID)

                    vessel1node1 = Node.Node()
                    vessel1node1.SetPosition(endnode.Position)
                    vessel1node1.SetLengthAlongVessel(0.0)
                    vessel1node1.SetRadius(r1)
                    vessel1node1.SetMajorVesselID(endnode.MajorVesselID)
                    vessel1node2 = Node.Node()
                    vessel1node2.SetPosition(pos1mid)
                    vessel1node2.SetLengthAlongVessel(0.5 * l1)
                    vessel1node2.SetRadius(r1)
                    vessel1node2.SetMajorVesselID(endnode.MajorVesselID)
                    vessel1node3 = Node.Node()
                    vessel1node3.SetPosition(pos1end)
                    vessel1node3.SetLengthAlongVessel(l1)
                    vessel1node3.SetRadius(r1)
                    vessel1node3.SetMajorVesselID(endnode.MajorVesselID)

                    vessel2node1 = Node.Node()
                    vessel2node1.SetPosition(endnode.Position)
                    vessel2node1.SetLengthAlongVessel(0.0)
                    vessel2node1.SetRadius(r2)
                    vessel2node1.SetMajorVesselID(endnode.MajorVesselID)
                    vessel2node2 = Node.Node()
                    vessel2node2.SetPosition(pos2mid)
                    vessel2node2.SetLengthAlongVessel(0.5 * l2)
                    vessel2node2.SetRadius(r2)
                    vessel2node2.SetMajorVesselID(endnode.MajorVesselID)
                    vessel2node3 = Node.Node()
                    vessel2node3.SetPosition(pos2end)
                    vessel2node3.SetLengthAlongVessel(l2)
                    vessel2node3.SetRadius(r2)
                    vessel2node3.SetMajorVesselID(endnode.MajorVesselID)

                    vessel1 = Vessel.Vessel()
                    vessel1.SetType(3)
                    vessel2 = Vessel.Vessel()
                    vessel2.SetType(3)
                    vessel1.SetNodes([vessel1node1, vessel1node2, vessel1node3])
                    vessel2.SetNodes([vessel2node1, vessel2node2, vessel2node3])

                    vessel1.SetLength(l1)
                    vessel2.SetLength(l2)
                    vessel1.SetMeanRadius(r1)
                    vessel2.SetMeanRadius(r2)
                    # vessel1.SetGridSize(0.5 * l1)
                    # vessel2.SetGridSize(0.5 * l2)
                    vessel1.SetMajorVesselID(endnode.MajorVesselID)
                    vessel2.SetMajorVesselID(endnode.MajorVesselID)
                    # Set connections
                    # do not add them to the main topology for now
                    bifurcationnode.AddConnection(vessel1node1)
                    bifurcationnode.AddConnection(vessel2node1)

                    bifurcationnode.SetRadius((r1 + endnode.Radius + r2) / 3)

                    vessel1node1.AddConnection(bifurcationnode)
                    vessel1node1.AddConnection(vessel1node2)
                    vessel1node2.AddConnection(vessel1node1)
                    vessel1node2.AddConnection(vessel1node3)
                    vessel1node3.AddConnection(vessel1node2)

                    vessel2node1.AddConnection(bifurcationnode)
                    vessel2node1.AddConnection(vessel2node2)
                    vessel2node2.AddConnection(vessel2node1)
                    vessel2node2.AddConnection(vessel2node3)
                    vessel2node3.AddConnection(vessel2node2)

                    if currentgen != 0:
                        endnode.AddConnection(bifurcationnode)
                        bifurcationnode.AddConnection(endnode)
                    else:
                        self.StartingTreeNodes = bifurcationnode

                    self.BifurcationNodes.append(bifurcationnode)
                    Newendnodes.append(vessel1node3)
                    Newendnodes.append(vessel2node3)

                    self.Vessels.append(vessel1)
                    self.Vessels.append(vessel2)
                    self.Nodes.append(bifurcationnode)

                    # self.Nodes.append(vessel1node1)
                    # self.Nodes.append(vessel1node2)
                    # self.Nodes.append(vessel1node3)
                    # self.Nodes.append(vessel2node1)
                    # self.Nodes.append(vessel2node2)
                    # self.Nodes.append(vessel2node3)
                    vessel1node3.SetDirectionVector()
                    vessel2node3.SetDirectionVector()
                    numbernodes = int(max(3, math.ceil(vessel1.Length / vessel1.GridSize) + 1))
                    vessel1.UpdateResolution(numbernodes)
                    numbernodes = int(max(3, math.ceil(vessel2.Length / vessel2.GridSize) + 1))
                    vessel2.UpdateResolution(numbernodes)

                    for node in vessel1.Nodes:
                        self.Nodes.append(node)
                    for node in vessel2.Nodes:
                        self.Nodes.append(node)

            if len(Newendnodes) == 0:
                break
            self.EndNodes = Newendnodes
            currentgen += 1
        # rotation to match the direction of the original vessel

        Vo = [-1 * self.Direction[0], -1 * self.Direction[1], -1 * self.Direction[2]]
        Vn = [-1 * self.Direction[0], -1 * self.Direction[1], 0]

        Vcross = np.cross(Vn, Vo)
        C = np.dot(Vn, Vo)
        Vx = [[0, -1 * Vcross[2], Vcross[1]],
              [Vcross[2], 0, -1 * Vcross[0]],
              [-1 * Vcross[1], Vcross[0], 0]]

        R = np.eye(3) + Vx + np.matmul(Vx, Vx) * (1 / (1 + C))
        Rnew = np.eye(4)
        Rnew[0:3, 0:3] = R

        Translate1 = np.array(
            [[1, 0, 0, -1 * self.InitialNode.Position[0]],
             [0, 1, 0, -1 * self.InitialNode.Position[1]],
             [0, 0, 1, -1 * self.InitialNode.Position[2]],
             [0, 0, 0, 1]])
        Translate2 = np.array(
            [[1, 0, 0, 1 * self.InitialNode.Position[0]],
             [0, 1, 0, 1 * self.InitialNode.Position[1]],
             [0, 0, 1, 1 * self.InitialNode.Position[2]],
             [0, 0, 0, 1]])

        TMatrix = np.dot(Translate2, np.dot(Rnew, Translate1))

        for node in self.Nodes:
            pos = node.Position
            vec = np.array([[pos[0]], [pos[1]], [pos[2]], [1]])
            position = np.dot(TMatrix, vec)
            posnew = [position[0][0], position[1][0], position[2][0]]
            node.SetPosition(posnew)

    for id in patient.Perfusion.CouplingPoints:
        tree = GenerateTreeFromResistance(patient, id.Node)
        id.Tree = tree

    patient.Topology.UpdateTopology()
    print("Generated %d bifurcating trees." % len(patient.Trees))


def add_network_pa_nodes(patient):
    print("Generating Pial Network.")
    # generate bifurcating trees until a certain cut-off radius
    # cutoff strongly scales the number of coupling points.
    patient.GenerateTreesAtCps(patient.ModelParameters["TreeCutoffRadius"])
    # generate trees based on resistance
    # Collaterals.GenerateTreesFromResistance(patient)

    # mapping outlets to the clusters by iterative division
    # map trees to the surface and find a node to connect the ends to the network.
    TreeToSurfaceMappingMetis(patient)
    ExportTreeEndNodePositions(patient)

    # connect trees to surface
    patient.AddTreesToTop()

    # load surface and create pial network
    # extracting edges and nodes for the pial network
    surfacefile = patient.Folders.ModellingFolder + "Clustering.vtp"
    patienttemp, nodepositions, edges, colleratalEdges = CreatePialNetwork(surfacefile,
                                                                           patient.ModelParameters[
                                                                               "separation"],
                                                                           patient.ModelParameters[
                                                                               "collateralpropability"])

    if patient.ModelParameters["UsingPAfile"] == "True":
        # use precalculated PAs
        PAfile = patient.Folders.ModellingFolder + "PA.vtp"
        penetratingartery, connectionsPA = PAfromFile(PAfile, nodepositions)
        patient.Topology.NumberOfPenetratingArteries = len(penetratingartery)
    else:
        # generate PAs on the primal graph surface
        totalarea = sum(patient.Perfusion.PrimalGraph.Areas)
        print("Total surface area: %f mm^2" % totalarea)
        # PA density is 1 per mm^2
        patient.Topology.NumberOfPenetratingArteries = int(totalarea)
        penetratingartery, connectionsPA = GeneratePAs(patienttemp.PrimalGraph,
                                                       patient.Topology.NumberOfPenetratingArteries)

    # merge pialnetwork and penetrating arteries
    edges.extend(connectionsPA)
    nodepositions.extend(penetratingartery)

    # generate simple vessels of the pial network and PAs
    colleratalVessels, pialnetworknodes, pialnetworkvessels = GenerateVessels(patient, nodepositions, edges,
                                                                              colleratalEdges,
                                                                              patient.ModelParameters[
                                                                                  "PialnetworkRadius"])

    # set outlet node parameters
    SetPAOutlets(patient, patient.ModelParameters["brainperfusionestimate"],
                 patient.ModelParameters["pialsurfacepressure"],
                 patient.ModelParameters["OUT_PRESSURE"])
    # # we need to update the pial network to proper 3-node vessels
    PialNetworkToPatientTopology(pialnetworkvessels, patient)
    RemoveUnconnectedSections(patient)

    # export files
    patient.SaveVesselAtlas()

    #map panodes to major vessel regions
    print("Finding outlets close to the surface.")
    #load dualPA
    patienttemp1 = Perfusion.Perfusion()
    # patienttemp1.PrimalGraph.LoadSurface(patient.Folders.ModellingFolder + "DualPA.vtp")
    patienttemp1.PrimalGraph.LoadSurface(patient.Folders.ModellingFolder + "Clustering.vtp")
    patienttemp1.SetDualGraph()
    dual_pa_pos = patienttemp1.DualGraph.PialSurface
    # KDTree = scipy.spatial.KDTree(patient.Perfusion.PrimalGraph.PialSurface)
    KDTree = scipy.spatial.KDTree(dual_pa_pos)
    pos = [i.Position for i in patient.Topology.PAnodes]
    # euclidean distance between outlets and surface
    _, MinDistanceIndex = KDTree.query(pos, k=1)
    patient.Topology.NodeVesselDict()
    # patienttemp2 = Perfusion.Perfusion()
    # patienttemp2.PrimalGraph.LoadSurface(patient.Folders.ModellingFolder + "PA.vtp")
    for nearest_index, node in zip(MinDistanceIndex, patient.Topology.PAnodes):
        node.MajorVesselID = patienttemp1.PrimalGraph.map[nearest_index]
        vessel = patient.Topology.NodeDict[node]
        vessel.MajorVesselID = node.MajorVesselID

    patient.WriteRegionMapping()

    patient.TopologyToVTP()
    patient.WriteSimFiles()
    ExportPAnodes(patient)  # save order of PA nodes
    # save regionsplits to file
    WriteRegionsSplitsToVTP(patient, patient.Folders.ModellingFolder + "RegionSplitting.vtp")
    # write output of pialnetwork
    file = patient.Folders.ModellingFolder + "PialNetworkWithPA.vtp"
    GeneralFunctions.WriteEdgesToVTP(nodepositions, edges, file)
    array = [1 if vessel in colleratalEdges else 0 for vessel in edges]
    GeneralFunctions.AddArrayToFile(file, array, "Collateral Vessels", True)
    # add marker for the collateral vessels to topology
    file = patient.Folders.ModellingFolder + "Topology.vtp"
    array = [1 if vessel in colleratalVessels else 0 for vessel in patient.Topology.Vessels]
    GeneralFunctions.AddArrayToFile(file, array, "Collateral Vessels", True)
    # add vessel type to the topology file
    vesseltype = [i.Type for i in patient.Topology.Vessels]
    GeneralFunctions.AddArrayToFile(file, vesseltype, "Vessel Type", True)
    # export system
    patient.Topology.WriteNodesCSV(patient.Folders.ModellingFolder + "Nodes.csv")
    patient.Topology.WriteVesselCSV(patient.Folders.ModellingFolder + "Vessels.csv")

def estimate_perfusion(patient):
    """
    Get perfusion values.
    Set brainperfusionestimate in model parameters to this value.
    Parameters
    ----------
    patient: patient object

    Returns
    -------
    """
    try:
        left_ica = [vessel[1][0] for vessel in patient.Results.MeanResults if vessel[0] == "L int. carotid I"][0]*1e-6
        right_ica = [vessel[1][0] for vessel in patient.Results.MeanResults if vessel[0] == "R int. carotid I"][0]*1e-6
        right_vertebral = [vessel[1][0] for vessel in patient.Results.MeanResults if vessel[0] == "R. vertebral"][0]*1e-6
        left_vertebral = [vessel[1][0] for vessel in patient.Results.MeanResults if vessel[0] == "L. vertebral"][0]*1e-6
        cerebral_perfusion = left_ica + right_ica + right_vertebral + left_vertebral

        print(f"Estimated cerebral perfusion: {cerebral_perfusion} mm^3/s = {cerebral_perfusion*1e6} mL/s = {cerebral_perfusion*60*1e6} mL/min")
        patient.ModelParameters["brainperfusionestimate"] = cerebral_perfusion
        patient.WriteModelParameters()
    except IndexError:
        print("Network does not contain relevant cerebral vessels. Skipping perfusion calculation.")
