#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Contains the perfusion,  surface and coupling points object classes.
"""
import copy
import itertools
import sys

import networkx as nx
import numpy
import scipy
import scipy.sparse
import scipy.spatial
import vtk
from mgmetis import metis
from vtk.util.numpy_support import vtk_to_numpy

from Blood_Flow_1D import Constants, GeneralFunctions, Node


class Surface:
    def __init__(self):
        """
        Initiate object
        """
        self.PialSurface = []
        self.Graph = nx.Graph()
        self.Links = []
        self.Polygons = []
        self.Triangles = []
        self.SideInfo = []
        self.Areas = []  # per polygon
        self.File = None
        self.Weights = []

        self.Labels = []  # of points
        self.Points = []
        self.Roots = []
        self.NodeColour = []
        self.PolygonColour = []
        self.MajorVesselID = []
        self.connected_regions = []

    def CalculateScipyGraph(self):
        """
        Convert networkx graph to a scipy graph.
        :return: Nothing
        """
        self.ScipyGraph = nx.to_scipy_sparse_matrix(self.Graph, dtype=numpy.int64)

    def IdentifyNeighbouringRegions(self):
        """
        Identify touching regions of the surface
        Output list of sets where the index of the set refers to the region,
         and the set contains the index of the neighbouring regions.
        Returns:  sets self.connected_regions
        -------

        """
        #create list of sets of touching regions
        number_regions = len(set(self.NodeColour))
        connected_regions = [set() for _ in range(number_regions)]
        for edge in self.Graph.edges:
            label_1 = self.NodeColour[edge[0]]
            label_2 = self.NodeColour[edge[1]]
            if not (label_1 == label_2):
                connected_regions[label_1].add(label_2)
                connected_regions[label_2].add(label_1)
        self.connected_regions = connected_regions

    def GetSurfaceRegion(self, regionids):
        """
        Get part of the surface.
        :param regionids: region id of the part to extract
        :return: Triangles, positions
        """
        trianglesids = [i for i, element in enumerate(self.MajorVesselID) if element in regionids]
        triangles = [self.Triangles[i] for i in trianglesids]
        pointids = [y for x in triangles for y in x]
        points = list(set(pointids))
        pointspos = [self.PialSurface[i] for i in points]
        newtriangles = [[points.index(p) for p in triangle] for triangle in triangles]

        return newtriangles, pointspos

    # def SeparatePialNetwork(self):
    #     """
    #     functionto
    #     :return:
    #     """
    #     file = "Test.vtp"
    #     vesselspolydata = vtk.vtkPolyData()
    #     nodes = vtk.vtkPoints()
    #     for i in self.PialSurface:
    #         nodes.InsertNextPoint(i)
    #     vesselspolydata.SetPoints(nodes)
    #
    #     NodeColour = vtk.vtkIntArray()
    #     NodeColour.SetNumberOfComponents(1)
    #     NodeColour.SetName("Node Colour")
    #     for colour in self.NodeColour:
    #         NodeColour.InsertNextValue(colour)
    #     vesselspolydata.GetPointData().AddArray(NodeColour)
    #
    #     # removing edges
    #     edges = []
    #     for index, link in enumerate(self.Links):
    #         for linkednode in link:
    #             if self.NodeColour[index] == self.NodeColour[linkednode]:
    #                 edges.append((index, linkednode))
    #
    #     lines = vtk.vtkCellArray()
    #     for index, element in enumerate(edges):
    #         line0 = vtk.vtkLine()
    #         line0.GetPointIds().SetNumberOfIds(2)
    #         line0.GetPointIds().SetId(0, element[0])
    #         line0.GetPointIds().SetId(1, element[1])
    #         lines.InsertNextCell(line0)
    #     vesselspolydata.SetLines(lines)
    #
    #     # removing polygons
    #     # polygons = vtk.vtkCellArray()
    #     # for poly in self.PolygonsList:
    #     #     nodeides = [self.NodeColour[p] for p in poly]
    #     #     if len(set(nodeides)) == 1:
    #     #         polygon = vtk.vtkPolygon()
    #     #         polygon.GetPointIds().SetNumberOfIds(len(poly))
    #     #         for i in range(0, len(poly)):
    #     #             polygon.GetPointIds().SetId(i, poly[i])
    #     #         polygons.InsertNextCell(polygon)
    #     # vesselspolydata.SetPolys(polygons)
    #
    #     print("Saving to %s" % file)
    #     writer = vtk.vtkXMLPolyDataWriter()
    #     writer.SetFileName(file)
    #     writer.SetInputData(vesselspolydata)
    #     writer.Write()

    def CalculateDijkstraMatrix(self):
        """
        Warning, this can require TBs of space.
        Size depends on the size of the surface,
        Calculate the dijkstra distance of the surface
        :return: distance matrix
        """
        a = scipy.sparse.csgraph.dijkstra(self.ScipyGraph, directed=False, limit=300)
        print("Matrix Done!")
        return a

    def FindNearestSurfaceNode(self, couplingpoints, threshold=1e10):
        """
        Map the outlets to the nearest point on the surface.
        A threshold can be used to remove outlets far away.
        :param couplingpoints: points to map
        :param threshold: threshold
        :return: Nothing
        """
        KDTree = scipy.spatial.KDTree(self.PialSurface)
        pos = [i.Node.Position for i in couplingpoints]
        # euclidean distance between outlets and surface

        MinDistance, MinDistanceIndex = KDTree.query(pos, k=1)
        print("Finding outlets close to the surface.")
        for index, node in enumerate(couplingpoints):
            if MinDistance[index] < threshold:
                node.PialSurfacePointID = MinDistanceIndex[index]
                self.Roots.append(MinDistanceIndex[index])
            else:
                print("Outlet not close to the surface.")

    def ExportTriangleColour(self, folder):
        """
        Export surface mapping.
        File contains the triangle number, cluster ID and area of the triangle.
        :param folder: output folder
        :return: Nothing
        """
        file = folder + "SurfaceNodesMapping.csv"
        print("Writing surface node mapping to file: %s" % file)
        with open(file, 'w') as f:
            f.write("TriangleNumber,ClusterID,Area\n")
            for index, item in enumerate(self.Triangles):
                f.write(str(index) + "," + str(self.PolygonColour[index]) + "," + str(self.Areas[index]) + "\n")

    def CalculateAreas(self):
        """
        Compute the area of every triangle of the surface.
        :return: Nothing
        """
        print("Calculating area per triangle.")
        trianglenodes = []
        for triangle in self.Triangles:
            nodes = [self.PialSurface[i].tolist() for i in triangle]
            trianglenodes.append(nodes)
        self.Areas = [GeneralFunctions.TriangleToArea(triangle) for triangle in trianglenodes]

    def DefineSides(self):
        """
        Set the SideInfo variable based on the x-coordinate.
        1=right, -1=left.
        :return: Nothing
        """
        for index, node in enumerate(self.PialSurface):
            if node[0] > 0:
                self.SideInfo.append(1)
            else:
                self.SideInfo.append(-1)

    def define_sides_plane(self, point, normal):
        """
        Given plane defined as a point on the plane and the normal direction of the plane, determine below and above regions as left/right.
        1=right, -1=left.
        :return: Nothing
        """
        for index, node in enumerate(self.PialSurface):
            node_pos = numpy.array(node)
            d = numpy.dot(normal, node_pos-point)
            if d > 0:
                self.SideInfo.append(1)
            else:
                self.SideInfo.append(-1)

    def WeightedSampling(self, region, n):
        """
        Return a weighted sampling of the surface triangles without replacement.
        :param region: region to sample.
        :param n: number of returned sampled.
        :return: list of sampled.
        """
        weights = numpy.array([self.Weights[i] for i in region])
        prob = weights / weights.sum()
        sample = numpy.random.choice(region, n, replace=False, p=prob)
        return list(sample)

    def CalculateDistanceMatRegion(self, method, roots, surfacepoints):
        """
        Calculate the distance matrix between two set of points.
        :param method: Distance method: euclidean or dijkstra
        :param roots: First set
        :param surfacepoints: second set of points
        :return: distance matrix
        """
        print("Calculating Distance between roots and the pial surface.")
        if method == "euclidean":
            rootspos = scipy.array([self.PialSurface[point] for point in roots])
            pointpos = scipy.array([self.PialSurface[point] for point in surfacepoints])
            return scipy.spatial.distance.cdist(rootspos, pointpos, 'euclidean')
        else:
            results = scipy.sparse.csgraph.dijkstra(self.ScipyGraph, directed=False, indices=roots)
            results = results[:, surfacepoints]
            return results

    def ToGraph(self):
        """
        Convert the surface to a networkx graph. Edgeweight is the distance between nodes.
        :return: Nothing
        """
        print("Converting pialSurface to a weighted Graph.")
        for number in range(0, len(self.Links)):
            node1 = number
            for othernodes in self.Links[node1]:
                if othernodes > node1:
                    self.Graph.add_edge(node1, othernodes,
                                        weight=GeneralFunctions.distancebetweenpoints(self.PialSurface[node1],
                                                                                      self.PialSurface[othernodes]))

    def LoadSurface(self, file="Surface.vtp"):
        """
        Load a surface.
        :param file: surface file
        :return: Nothing
        """
        print("Loading surface: %s" % file)
        if file[-3:] == "vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif file[-3:] == "ply":
            reader = vtk.vtkPLYReader()
        elif file[-3:] == "msh":
            self.LoadSurfaceMSH(file)
            return
        else:
            return
        reader.SetFileName(file)
        reader.Update()

        pos_vtk = reader.GetOutput().GetPoints().GetData()
        pos = vtk_to_numpy(pos_vtk)

        NumberOfPoints = reader.GetOutput().GetNumberOfPoints()

        pialsurface = []
        for i in range(0, NumberOfPoints):
            pialsurface.append(pos[i])

        Connections_vtk = reader.GetOutput().GetPolys().GetData()
        Connections = vtk_to_numpy(Connections_vtk)
        NumberOfPolys = reader.GetOutput().GetNumberOfPolys()

        sideinfo = []
        narray = reader.GetOutput().GetPointData().GetNumberOfArrays()
        for i in range(0, narray):
            arrayname = reader.GetOutput().GetPointData().GetArrayName(i)
            if arrayname == "Result":
                sideinfo = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(i))
            if arrayname == "Major Cerebral Artery":
                self.map = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(i))
            if arrayname == "Volume Flow Rate (mL/s)":
                self.flowdata = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(i))
            if arrayname == "Colour":
                flowpercluster = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(i))
                self.PolygonColour = [int(i) for i in flowpercluster]
            if arrayname == "Node Colour":
                flowpercluster = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(i))
                self.NodeColour = [int(i) for i in flowpercluster]

        narray = reader.GetOutput().GetCellData().GetNumberOfArrays()
        for i in range(0, narray):
            arrayname = reader.GetOutput().GetCellData().GetArrayName(i)
            if arrayname == "Major Cerebral Artery":
                self.map = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray(i))
            if arrayname == "Volume Flow Rate (mL/s)":
                self.flowdata = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray(i))
            if arrayname == "Colour":
                flowpercluster = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray(i))
                self.PolygonColour = [int(i) for i in flowpercluster]
        #
        # Triangles = []  # triangles
        # for i in range(0, NumberOfPolys):
        #     # 3 points per polygon
        #     # every 4 int from connections
        #     Triangles.append(Connections[i * 4 + 1:i * 4 + 4])

        currentpos = 0
        Polygons = []
        while currentpos < len(Connections):
            currentpoly = Connections[currentpos]
            poly = Connections[currentpos + 1:currentpos + currentpoly + 1]
            currentpos += currentpoly + 1
            Polygons.append(poly)

        links = [set() for i in range(0, NumberOfPoints)]
        for t in Polygons:
            for i in range(0, len(t)):
                node1 = t[i]
                node2 = t[i - 1]
                links[node1].add(node2)
                links[node2].add(node1)

        # for t in Triangles:
        #     t1 = t[0]
        #     t2 = t[1]
        #     t3 = t[2]
        #     links[t1].add(t2)
        #     links[t1].add(t3)
        #     links[t2].add(t1)
        #     links[t2].add(t3)
        #     links[t3].add(t1)
        #     links[t3].add(t2)

        self.PialSurface = pialsurface
        if len(sideinfo) == 0:
            self.DefineSides()
        else:
            self.SideInfo = list(sideinfo)

        self.Triangles = [i for i in Polygons if len(i) == 3]
        self.CalculateAreas()
        self.Polygons = reader.GetOutput().GetPolys()  # polygons
        self.PolygonsList = Polygons
        self.Links = links
        self.ToGraph()

    def LoadSurfaceMSH(self, file="labeled_vol_mesh.msh"):
        """
        Load a msh surface file.
        :param file: msh file
        :return: Nothing
        """
        self.MSH = GeneralFunctions.MSHfile()
        self.MSH.Loadfile(file)
        regionsIDs = [4, 21, 22, 23, 24, 25, 26, 30]
        elements, indexes = self.MSH.GetElements(regionsIDs)

        Trianglesmsh = [[i[-1], i[-2], i[-3]] for i in elements]

        pointids = set()
        for i in Trianglesmsh:
            pointids.add(i[0])
            pointids.add(i[1])
            pointids.add(i[2])

        nodeids = sorted(list(pointids))
        NumberOfPoints = len(nodeids)

        Triangles = [[nodeids.index(i) for i in triangle] for triangle in Trianglesmsh]

        pos = [self.MSH.Nodes[i - 1][1:] for i in nodeids]

        links = [set() for i in range(0, NumberOfPoints)]
        for t in Triangles:
            t1 = t[0]
            t2 = t[1]
            t3 = t[2]
            links[t1].add(t2)
            links[t1].add(t3)
            links[t2].add(t1)
            links[t2].add(t3)
            links[t3].add(t1)
            links[t3].add(t2)

        self.PialSurface = pos
        sideinfo = []
        if len(sideinfo) == 0:
            self.DefineSides()
        else:
            self.SideInfo = list(sideinfo)
        self.Triangles = Triangles
        self.CalculateAreas()
        polygons = vtk.vtkCellArray()
        for poly in Triangles:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(len(poly))
            for i in range(0, len(poly)):
                polygon.GetPointIds().SetId(i, poly[i])
            polygons.InsertNextCell(polygon)
        self.Polygons = polygons
        self.Links = links
        self.ToGraph()

        regiondict = {
            21: 5,
            22: 4,
            23: 7,
            24: 2,
            25: 3,
            26: 6,
            4: 8,
            30: 9
        }

        self.MajorVesselID = [regiondict[i[3]] for i in elements]

    def GetTriangleCentroids(self):
        """
        Calculate the centroid of every triangle of the surface.
        :return: list of centroid coordinates.
        """
        centers = [[] for i in range(0, len(self.Triangles))]
        for index, triangle in enumerate(self.Triangles):
            x = numpy.mean([self.PialSurface[i][0] for i in triangle])
            y = numpy.mean([self.PialSurface[i][1] for i in triangle])
            z = numpy.mean([self.PialSurface[i][2] for i in triangle])
            centers[index] = [x, y, z]
        return centers

    def GetCentersGraph(self, method="vertices"):
        print("Calculating the triangle centers.")
        centers = self.GetTriangleCentroids()

        centergraph = nx.Graph()
        [centergraph.add_node(i) for i in range(0, len(self.Triangles))]

        nodestotriagles = [[] for i in self.PialSurface]
        for index, triangle in enumerate(self.Triangles):
            for node in triangle:
                nodestotriagles[node].append(index)
        print("Calculating the connections.")
        # for each edge find the triangles
        if method == "vertices":
            print("Method: Vertices")
            # method 2: connect center by shared vertex
            links = []
            for index, node in enumerate(nodestotriagles):
                link = list(itertools.permutations(node, 2))
                [links.append(i) for i in link]
            [centergraph.add_edge(i[0], i[1],
                                  weight=GeneralFunctions.distancebetweenpoints(centers[i[0]], centers[i[1]])) for i in
             links]
        else:
            print("Method: Edges")
            # method 1: connect center by shared edges
            edges = [tuple(sorted(s)) for s in list(self.Graph.edges())]
            edgesdict = dict()
            for index, edge in enumerate(edges):
                edgesdict[edge] = index

            triangletoedge = [[] for i in self.Triangles]
            for index, triangle in enumerate(self.Triangles):
                sortedtriangle = sorted(triangle)
                triangletoedge[index] = [edgesdict[i] for i in
                                         [(sortedtriangle[0], sortedtriangle[1]),
                                          (sortedtriangle[0], sortedtriangle[2]),
                                          (sortedtriangle[1], sortedtriangle[2])]]

            edgestotriangle = [[] for i in edges]
            for index, triangle in enumerate(triangletoedge):
                for edge in triangle:
                    edgestotriangle[edge].append(index)
            [centergraph.add_edge(i[0], i[1],
                                  weight=GeneralFunctions.distancebetweenpoints(centers[i[0]], centers[i[1]])) for i in
             edgestotriangle if len(i) > 1]

        # get the links for each point
        # print(list(centergraph.edges()))
        lines = [tuple(sorted(s)) for s in list(centergraph.edges())]
        linkspercenter = [set() for i in range(0, len(centers))]
        for t in lines:
            t1 = t[0]
            t2 = t[1]
            linkspercenter[t1].add(t2)
            linkspercenter[t2].add(t1)

        sideinfotriangles = []
        if len(self.SideInfo) > 0:
            # for each triangle determine the side, since every node of the triangle is on the same side, this is easy
            for index, triangle in enumerate(self.Triangles):
                side = self.SideInfo[triangle[0]]
                sideinfotriangles.append(side)
        else:
            self.DefineSides()

        print("Determining the new polygons.")
        # Calculate the polygons
        polygons = vtk.vtkCellArray()
        # the order of the polygon is imported
        for numberpoly, element in enumerate(nodestotriagles):
            if element:
                # first point is starting point
                np = len(element)
                newelement = [element[0]]
                element.remove(element[0])
                for index in range(1, np):
                    distanceto = [GeneralFunctions.distancebetweenpoints(centers[i], centers[newelement[-1]]) for i in
                                  element]
                    closest = min(range(len(distanceto)), key=lambda k: distanceto[k])
                    newelement.append(element[closest])
                    element.remove(element[closest])
                nodestotriagles[numberpoly] = newelement

        for poly in nodestotriagles:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(len(poly))
            for i in range(0, len(poly)):
                polygon.GetPointIds().SetId(i, poly[i])
            polygons.InsertNextCell(polygon)

        return centergraph, centers, linkspercenter, polygons, sideinfotriangles

    def GraphToVTP(self, folder=""):
        file = folder + self.File
        vesselspolydata = vtk.vtkPolyData()
        # Add all the points
        nodes = vtk.vtkPoints()
        for i in self.PialSurface:
            nodes.InsertNextPoint(i)
        vesselspolydata.SetPoints(nodes)

        vesselspolydata.SetPolys(self.Polygons)
        if len(self.SideInfo) > 0:
            nodetype = vtk.vtkIntArray()
            nodetype.SetName("Side")
            for index, triangle in enumerate(self.PialSurface):
                nodetype.InsertNextValue(self.SideInfo[index])
            vesselspolydata.GetPointData().SetScalars(nodetype)

        if len(self.PolygonColour) > 0:
            Colour = vtk.vtkFloatArray()
            Colour.SetNumberOfComponents(1)
            Colour.SetName("Major Cerebral Artery")
            for colour in self.PolygonColour:
                Colour.InsertNextValue(colour)
            vesselspolydata.GetCellData().AddArray(Colour)

        if len(self.NodeColour) > 0:
            NodeColour = vtk.vtkIntArray()
            NodeColour.SetNumberOfComponents(1)
            NodeColour.SetName("Node Colour")
            for colour in self.NodeColour:
                NodeColour.InsertNextValue(colour)
            vesselspolydata.GetPointData().AddArray(NodeColour)

        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputData(vesselspolydata)
        cleanPolyData.Update()
        PolyData= cleanPolyData.GetOutput()

        print("Saving to %s" % file)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(file)
        writer.SetInputData(PolyData)
        writer.Write()

    def GraphEdgesToVTP(self, file="GraphEdges.vtp"):
        vesselspolydata = vtk.vtkPolyData()
        # Add all the points
        nodes = vtk.vtkPoints()
        for i in self.PialSurface:
            nodes.InsertNextPoint(i)
        vesselspolydata.SetPoints(nodes)

        # vesselspolydata.SetPolys(self.Polygons)
        # if len(self.SideInfo) > 0:
        #     nodetype = vtk.vtkIntArray()
        #     nodetype.SetName("Side")
        #     for index, triangle in enumerate(self.PialSurface):
        #         nodetype.InsertNextValue(self.SideInfo[index])
        #     vesselspolydata.GetPointData().SetScalars(nodetype)

        # if len(self.PolygonColour) > 0:
        #     Colour = vtk.vtkFloatArray()
        #     Colour.SetNumberOfComponents(1)
        #     Colour.SetName("Major Cerebral Artery")
        #     for colour in self.PolygonColour:
        #         Colour.InsertNextValue(colour)
        #     vesselspolydata.GetCellData().AddArray(Colour)

        lines = vtk.vtkCellArray()
        for index, link in enumerate(self.Links):
            for line in link:
                lines.InsertNextCell(2)
                lines.InsertCellPoint(index)
                lines.InsertCellPoint(line)

        vesselspolydata.SetLines(lines)

        if len(self.NodeColour) > 0:
            NodeColour = vtk.vtkIntArray()
            NodeColour.SetNumberOfComponents(1)
            NodeColour.SetName("Node Colour")
            for colour in self.NodeColour:
                NodeColour.InsertNextValue(colour)
            vesselspolydata.GetPointData().AddArray(NodeColour)

        print("Saving to %s" % file)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(file)
        writer.SetInputData(vesselspolydata)
        writer.Write()


class Perfusion:
    def __init__(self):
        self.CouplingPoints = []  # these are the nodes that connect to the pial surface with their 3d position
        self.PrimalGraph = Surface()
        self.PrimalGraph.File = "PrimalGraph.vtp"
        self.DualGraph = Surface()
        self.DualGraph.File = "DualGraph.vtp"

        self.regionids = [2, 3, 4, 5, 6, 7]
        self.leftregionids = [4, 5, 7]
        self.rightregionids = [2, 3, 6]

        self.TMatrix = None
        self.AllignmentResult = None

    def AddCouplingPoint(self, couplingpoint):
        self.CouplingPoints.append(couplingpoint)

    def RemoveCouplingPoint(self, couplingpoint):
        self.CouplingPoints.remove(couplingpoint)

    def LoadPrimalGraph(self, file):
        self.PrimalGraph.LoadSurface(file)

    def SetClusterIDs(self):
        for index, cp in enumerate(self.CouplingPoints):
            cp.ClusterID = index

    def ClusterArea(self, graph):
        for index, cp in enumerate(self.CouplingPoints):
            cp.SurfaceNodes = [i for i, label in enumerate(self.DualGraph.NodeColour) if label == index]
            cp.Area = sum([graph.Areas[i] for i in cp.SurfaceNodes])
            cp.AreaSampling = sum([graph.Areas[i] for i in cp.Pialpoints])

    def FindNearestSurfaceNodeSides(self):
        """
        Map the outlets to the nearest point on the surface.
        """
        # pos = [i.Node.Position for i in self.CouplingPoints]
        # euclidean distance between outlets and surface
        leftvessels = self.leftregionids
        rightvessels = self.rightregionids

        sys.setrecursionlimit(10000)
        # left and right sides of the pial surface
        rightpialsurface = [index for index, node in enumerate(self.DualGraph.PialSurface) if
                            self.DualGraph.SideInfo[index] > 0]
        leftpialsurface = [index for index, node in enumerate(self.DualGraph.PialSurface) if
                           self.DualGraph.SideInfo[index] < 0]
        KDTreeleft = scipy.spatial.KDTree([self.DualGraph.PialSurface[i] for i in leftpialsurface])
        KDTreeright = scipy.spatial.KDTree([self.DualGraph.PialSurface[i] for i in rightpialsurface])
        KDTree = scipy.spatial.KDTree(self.DualGraph.PialSurface)

        print("Finding outlets close to the surface.")
        for index, node in enumerate(self.CouplingPoints):
            if node.Node.MajorVesselID in leftvessels:
                _, MinDistanceIndex = KDTreeleft.query(node.Node.Position, k=1)
                index = leftpialsurface[MinDistanceIndex]
            elif node.Node.MajorVesselID in rightvessels:
                _, MinDistanceIndex = KDTreeright.query(node.Node.Position, k=1)
                index = rightpialsurface[MinDistanceIndex]
            else:
                _, MinDistanceIndex = KDTree.query(node.Node.Position, k=1)
                index = MinDistanceIndex
            node.PialSurfacePointID = index
        self.Roots = [node.PialSurfacePointID for node in self.CouplingPoints]

    def FindNearestSurfaceNodeVEASL(self):
        """
        Map the outlets to the nearest point on the surface.
        """
        majorvesselids = set(self.DualGraph.MajorVesselID)
        print("Finding outlets close to the surface.")
        for majorid in list(majorvesselids):
            veaslnodes = [i for i, pos in enumerate(self.DualGraph.PialSurface) if
                          self.DualGraph.MajorVesselID[i] == majorid]
            veasldata = [self.DualGraph.PialSurface[i] for i in veaslnodes]
            KDTree = scipy.spatial.KDTree(veasldata)

            couplingpoints = [cp for cp in self.CouplingPoints if cp.Node.MajorVesselID == majorid]

            pos = [i.Node.Position for i in couplingpoints]
            # euclidean distance between outlets and surface
            MinDistance, MinDistanceIndex = KDTree.query(pos, k=1)

            for index, node in enumerate(couplingpoints):
                node.PialSurfacePointID = veaslnodes[MinDistanceIndex[index]]

        self.Roots = [node.PialSurfacePointID for node in self.CouplingPoints]

    def OutputClusteringSteps(self, file="OutputSteps.vtp", rootsfile="RootsOutputSteps.vtp"):
        """
        Output the clustering steps
        :param file: File to write the clustering of the surface to, per iteration
        :param rootsfile: File to write the path of the root of each cluster to.
        :return: Nothing
        """
        print("Writing steps of clustering to file.")
        VesselsPolyData = vtk.vtkPolyData()
        nodes = vtk.vtkPoints()
        for index, item in enumerate(self.PrimalGraph.PialSurface):
            nodes.InsertNextPoint(item)
        VesselsPolyData.SetPoints(nodes)
        VesselsPolyData.SetPolys(self.PrimalGraph.Polygons)

        Colour = vtk.vtkFloatArray()
        Colour.SetNumberOfComponents(1)
        Colour.SetName("VEASL Major Cerebral Artery")
        for colour in self.PrimalGraph.map:
            Colour.InsertNextValue(colour)
        VesselsPolyData.GetCellData().AddArray(Colour)

        regionpoints = self.ClusteringResults[0]
        for index, result in enumerate(self.ClusteringResults[1]):
            root = vtk.vtkIntArray()
            root.SetNumberOfComponents(1)
            root.SetName("Clustering: %d" % index)

            MajorArteries = vtk.vtkIntArray()
            MajorArteries.SetNumberOfComponents(1)
            MajorArteries.SetName("Clustering Major Vessel: %d" % index)

            for element, _ in enumerate(self.PrimalGraph.Triangles):
                if element in regionpoints:
                    colour = result[0][regionpoints.index(element)]
                    colourMajorID = self.CouplingPoints[colour].Node.MajorVesselID
                else:
                    colour = -1
                    colourMajorID = -1
                root.InsertNextValue(colour)
                MajorArteries.InsertNextValue(colourMajorID)

            VesselsPolyData.GetCellData().AddArray(root)
            VesselsPolyData.GetCellData().AddArray(MajorArteries)

        writer = vtk.vtkXMLPolyDataWriter()
        print("Writing Clustering to file: %s" % file)
        writer.SetFileName(file)
        writer.SetInputData(VesselsPolyData)
        writer.Write()

        print("Writing root steps of clustering to file.")
        VesselsPolyData2 = vtk.vtkPolyData()
        nodes = vtk.vtkPoints()
        nodes.SetNumberOfPoints(len(self.DualGraph.PialSurface))
        for index, item in enumerate(self.DualGraph.PialSurface):
            nodes.SetPoint(index, item)
        VesselsPolyData2.SetPoints(nodes)

        lines = vtk.vtkCellArray()
        for cp in self.CouplingPoints:
            NumberOfIds = len(cp.RootPos)
            lines.InsertNextCell(NumberOfIds)
            if NumberOfIds == 1:
                lines.InsertCellPoint(cp.RootPos[0])
            else:
                for index, pos in enumerate(cp.RootPos):
                    lines.InsertCellPoint(pos)

        VesselsPolyData2.SetLines(lines)

        atlas = vtk.vtkIntArray()
        [atlas.InsertNextValue(index) for index, i in enumerate(self.CouplingPoints)]
        atlas.SetName("Root Ids")
        VesselsPolyData2.GetCellData().SetScalars(atlas)

        atlas = vtk.vtkIntArray()
        [atlas.InsertNextValue(i.Node.MajorVesselID) for index, i in enumerate(self.CouplingPoints)]
        atlas.SetName("Root Major Vessel Ids")
        VesselsPolyData2.GetCellData().SetScalars(atlas)

        writer = vtk.vtkXMLPolyDataWriter()
        print("Writing Roots to file: %s" % rootsfile)
        writer.SetFileName(rootsfile)
        writer.SetInputData(VesselsPolyData2)
        writer.Write()

    def WriteClusteringMapping(self, folder):
        file = folder + "Clusters.csv"
        print("Writing Clustering map to file: %s" % file)
        with open(file, 'w') as f:
            f.write(
                "NodeID,Position,Number of CouplingPoints,ClusterID,Area,MajorVesselID\n")
            for index, couplingpoint in enumerate(self.CouplingPoints):
                number = couplingpoint.Node.Number
                position = couplingpoint.Node.Position
                numbernodes = couplingpoint.NumberOfPialPoints
                line = ",".join((str(number),
                                 (str(position[0]) + " " + str(position[1]) + " " + str(position[2])),
                                 str(numbernodes),
                                 str(index),
                                 str(couplingpoint.Area),
                                 str(couplingpoint.Node.MajorVesselID),
                                 # ";".join([str(i) for i in couplingpoint.Pialpoints])))
                                 ))
                f.write(line)
                f.write("\n")

    def UpdateMappedRegionsFlowdata(self, file):
        with open(file, 'w') as f:
            # f.write("# region I,Q [ml/s],p [Pa],feeding artery ID\n")
            f.write("# region I,Q [ml/s],p [Pa],feeding artery ID,BC: p->0 or Q->1\n")
            for index, i in enumerate(self.CouplingPoints):
                if i.Node.R1 is None:
                    f.write("%d,%.16g,%.16g,%d,%d\n" % (
                        Constants.StartClusteringIndex + index, i.Node.AccumulatedFlowRate, i.Node.Pressure,
                        Constants.MajorIDdict[i.Node.MajorVesselID], 0))
                else:
                    f.write("%d,%.16g,%.16g,%d,%d\n" % (
                        Constants.StartClusteringIndex + index, i.Node.WKNode.AccumulatedFlowRate, i.Node.Pressure,
                        Constants.MajorIDdict[i.Node.MajorVesselID], 0))

    def SetDualGraph(self, method="vertices"):
        print("Calculating the dual graph.")
        graph, centers, links, poly, sideinfotriangles = self.PrimalGraph.GetCentersGraph(method=method)
        self.DualGraph.Graph = graph
        self.DualGraph.PialSurface = centers
        self.DualGraph.Links = links
        self.DualGraph.Polygons = poly
        self.DualGraph.SideInfo = sideinfotriangles
        self.DualGraph.Weights = self.PrimalGraph.Areas

        ids = []
        idList = vtk.vtkIdList()
        poly.InitTraversal()
        while poly.GetNextCell(idList):
            currentpoly = []
            for i in range(0, idList.GetNumberOfIds()):
                pId = idList.GetId(i)
                currentpoly.append(pId)
            ids.append(currentpoly)

        self.PolygonsList = ids
        self.DualGraph.Areas = [GeneralFunctions.poly_area([self.DualGraph.PialSurface[i] for i in poly]) for poly in
                                ids]

    def MapDualGraphToPrimalGraph(self):
        self.PrimalGraph.PolygonColour = self.DualGraph.NodeColour
        self.ClusterArea(self.PrimalGraph)

    def SquareLatticeCP(self, clusters):
        self.PrimalGraph.MajorVesselID = [2 for node in self.PrimalGraph.PialSurface]
        self.CouplingPoints = [CouplingPoint(Node.Node()) for node in range(0, clusters)]
        nodes = numpy.random.choice(len(self.PrimalGraph.PialSurface), size=clusters, replace=False)
        for index, node in enumerate(self.CouplingPoints):
            node.Node.SetPosition(self.PrimalGraph.PialSurface[nodes[index]])
            node.NumberOfPialPoints = int(1 * len(self.PrimalGraph.PialSurface) / clusters)
            node.Node.MajorVesselID = 2
        self.regionids = [2]

    def SquareLattice(self, graph, nn):
        print("generating square lattice for testing")
        lattice = nx.generators.lattice.grid_2d_graph(nn, nn)
        # graph.PialSurface = [[0, numpy.random.normal()*0.1+i // nn, numpy.random.normal()*0.1+i % nn] for i in range(0, nn * nn)]
        graph.PialSurface = [[0, i // nn, i % nn] for i in
                             range(0, nn * nn)]
        # graph.Graph = nx.Graph()
        links = [set() for i in range(0, nn * nn)]
        for edge in lattice.edges():
            u, v = edge
            node1 = u[0] * nn + u[1]
            node2 = v[0] * nn + v[1]
            links[node1].add(node2)
            links[node2].add(node1)

        d = numpy.sqrt(2)
        graph.Graph = lattice
        for n, p in enumerate(graph.PialSurface):
            graph.Graph.node[tuple(p[1:])]['pos'] = p[1:]

        for index, node in enumerate(graph.PialSurface):
            if not (index <= nn or index > nn * nn - nn or index % nn == 0 or index % nn == nn - 1):
                # graph.Graph.add_edge(index, index + nn + 1, weight=d)
                # graph.Graph.add_edge(index, index + nn - 1, weight=d)
                # graph.Graph.add_edge(index, index - nn + 1, weight=d)
                # graph.Graph.add_edge(index, index - nn - 1, weight=d)
                links[index].add(index + nn + 1)
                links[index].add(index + nn - 1)
                links[index].add(index - nn + 1)
                links[index].add(index - nn - 1)
                links[index + nn + 1].add(index)
                links[index + nn - 1].add(index)
                links[index - nn + 1].add(index)
                links[index - nn - 1].add(index)
                # graph.Graph.add_edge(index + nn + 1, index, weight=d)
                # graph.Graph.add_edge(index + nn - 1, index, weight=d)
                # graph.Graph.add_edge(index - nn + 1, index , weight=d)
                # graph.Graph.add_edge(index - nn - 1, index , weight=d)

        # graph.Graph.add_edge(1, nn, weight=d)
        # graph.Graph.add_edge(nn - 2, 2 * nn - 1, weight=d)
        # graph.Graph.add_edge(nn * nn - 2, nn * nn - 1 - nn, weight=d)
        # graph.Graph.add_edge(nn * nn - 2 * nn, nn * nn - 1 * nn + 1, weight=d)

        # graph.Graph.add_edge(nn, 1, weight=d)
        # graph.Graph.add_edge(2 * nn - 1, nn - 2, weight=d)
        # graph.Graph.add_edge(nn * nn - 1 - nn,nn * nn - 2, weight=d)
        # graph.Graph.add_edge(nn * nn - 1 * nn + 1, nn * nn - 2 * nn, weight=d)

        links[1].add(nn)
        links[nn].add(1)
        links[nn - 2].add(2 * nn - 1)
        links[2 * nn - 1].add(nn - 2)
        links[nn * nn - 1 - nn].add(nn * nn - 2)
        links[nn * nn - 2].add(nn * nn - 1 - nn)
        links[nn * nn - 1 * nn + 1].add(nn * nn - 2 * nn)
        links[nn * nn - 2 * nn].add(nn * nn - 1 * nn + 1)

        graph.Links = [list(i) for i in links]
        graph.Areas = [1 for i in graph.PialSurface]  # equal area for all
        graph.Weights = [1 for i in graph.PialSurface]  # equal area for all

        for index, element in enumerate(links):
            for index2, element2 in enumerate(element):
                node1 = tuple(graph.PialSurface[index][1:])
                node2 = tuple(graph.PialSurface[element2][1:])
                d = GeneralFunctions.distancebetweenpoints(node1, node2)
                graph.Graph.add_edge(node1, node2, weight=d)

        # for index, element in enumerate(graph.PialSurface):
        #     for index2, element2 in enumerate(graph.PialSurface):
        #         node1 = tuple(graph.PialSurface[index][1:])
        #         node2 = tuple(graph.PialSurface[index2][1:])
        #         d = distancebetweenpoints(node1,node2)
        #         graph.Graph.add_edge(node1,node2, weight=d)

        # import matplotlib.pyplot as plt
        # labels = nx.get_edge_attributes(graph.Graph, 'weight')
        # pos = nx.get_node_attributes(graph.Graph, 'pos')
        # nx.draw(graph.Graph,pos)
        # # nx.draw_networkx_edge_labels(graph.Graph, edge_labels=labels)
        # plt.show()

    def ExportSurface(self, file, graph):
        nodes = vtk.vtkPoints()
        for index, item in enumerate(graph.PialSurface):
            nodes.InsertNextPoint(item)

        Colour = vtk.vtkFloatArray()
        Colour.SetNumberOfComponents(1)
        Colour.SetName("Colour")
        for colour in graph.PolygonColour:
            Colour.InsertNextValue(colour)

        MajorArteries = vtk.vtkIntArray()
        MajorArteries.SetNumberOfComponents(1)
        MajorArteries.SetName("Major Cerebral Artery")
        for i in range(0, len(graph.PolygonColour)):
            regionid = graph.PolygonColour[i]
            couplingpoint = self.CouplingPoints[regionid]
            MajorArteries.InsertNextValue(couplingpoint.Node.MajorVesselID)

        VesselsPolyData = vtk.vtkPolyData()
        VesselsPolyData.SetPoints(nodes)
        VesselsPolyData.SetPolys(graph.Polygons)
        VesselsPolyData.GetCellData().AddArray(Colour)
        VesselsPolyData.GetCellData().AddArray(MajorArteries)

        writer = vtk.vtkXMLPolyDataWriter()
        print("Writing Clustering to file: %s" % file)
        writer.SetFileName(file)
        writer.SetInputData(VesselsPolyData)
        writer.Write()

    def ExportSurfacePoints(self, file, graph):
        nodes = vtk.vtkPoints()
        for index, item in enumerate(graph.PialSurface):
            nodes.InsertNextPoint(item)

        Colour = vtk.vtkFloatArray()
        Colour.SetNumberOfComponents(1)
        Colour.SetName("Colour")
        for index, colour in enumerate(graph.PolygonColour):
            # if index in graph.Roots:
            #     Colour.InsertNextValue(-1)
            # else:
            Colour.InsertNextValue(colour)

        MajorArteries = vtk.vtkIntArray()
        MajorArteries.SetNumberOfComponents(1)
        MajorArteries.SetName("Major Cerebral Artery")
        for i in range(0, len(graph.PolygonColour)):
            regionid = graph.PolygonColour[i]
            couplingpoint = self.CouplingPoints[regionid]
            MajorArteries.InsertNextValue(couplingpoint.Node.MajorVesselID)

        lines = vtk.vtkCellArray()
        for index, element in enumerate(graph.Links):
            for otherelement in element:
                line0 = vtk.vtkLine()
                line0.GetPointIds().SetId(0, index)
                line0.GetPointIds().SetId(1, otherelement)
                lines.InsertNextCell(line0)

        VesselsPolyData = vtk.vtkPolyData()
        VesselsPolyData.SetPoints(nodes)
        # VesselsPolyData.SetPolys(graph.Polygons)
        VesselsPolyData.GetPointData().AddArray(Colour)
        VesselsPolyData.GetPointData().AddArray(MajorArteries)
        VesselsPolyData.SetLines(lines)

        writer = vtk.vtkXMLPolyDataWriter()
        print("Writing Clustering to file: %s" % file)
        writer.SetFileName(file)
        writer.SetInputData(VesselsPolyData)
        writer.Write()

    def Clustering(self, dualgraph, distancemat="Distancemat.npy", method="", fractiontriangles=1, debug=False,
                   maxiter=0, useregion=False):
        """

        :param dualgraph:
        :param distancemat:
        :param method:
        :param fractiontriangles:
        :param debug:
        :param maxiter:
        :return:
        """
        print("Start clustering.")
        # copy paste and small changes from clusteringbyscalinglaw, seems fasted to copy-paste
        if useregion:
            self.FindNearestSurfaceNodeVEASL()
        else:
            # dualgraph.FindNearestSurfaceNode(self.CouplingPoints)
            self.FindNearestSurfaceNodeSides()
        dualgraph.CalculateScipyGraph()

        regionsum = len([i for i, x in enumerate(dualgraph.MajorVesselID)])
        regionroots = [node for node in self.CouplingPoints]

        TotalTriangles = int(regionsum * fractiontriangles)
        inputRadii = [i.Node.Radius for i in regionroots]
        RtCubed = sum([numpy.power(i, 3) for i in inputRadii])

        fractionArea = [numpy.power(i, 3) / RtCubed for i in inputRadii]

        numberPialSurfaceElements = [i * TotalTriangles for i in fractionArea]

        roundedNumberPialSurfaceElements = [int(numpy.floor(i)) for i in numberPialSurfaceElements]
        for index, node in enumerate(regionroots):
            node.NumberOfPialPoints = max(1, roundedNumberPialSurfaceElements[index])

        rootsum = sum(roundedNumberPialSurfaceElements)

        print("Number of coupling points: " + str(
            rootsum) + ", number of surface elements: " + str(regionsum))
        if rootsum > regionsum:
            errorline = "Error: Number of coupling points: " + str(
                rootsum) + " exceeds the number of surface elements: " + str(regionsum)
            raise Exception(errorline)

        region = [i for i, x in enumerate(dualgraph.MajorVesselID)]
        NRootsLabels = [node.NumberOfPialPoints for node in regionroots]
        PointsN = sum(NRootsLabels)
        RootsN = len(regionroots)
        regionpoints = dualgraph.WeightedSampling(region, PointsN)
        roots = [point.PialSurfacePointID for point in regionroots]

        # calculate distance matrix for the pial surface.
        if GeneralFunctions.is_non_zero_file(distancemat):
            print("Loading Distance Matrix.")
            CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='r',
                                               shape=(len(dualgraph.PialSurface), len(dualgraph.PialSurface)))
        else:
            CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='w+',
                                               shape=(len(dualgraph.MajorVesselID), len(dualgraph.MajorVesselID)))
            print("Allocated diskspace for the matrix.")
            # CompleteDijkstraMap = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False)
            for i in range(0, len(dualgraph.MajorVesselID)):
                CompleteDijkstraMap[:, i] = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False,
                                                                          indices=i)

        print("Optimizing clustering.")
        iter = 0
        resultcluster = []
        while 1:
            # flat_list = [item for sublist in DistanceMat for item in sublist]
            AssignedLabelsN = [0 for i in range(0, RootsN)]
            Labels = [-1 for i in range(0, PointsN)]

            # sortedindex = sorted(range(len(flat_list)), key=lambda k: flat_list[k])

            # flat_list.sort(range(len(flat_list)), key=lambda k: flat_list[k])
            # flat_list[:] = numpy.argsort(DistanceMat)
            for pointnumber, point in enumerate(regionpoints):
                distancetoroots = CompleteDijkstraMap[roots, point]
                # distancetoroots = numpy.argsort(distancetoroots)
                distancetoroots = sorted(range(len(distancetoroots)), key=lambda k: distancetoroots[k])
                for clusterid, _ in enumerate(distancetoroots):
                    if AssignedLabelsN[clusterid] < NRootsLabels[clusterid] and Labels[pointnumber] == -1:
                        # if Labels[PointIndex] == -1: # no constraint
                        Labels[pointnumber] = clusterid
                        AssignedLabelsN[clusterid] += 1

            swapiter = 0
            while 1:
                swaplist = []
                for point in range(0, PointsN):
                    # possiblelabels = DistanceMat[:, point]
                    possiblelabels = CompleteDijkstraMap[roots, point]
                    # possiblelabels = CompleteDijkstraMap[numpy.ix_(roots, regionpoints[point])]
                    # sortedlabels = sorted(range(len(possiblelabels)), key=lambda k: possiblelabels[k])
                    # sortedlabels = numpy.argsort(possiblelabels)
                    sortedlabels = sorted(range(len(possiblelabels)), key=lambda k: possiblelabels[k])
                    currentlabel = Labels[point]
                    if currentlabel != sortedlabels[0]:
                        swaplist.append(point)
                changecounter = 0
                for swap in swaplist:
                    differenceafterswap = [(CompleteDijkstraMap[roots[Labels[swap]], regionpoints[swap]] +
                                            CompleteDijkstraMap[roots[Labels[i]], regionpoints[i]])
                                           - (CompleteDijkstraMap[roots[Labels[i]], regionpoints[swap]] +
                                              CompleteDijkstraMap[roots[Labels[swap]], regionpoints[i]])
                                           for i in range(0, PointsN)]

                    # differenceafterswap = [(DistanceMat[Labels[swap], swap] + DistanceMat[Labels[i], i])
                    #                        - (DistanceMat[Labels[i], swap] + DistanceMat[Labels[swap], i])
                    #                        for i in range(0, PointsN)]

                    differencebyindex = max(range(len(differenceafterswap)), key=lambda k: differenceafterswap[k])
                    if differenceafterswap[differencebyindex] > 0:
                        Labels[differencebyindex], Labels[swap] = Labels[swap], Labels[differencebyindex]
                        changecounter += 1
                print("Total changes: %d" % changecounter)
                swapiter += 1
                if changecounter == 0 or swapiter >= 10:
                    break

            stopcrit = 0

            print("Recalculating roots.")
            clusterelements = [[i for i in range(0, len(Labels)) if Labels[i] == element] for element
                               in range(0, RootsN)]

            for index, cluster in enumerate(clusterelements):

                distancematforcluster = numpy.empty([len(cluster), len(cluster)])
                for num, point in enumerate(cluster):
                    for num2, point2 in enumerate(cluster):
                        # distancematforcluster[num, num2] = PialDMat[point, point2]
                        distancematforcluster[num, num2] = CompleteDijkstraMap[
                            regionpoints[point], regionpoints[point2]]
                distances = numpy.amax(distancematforcluster, axis=0)
                newguess = min(range(0, len(distances)), key=lambda k: distances[k])
                newguessPoint = cluster[newguess]
                # new guess for root node is the Jardan Center of the cluster
                # calculate the change in position of the root
                newpos = dualgraph.PialSurface[regionpoints[newguessPoint]]
                oldpos = dualgraph.PialSurface[roots[index]]
                diff = GeneralFunctions.distancebetweenpoints(oldpos, newpos)
                stopcrit += diff
                roots[index] = regionpoints[newguessPoint]

            print("Root displacement: %f" % stopcrit)
            iter += 1
            # resultcluster.append((Labels, copy.deepcopy(roots)))
            resultcluster.append((copy.deepcopy(Labels), copy.deepcopy(roots)))
            if stopcrit == 0 or iter >= maxiter:
                break
            # DistanceMat = CompleteDijkstraMap[numpy.ix_(roots, regionpoints)]

        ##########################################################################
        # final result from clustering
        self.ClusteringResults = (regionpoints, resultcluster)
        dualgraph.Points = regionpoints
        dualgraph.Labels = resultcluster[-1][0]

        for index, cp in enumerate(self.CouplingPoints):
            cp.RootPos = [i[1][index] for i in resultcluster]
            points = [dualgraph.Points[i] for i in range(0, len(dualgraph.Labels)) if dualgraph.Labels[i] == index]
            cp.Pialpoints = points

        # For each outlet, save the connected pial points
        for iter in range(0, len(self.CouplingPoints)):
            points = [dualgraph.Points[i] for i in range(0, len(dualgraph.Labels)) if dualgraph.Labels[i] == iter]
            self.CouplingPoints[iter].Pialpoints = points

        # Colour the mesh with the nearest coupling point
        # Colouring the mesh
        meshcolour = [-1 for node in dualgraph.PialSurface]
        for i in range(0, len(dualgraph.Points)):
            point = dualgraph.Points[i]
            meshcolour[point] = dualgraph.Labels[i]

        iter = 0
        while 1:
            nocolour = 0
            colourupdate = []
            for index, item in enumerate(meshcolour):
                if item == -1:
                    colouredlinkednodes = [i for i in dualgraph.Links[index] if meshcolour[i] >= 0]
                    if len(colouredlinkednodes) > 0:
                        # calculate distance to each linked node and take the nearest node's colour
                        distancetocolourednodes = [
                            GeneralFunctions.distancebetweenpoints(dualgraph.PialSurface[index],
                                                                   dualgraph.PialSurface[p])
                            for p in colouredlinkednodes]

                        mindifferencebyindex = min(range(len(distancetocolourednodes)),
                                                   key=lambda k: distancetocolourednodes[k])
                        # update after the for loop
                        colourupdate.append((index, colouredlinkednodes[mindifferencebyindex]))
                    else:
                        nocolour += 1
            for element in colourupdate:
                meshcolour[element[0]] = meshcolour[element[1]]
            iter += 1
            if nocolour == 0 or iter == 1000:
                break
        dualgraph.NodeColour = meshcolour

    def ClusteringByRegion(self, dualgraph, distancemat="Distancemat.npy", method="",
                           fractiontriangles=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5), debug=False, maxiter=0,
                           useregion=False,scaling_factor=3):
        """
        Algorithm to estimate perfusion territories.
        :param dualgraph: Graph to use for the clustering. Clustering happens with the vertices of the graph, clustering of triangles requires the dual graph.
        :param method: Default option: Dijkstra, Other option: 'euclidean'
        :param fractiontriangles: Sets the fraction of triangles to use for the initial clustering. Mainly for performance reasons.
        :param debug: Default:False. True sets the number of points per cluster to one. Use this to get a quickyclustering to the nearst outlet.
        :return: Clustering result is stored under dualgraph.NodeColour. See MapDualGraphToPrimalGraph() for mapping back to the primal graph.
        """
        print("Start clustering.")
        resultcluster = [[] for i in self.regionids]

        if useregion:
            self.FindNearestSurfaceNodeVEASL()
        else:
            # dualgraph.FindNearestSurfaceNode(self.CouplingPoints)
            self.FindNearestSurfaceNodeSides()
        dualgraph.CalculateScipyGraph()

        if GeneralFunctions.is_non_zero_file(distancemat):
            print("Loading Distance Matrix.")
            CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='r',
                                               shape=(len(dualgraph.PialSurface), len(dualgraph.PialSurface)))
        else:
            CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='w+',
                                               shape=(len(dualgraph.MajorVesselID), len(dualgraph.MajorVesselID)))
            print("Allocated diskspace for the matrix.")
            # CompleteDijkstraMap = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False)
            for i in range(0, len(dualgraph.MajorVesselID)):
                CompleteDijkstraMap[:, i] = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False,
                                                                          indices=i)

        for regionindex, currentregion in enumerate(self.regionids):
            regionsum = len([i for i, x in enumerate(dualgraph.MajorVesselID) if x == currentregion])
            regionroots = [node for node in self.CouplingPoints if node.Node.MajorVesselID == currentregion]
            # rootsum = sum([node.NumberOfPialPoints for node in regionroots])

            TotalTriangles = int(regionsum * fractiontriangles[regionindex])
            # print("Using %d of the %d triangles for region %d" % (TotalTriangles, regionsum, currentregion))
            inputRadii = [i.Node.Radius for i in regionroots]
            RtCubed = sum([numpy.power(i, scaling_factor) for i in inputRadii])

            fractionArea = [numpy.power(i, scaling_factor) / RtCubed for i in inputRadii]

            numberPialSurfaceElements = [i * TotalTriangles for i in fractionArea]

            roundedNumberPialSurfaceElements = [int(numpy.floor(i)) for i in numberPialSurfaceElements]
            for index, node in enumerate(regionroots):
                node.NumberOfPialPoints = roundedNumberPialSurfaceElements[index]

            rootsum = sum(roundedNumberPialSurfaceElements)
            # if sum(roundedNumberPialSurfaceElements) > regionsum:
            #     print("Error in the number of PialSurfaceElements: too many assigned")

            # if rootsum == len(regionroots):
            #     print("Setting number of surface elements to 50 per root.")
            #     for node in regionroots:
            #         node.NumberOfPialPoints = 50
            #     rootsum = len(regionroots) * 50
            if debug:
                print("Debug Mode.")
                for index, node in enumerate(regionroots):
                    node.NumberOfPialPoints = 1

            print("RegionID: " + str(currentregion) + ", Number of coupling points: " + str(
                rootsum) + ", number of surface elements: " + str(regionsum))
            if rootsum > regionsum:
                errorline = "Error: Number of coupling points: " + str(
                    rootsum) + " exceeds the number of surface elements: " + str(regionsum)
                raise Exception(errorline)

        for regionindex, currentregion in enumerate(self.regionids):
            region = [i for i, x in enumerate(dualgraph.MajorVesselID) if x == currentregion]
            regionroots = [node for node in self.CouplingPoints if node.Node.MajorVesselID == currentregion]
            NRootsLabels = [node.NumberOfPialPoints for node in regionroots]
            PointsN = sum(NRootsLabels)  # total points
            RootsN = len(regionroots)
            regionpoints = dualgraph.WeightedSampling(region, PointsN)

            # calculate distance matrix for the pial surface.
            roots = [point.PialSurfacePointID for point in regionroots]
            # PialDMat = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False, indices=regionpoints)
            # PialDMat = PialDMat[:, regionpoints]

            print("Current region: %d" % currentregion)
            # PialDMat = CompleteDijkstraMap[numpy.ix_(regionpoints, regionpoints)]
            # DistanceMat = CompleteDijkstraMap[numpy.ix_(roots, regionpoints)]

            # print("Optimizing clustering.")
            # DistanceMat = dualgraph.CalculateDistanceMatRegion(method, roots, regionpoints)
            iter = 0
            # rootpositions = []
            while 1:
                # save the roots during this iteration to plot their trajectory.
                # rootpositions.append(copy.deepcopy(roots))

                # flat_list = [item for sublist in DistanceMat for item in sublist]
                AssignedLabelsN = [0 for i in range(0, RootsN)]
                Labels = [-1 for i in range(0, PointsN)]
                for pointnumber, point in enumerate(regionpoints):
                    distancetoroots = CompleteDijkstraMap[roots, point]
                    # distancetoroots = numpy.argsort(distancetoroots)
                    distancetoroots = sorted(range(len(distancetoroots)), key=lambda k: distancetoroots[k])
                    for clusterid, _ in enumerate(distancetoroots):
                        if AssignedLabelsN[clusterid] < NRootsLabels[clusterid] and Labels[pointnumber] == -1:
                            # if Labels[PointIndex] == -1: # no constraint
                            Labels[pointnumber] = clusterid
                            AssignedLabelsN[clusterid] += 1
                # sortedindex = sorted(range(len(flat_list)), key=lambda k: flat_list[k])
                #
                # for index in sortedindex:
                #     ClusterIndex = index // PointsN
                #     PointIndex = index % PointsN
                #
                #     if AssignedLabelsN[ClusterIndex] < NRootsLabels[ClusterIndex] and Labels[PointIndex] == -1:
                #         # if Labels[PointIndex] == -1:
                #         Labels[PointIndex] = ClusterIndex
                #         AssignedLabelsN[ClusterIndex] += 1

                swapiter = 0
                while 1:
                    swaplist = []
                    for point in range(0, PointsN):
                        # possiblelabels = DistanceMat[:, point]
                        possiblelabels = CompleteDijkstraMap[roots, point]
                        sortedlabels = sorted(range(len(possiblelabels)), key=lambda k: possiblelabels[k])
                        currentlabel = Labels[point]
                        if currentlabel != sortedlabels[0]:
                            swaplist.append(point)
                    changecounter = 0
                    for swap in swaplist:
                        differenceafterswap = [(CompleteDijkstraMap[roots[Labels[swap]], regionpoints[swap]] +
                                                CompleteDijkstraMap[roots[Labels[i]], regionpoints[i]])
                                               - (CompleteDijkstraMap[roots[Labels[i]], regionpoints[swap]] +
                                                  CompleteDijkstraMap[roots[Labels[swap]], regionpoints[i]])
                                               for i in range(0, PointsN)]
                        # differenceafterswap = [(DistanceMat[Labels[swap], swap] + DistanceMat[Labels[i], i])
                        #                        - (DistanceMat[Labels[i], swap] + DistanceMat[Labels[swap], i])
                        #                        for i in range(0, PointsN)]

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
                    # newpos = numpy.mean([dualgraph.PialSurface[regionpoints[i]] for i in cluster], axis=0)
                    # diff = distancebetweenpoints(roots[index], newpos)
                    # stopcrit += diff
                    # roots[index] = newpos

                    distancematforcluster = numpy.empty([len(cluster), len(cluster)])
                    for num, point in enumerate(cluster):
                        for num2, point2 in enumerate(cluster):
                            # distancematforcluster[num, num2] = PialDMat[point, point2]
                            distancematforcluster[num, num2] = CompleteDijkstraMap[
                                regionpoints[point], regionpoints[point2]]
                    distances = numpy.amax(distancematforcluster, axis=0)
                    newguess = min(range(0, len(distances)), key=lambda k: distances[k])
                    newguessPoint = cluster[newguess]
                    # new guess for root node is the Jardan Center of the cluster
                    # calculate the change in position of the root
                    newpos = dualgraph.PialSurface[regionpoints[newguessPoint]]
                    oldpos = dualgraph.PialSurface[roots[index]]
                    diff = GeneralFunctions.distancebetweenpoints(oldpos, newpos)
                    stopcrit += diff
                    roots[index] = regionpoints[newguessPoint]
                    # print(roots[index])
                print("Root displacement: %f" % stopcrit)
                iter += 1
                # print(rootpositions)

                resultcluster[regionindex].append([regionpoints, Labels, roots])
                if stopcrit == 0 or iter >= maxiter:
                    # resultcluster[regionindex] = (regionpoints, Labels, rootpositions)
                    break

                # if method == "euclidean":
                #     DistanceMat = dualgraph.CalculateDistanceMatRegion(method, roots, regionpoints)
                # else:
                # DistanceMat = dualgraph.CalculateDistanceMatRegion(method, roots, regionpoints)
                DistanceMat = CompleteDijkstraMap[numpy.ix_(roots, regionpoints)]
                # DistanceMat = PialDMat[roots, :]

        reorderedcouplingpoints = []
        CurrentID = 0
        for index, region in enumerate(resultcluster):
            majorid = self.regionids[index]
            regionroots = [node for node in self.CouplingPoints if node.Node.MajorVesselID == majorid]
            dualgraph.Points += list(region[-1][0])
            for iter in region:
                iter[1] = [i + CurrentID for i in iter[1]]  # overwrite current labels

                for index2, root in enumerate(regionroots):
                    rootpos = iter[2][index2]
                    root.RootPos.append(rootpos)

            dualgraph.Labels += region[-1][1]
            reorderedcouplingpoints += regionroots
            CurrentID += len(regionroots)

        self.CouplingPoints = reorderedcouplingpoints

        self.ClusteringResults = (dualgraph.Points, [])

        maxiter = max([len(res) for res in resultcluster])
        for i in range(0, maxiter):
            labels = []
            roots = []
            for region in resultcluster:
                if i < len(region):
                    data = region[i]
                else:
                    data = region[-1]
                labels += data[1]
                roots += data[2]

            self.ClusteringResults[1].append([copy.deepcopy(labels), copy.deepcopy(roots)])

        # dualgraph.Points = []
        # dualgraph.Labels = []
        # reorderedcouplingpoints = []
        # CurrentID = 0
        # for index, result in enumerate(resultcluster):
        #     majorid = self.regionids[index]
        #     # add final results to dualgraph
        #     dualgraph.Points += list(result[-1][0])
        #     labels = [i + CurrentID for i in result[-1][1]]
        #     dualgraph.Labels += labels
        #
        #     regionroots = [node for node in self.CouplingPoints if node.Node.MajorVesselID == majorid]
        #     for index2, root in enumerate(regionroots):
        #         rootpos = [iteration[index2] for iteration in result[-1][2]]
        #         root.RootPos = rootpos
        #     reorderedcouplingpoints += regionroots
        #     CurrentID += len(regionroots)

        # For each outlet, save the connected pial points
        for iter in range(0, len(self.CouplingPoints)):
            points = [dualgraph.Points[i] for i in range(0, len(dualgraph.Labels)) if dualgraph.Labels[i] == iter]
            self.CouplingPoints[iter].Pialpoints = points

        # Colour the mesh with the nearest coupling point
        # Colouring the mesh
        meshcolour = [-1 for node in dualgraph.PialSurface]
        for i in range(0, len(dualgraph.Points)):
            point = dualgraph.Points[i]
            meshcolour[point] = dualgraph.Labels[i]

        iter = 0
        while 1:
            nocolour = 0
            colourupdate = []
            for index, item in enumerate(meshcolour):
                if item == -1:
                    colouredlinkednodes = [i for i in dualgraph.Links[index] if
                                           meshcolour[i] >= 0 and dualgraph.MajorVesselID[i] == dualgraph.MajorVesselID[
                                               index]]
                    if len(colouredlinkednodes) > 0:
                        # calculate distance to each linked node and take the nearest node's colour
                        distancetocolourednodes = [
                            GeneralFunctions.distancebetweenpoints(dualgraph.PialSurface[index],
                                                                   dualgraph.PialSurface[p])
                            for p in colouredlinkednodes]

                        mindifferencebyindex = min(range(len(distancetocolourednodes)),
                                                   key=lambda k: distancetocolourednodes[k])
                        # update after the for loop
                        colourupdate.append((index, colouredlinkednodes[mindifferencebyindex]))
                    else:
                        nocolour += 1
            for element in colourupdate:
                meshcolour[element[0]] = meshcolour[element[1]]
            iter += 1
            if nocolour == 0 or iter == 1000:
                break
        dualgraph.NodeColour = meshcolour

    def ClusteringMetis(self, dualgraph, distancemat="Distancemat.npy", maxiter=10, useregion=False):
        print("Start clustering.")
        if useregion:
            self.FindNearestSurfaceNodeVEASL()
        else:
            # dualgraph.FindNearestSurfaceNode(self.CouplingPoints)
            self.FindNearestSurfaceNodeSides()

        randompoints = numpy.random.choice(range(0, len(self.DualGraph.PialSurface)), len(self.CouplingPoints),
                                           replace=False)
        for node, randomid in zip(self.CouplingPoints, randompoints):
            node.PialSurfacePointID = randomid

        # if GeneralFunctions.is_non_zero_file(distancemat):
        #     print("Loading Distance Matrix.")
        #     CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='r',
        #                                        shape=(len(dualgraph.PialSurface), len(dualgraph.PialSurface)))
        # else:
        #     CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='w+',
        #                                        shape=(len(dualgraph.MajorVesselID), len(dualgraph.MajorVesselID)))
        #     print("Allocated diskspace for the matrix.")
        #     # CompleteDijkstraMap = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False)
        #     for i in range(0, len(dualgraph.MajorVesselID)):
        #         CompleteDijkstraMap[:, i] = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False,
        #                                                                   indices=i)

        self.DualGraph.CalculateScipyGraph()
        nparts = len(self.CouplingPoints)

        uniquelocations = set([i.PialSurfacePointID for i in self.CouplingPoints])
        if len(uniquelocations) < len(self.CouplingPoints):
            print("Error in projection.")

        # calculate weights of the vertices
        areas = [i * 1e3 for i in self.DualGraph.Weights]
        locationweights = numpy.zeros(len(self.DualGraph.PialSurface))
        for index, cp in enumerate(self.CouplingPoints):
            locationweights[cp.PialSurfacePointID] = index + 1

        vertexweight = []
        for i in range(0, len(self.DualGraph.PialSurface)):
            vertexweight.append(int(areas[i]))
            vertexweight.append(int(locationweights[i]))
        vertexweight = numpy.array(vertexweight)

        # calculate target weights for the partitions
        RtCubed = sum([numpy.power(i.Node.Radius, 3) for i in self.CouplingPoints])
        weights = [numpy.power(i.Node.Radius, 3) / RtCubed for i in self.CouplingPoints]

        scaling = numpy.sum(numpy.arange(1, nparts + 1))
        rootweight = [i / scaling for i in range(1, nparts + 1)]

        totalweigths = numpy.zeros((nparts, 2))
        totalweigths[:, 0] = weights
        totalweigths[:, 1] = rootweight

        # network as metis wants it
        xadj = numpy.array(self.DualGraph.ScipyGraph.indptr.flatten(), dtype=numpy.int64)
        adjncy = self.DualGraph.ScipyGraph.indices.flatten()

        regionpoints = range(0, len(self.DualGraph.PialSurface))
        roots = [point.PialSurfacePointID for point in self.CouplingPoints]

        print("Optimizing clustering.")
        iter = 0
        resultcluster = []
        totalarea = sum(self.PrimalGraph.Areas)
        ubvec = numpy.array((1.01, 1.01))  # tolerance for each constraint
        # opts = metis.get_default_options()
        # opts[OPTION.NUMBERING] = 0
        # opts[OPTION.DBGLVL] = 1

        while 1:
            # partition the network using metis and the calculated weights and targets
            objval, npart = metis.part_graph_recursize(nparts, xadj, adjncy, ncon=2, vwgt=vertexweight,
                                                       tpwgts=totalweigths, ubvec=ubvec)  # , options=opts)
            for iter in range(0, len(self.CouplingPoints)):
                points = [i for i in range(0, len(npart)) if npart[i] == iter]
                area = sum([self.PrimalGraph.Areas[i] for i in points])
                print("%f==%f" % (area / totalarea, weights[iter]))
            stopcrit = 0

            # recompute the roots and redo the partitioning to see if that affects the partitioning
            print("Recalculating roots.")

            clusterelements = [[i for i in range(0, len(npart)) if npart[i] == element] for element
                               in range(0, nparts)]

            for index, cluster in enumerate(clusterelements):
                # distancematforcluster = numpy.empty([len(cluster), len(cluster)])
                # for num, point in enumerate(cluster):
                #     for num2, point2 in enumerate(cluster):
                #         # distancematforcluster[num, num2] = PialDMat[point, point2]
                #         distancematforcluster[num, num2] = CompleteDijkstraMap[
                #             regionpoints[point], regionpoints[point2]]
                # distances = numpy.amax(distancematforcluster, axis=0)
                # newguess = min(range(0, len(distances)), key=lambda k: distances[k])
                # newguessPoint = cluster[newguess]
                # new guess for root node is the Jardan Center of the cluster

                # this seems to have an error is the subgraph is not connected.
                # also related to the partitioning with zero element.
                if len(cluster) == 0:
                    newguessPoint = roots[index]
                else:
                    subgraph = self.DualGraph.Graph.subgraph(cluster)
                    newguessPoint = nx.center(subgraph)[0]

                # calculate the change in position of the root
                newpos = dualgraph.PialSurface[newguessPoint]
                oldpos = dualgraph.PialSurface[index]
                diff = GeneralFunctions.distancebetweenpoints(oldpos, newpos)
                stopcrit += diff
                roots[index] = newguessPoint

            print("Root displacement: %f" % stopcrit)
            iter += 1
            # resultcluster.append((Labels, copy.deepcopy(roots)))
            resultcluster.append((copy.deepcopy(nparts), copy.deepcopy(roots)))
            if stopcrit == 0 or iter >= maxiter:
                break
            # DistanceMat = CompleteDijkstraMap[numpy.ix_(roots, regionpoints)]

        ##########################################################################
        # final result from clustering
        self.ClusteringResults = (regionpoints, resultcluster)
        dualgraph.Points = regionpoints
        dualgraph.Labels = resultcluster[-1][0]

        for index, cp in enumerate(self.CouplingPoints):
            cp.RootPos = [i[1][index] for i in resultcluster]
            points = [dualgraph.Points[i] for i in range(0, len(dualgraph.Labels)) if dualgraph.Labels[i] == index]
            cp.Pialpoints = points

        # For each outlet, save the connected pial points
        for iter in range(0, len(self.CouplingPoints)):
            points = [dualgraph.Points[i] for i in range(0, len(dualgraph.Labels)) if dualgraph.Labels[i] == iter]
            self.CouplingPoints[iter].Pialpoints = points

        # Colour the mesh with the nearest coupling point
        # Colouring the mesh
        meshcolour = [-1 for node in dualgraph.PialSurface]
        for i in range(0, len(dualgraph.Points)):
            point = dualgraph.Points[i]
            meshcolour[point] = dualgraph.Labels[i]

        iter = 0
        while 1:
            nocolour = 0
            colourupdate = []
            for index, item in enumerate(meshcolour):
                if item == -1:
                    colouredlinkednodes = [i for i in dualgraph.Links[index] if meshcolour[i] >= 0]
                    if len(colouredlinkednodes) > 0:
                        # calculate distance to each linked node and take the nearest node's colour
                        distancetocolourednodes = [
                            GeneralFunctions.distancebetweenpoints(dualgraph.PialSurface[index],
                                                                   dualgraph.PialSurface[p])
                            for p in colouredlinkednodes]

                        mindifferencebyindex = min(range(len(distancetocolourednodes)),
                                                   key=lambda k: distancetocolourednodes[k])
                        # update after the for loop
                        colourupdate.append((index, colouredlinkednodes[mindifferencebyindex]))
                    else:
                        nocolour += 1
            for element in colourupdate:
                meshcolour[element[0]] = meshcolour[element[1]]
            iter += 1
            if nocolour == 0 or iter == 1000:
                break
        dualgraph.NodeColour = meshcolour

    def RemapMajorRegions(self):
        """
        Remap wrongly mapped regions to the nearst major region.
        :return: Updated mappings
        """
        print("Remapping the Major Regions.")
        regionsMajorID = []

        uniqueids = set(self.PrimalGraph.map)
        for id in uniqueids:
            points = [i for i, idcomp in enumerate(self.PrimalGraph.map) if idcomp == id]
            regions = []
            while len(points) > 0:
                region = [points[0]]
                candicates = [points[0]]
                points.remove(points[0])
                while len(candicates) > 0:
                    newcandicates = []
                    for index, item in enumerate(candicates):
                        for index2, item2 in enumerate(self.DualGraph.Links[item]):
                            if item2 in points:
                                region.append(item2)
                                newcandicates.append(item2)
                                points.remove(item2)
                    candicates = newcandicates
                regions.append(region)
            regionsMajorID.append(regions)

        LargestRegions = [sorted(regions, key=lambda x: len(x), reverse=True) for regions in regionsMajorID]
        # print(LargestRegions)
        # keptregions = [i[0] for i in LargestRegions]

        nodes = []
        wronglabel = [i[1:] for i in LargestRegions]
        for i, item in enumerate(wronglabel):
            for element in item:
                nodes.extend(element)

        for node in nodes:
            self.PrimalGraph.map[node] = -1

        iter = 0
        while 1:
            nocolour = 0
            colourupdate = []
            for _, item in enumerate(nodes):
                if self.PrimalGraph.map[item] == -1:
                    colouredlinkednodes = [i for i in self.DualGraph.Links[item] if self.PrimalGraph.map[i] >= 0]
                    if len(colouredlinkednodes) > 0:
                        # calculate distance to each linked node and take the nearest node's colour
                        distancetocolourednodes = [
                            GeneralFunctions.distancebetweenpoints(self.DualGraph.PialSurface[item],
                                                                   self.DualGraph.PialSurface[p])
                            for p in colouredlinkednodes]

                        mindifferencebyindex = min(range(len(distancetocolourednodes)),
                                                   key=lambda k: distancetocolourednodes[k])
                        # update after the for loop
                        colourupdate.append((item, colouredlinkednodes[mindifferencebyindex]))
                    else:
                        nocolour += 1
            for element in colourupdate:
                self.PrimalGraph.map[element[0]] = self.PrimalGraph.map[element[1]]
            iter += 1
            if nocolour == 0 or iter == 1000:
                break


class CouplingPoint:
    def __init__(self, node):
        self.Node = node
        self.PialSurfacePointID = None
        self.Pialpoints = []  # sampling used in the clustering
        self.NumberOfPialPoints = 0
        self.SurfaceNodes = []  # mapping after the clusting (triangles in primal graph)
        self.Area = 0
        self.Tree = None
        self.RootPos = []
