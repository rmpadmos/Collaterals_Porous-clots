#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Contains static functions for general use.
"""
import contextlib
import os
import sys
from operator import itemgetter

import numpy as np
import scipy
import scipy.sparse
import scipy.spatial
import vtk
from scipy.interpolate import interpolate
from vtk.util.numpy_support import vtk_to_numpy

from Blood_Flow_1D import Constants, Patient, Perfusion


def is_non_zero_file(fpath):
    """
    Return 1 if file exists and has data.
    :param fpath: path to file
    :return: boolean
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def TMatrix(scaling, rotation, translation):
    """
    4x4 rotation and translation matrix.
    Order of transformation is translation, X-Y-Z rotation.
    Scaling is isotropic
    :param scaling: scaling factor
    :param rotation: rotation angles: x,y,z values
    :param translation: translation values: x,y,z values
    :return: translation matrix
    """
    XCos = np.cos(np.radians(rotation[0]))
    YCos = np.cos(np.radians(rotation[1]))
    ZCos = np.cos(np.radians(rotation[2]))

    XSin = np.sin(np.radians(rotation[0]))
    YSin = np.sin(np.radians(rotation[1]))
    ZSin = np.sin(np.radians(rotation[2]))
    Translate = np.array(
        [[scaling, 0, 0, translation[0]], [0, scaling, 0, translation[1]], [0, 0, scaling, translation[2]],
         [0, 0, 0, 1]])
    RotateX = np.array([[1, 0, 0, 0], [0, XCos, -XSin, 0], [0, XSin, XCos, 0], [0, 0, 0, 1]])
    RotateY = np.array([[YCos, 0, YSin, 0], [0, 1, 0, 0], [-YSin, 0, YCos, 0], [0, 0, 0, 1]])
    RotateZ = np.array([[ZCos, -ZSin, 0, 0], [ZSin, ZCos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return np.dot(RotateZ, np.dot(RotateY, np.dot(RotateX, Translate)))


def TMatrixNonUniform(scaling, rotation, translation):
    """
    4x4 rotation and translation matrix.
    Order of transformation is translation, X-Y-Z rotation.
    Scaling is non-isotropic
    :param scaling: scaling factor: x,y,z values
    :param rotation: rotation angles: x,y,z values
    :param translation: translation values: x,y,z values
    :return: translation matrix
    """
    XCos = np.cos(np.radians(rotation[0]))
    YCos = np.cos(np.radians(rotation[1]))
    ZCos = np.cos(np.radians(rotation[2]))

    XSin = np.sin(np.radians(rotation[0]))
    YSin = np.sin(np.radians(rotation[1]))
    ZSin = np.sin(np.radians(rotation[2]))
    Translate = np.array(
        [[scaling[0], 0, 0, translation[0]], [0, scaling[1], 0, translation[1]], [0, 0, scaling[2], translation[2]],
         [0, 0, 0, 1]])
    RotateX = np.array([[1, 0, 0, 0], [0, XCos, -XSin, 0], [0, XSin, XCos, 0], [0, 0, 0, 1]])
    RotateY = np.array([[YCos, 0, YSin, 0], [0, 1, 0, 0], [-YSin, 0, YCos, 0], [0, 0, 0, 1]])
    RotateZ = np.array([[ZCos, -ZSin, 0, 0], [ZSin, ZCos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return np.dot(RotateZ, np.dot(RotateY, np.dot(RotateX, Translate)))


def TriangleToArea(nodes):
    """
    Take a list of node positions and return the area using Heron's formula.
    :param nodes: Set of points
    :return: area
    """
    a = distancebetweenpoints(nodes[0], nodes[1])
    b = distancebetweenpoints(nodes[1], nodes[2])
    c = distancebetweenpoints(nodes[0], nodes[2])
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def meanpos(nodes):
    """
    Take a list of node positions and return the centroid (mean position).
    :param nodes: list of node positions
    :return: mean x,y,z coordinates
    """
    x = np.mean([nodes[0][0], nodes[1][0], nodes[2][0]])
    y = np.mean([nodes[0][1], nodes[1][1], nodes[2][1]])
    z = np.mean([nodes[0][2], nodes[1][2], nodes[2][2]])
    return [x, y, z]


def distancebetweenpoints(p1, p2):
    """
    Calculate the euclidean distance between two points.
    :param p1: List of coordinates
    :param p2: List of coordinates
    :return:
    """
    dsq = sum([np.square(p1[i] - p2[i]) for i in range(0, len(p1))])
    return np.sqrt(dsq)


def Merge(patient, brava):
    """
    Merge a patient network with a brava network.
    :param patient: patient object containing the patient network
    :param brava:  patient object containing the brava network
    :return: Nothing
    """
    print("Merging patient network with another network.")
    matchpoints = ["R. ACA, A2", "R. MCA", "L. MCA", "L. ACA, A2", "R. PCA, P2", "L. PCA, P2"]
    bravaid = [2, 3, 4, 5, 6, 7]

    for vessel in patient.Topology.Vessels:
        vessel.SetType(1)
    for vessel in brava.Topology.Vessels:
        vessel.SetType(2)

    vesselnumbers = patient.Topology.Vessels[-1].ID  # last vessel id, renumber appended vessels from here

    # calculate the wk parameters without the merged network
    # make sure that we can find the relevant vessels
    with contextlib.redirect_stdout(None):
        patient.calculate_wk_parameters()
        patient.Topology.RedefineDirection()

    _, _, _, gens = patient.Topology.GetDownstreamVessels("Ascending Aorta")
    for index, gen in enumerate(gens):
        for vessel in gen:
            vessel.GenerationNumber = index

    for index, name in enumerate(matchpoints):
        vessels, _, _, _ = patient.Topology.GetDownstreamVessels(name)
        [vessel.SetMajorVesselID(bravaid[index]) for vessel in vessels]

    # find the relevent vessels and nodes in the patient network!
    # these are by definition the end nodes of the vessel
    nodedict = brava.Topology.MapNodesToVesselID()

    # first find all the major branches and their downsteam vessels.
    mainvesselscouplingbif = []
    endvesselsforwkupdate = []
    for loop in range(0, len(bravaid)):
        patientvesselname = bravaid[loop]  # MajorvesselID
        # extract vessels that match name
        vesselsmatch = [brava.Topology.Vessels[i] for i in range(0, len(brava.Topology.Vessels)) if
                        brava.Topology.Vessels[i].MajorVesselID == patientvesselname]

        # The target vessel is connected to other vessels that are not part of the set
        # check for a bifurcation is linked to another vessel type
        comparevesselbrava = []
        vesselnodes = nodedict[patientvesselname]

        for vessel in vesselsmatch:
            # find the bifurcations of the vessels
            bifurcations = [node for node in vessel.Nodes[0].Connections if node in brava.Topology.BifurcationNodes]
            [bifurcations.append(node) for node in vessel.Nodes[-1].Connections if
             node in brava.Topology.BifurcationNodes]

            for bif in bifurcations:
                targetnodes = bif.Connections
                for node in targetnodes:
                    if not (node in vesselnodes):
                        comparevesselbrava = vessel
                        BiflickedtoCow = bif

        firstnode = [node for node in comparevesselbrava.Nodes if node in BiflickedtoCow.Connections][0]
        mainvesselscouplingbif.append((comparevesselbrava, firstnode))
        firstnode.RemoveConnection(BiflickedtoCow)

    # for each major branch, find the downsteam vessels and rename them.
    brava.Topology.InletNodes = [(node[1], "") for node in mainvesselscouplingbif]
    with contextlib.redirect_stdout(None):
        brava.Topology.RedefineDirection()
    for loop in range(0, len(bravaid)):
        vessels, bifurcations, vesselnames, gens = brava.Topology.GetDownstreamVessels(mainvesselscouplingbif[loop][0])
        gens = brava.Topology.ReorderGens(gens)
        for index, gen in enumerate(gens):
            if index == 0:
                name = matchpoints[loop]
            else:
                name = matchpoints[loop] + ", gen:" + str(index)
            for index2, vessel in enumerate(gen):
                if index > 0:
                    vessel.Name = name + "," + str(index2)
                else:
                    vessel.Name = name

        patientvesselname = matchpoints[loop]
        vesselspatient, bifurcationspatient, vesselnamespatient, genspatient = patient.Topology.GetDownstreamVessels(
            patientvesselname)

        # find the equivalent vessels in the bravaset
        # if the patient anatomy contains more vessels beyond the first one.

        matchingvessels = [(genspatient[0][0], gens[0][0])]
        ends = matchingvessels
        while len(vesselspatient) != len(matchingvessels):
            newends = []
            for end in ends:
                patientvessel, bravavessel = end
                _, _, _, newgens = brava.Topology.GetDownstreamVessels(bravavessel)
                _, _, _, newgenspatient = patient.Topology.GetDownstreamVessels(patientvessel)
                newgenspatient = patient.Topology.ReorderGens(newgenspatient)
                newgens = brava.Topology.ReorderGens(newgens)

                if len(newgenspatient) > 1 and len(newgens) > 1:
                    # error if there are anastomoses since these are not present in the dataset
                    if len(newgenspatient[1]) > 2:
                        print("Warning, likely there are anastomoses.")
                        newgenspatient[1] = [vessel for vessel in newgenspatient[1] if
                                             vessel.GenerationNumber > newgenspatient[0][0].GenerationNumber]

                    for index, element in enumerate(newgenspatient[1]):
                        matchingvessels.append((element, newgens[1][index]))
                        newends.append((element, newgens[1][index]))
            ends = newends

        # note that the connectivity in this method might be different from the one above.
        # this seems to lead to some duplications if the ends of a major branch are of different generations
        # patientvessel, bravavessel = (genspatient[0][0], gens[0][0])
        # bravavessels, _, _, newgens = brava.Topology.GetDownstreamVessels(bravavessel)
        # patientvessels, _, _, newgenspatient = patient.Topology.GetDownstreamVessels(patientvessel)
        # matchingvessels = []
        # names = []
        # newgenspatient = patient.Topology.ReorderGens(newgenspatient)
        # newgens = brava.Topology.ReorderGens(newgens)
        #
        # for index, gen in enumerate(newgenspatient):
        #     for indexves, vessel in enumerate(gen):
        #         bravavessel = newgens[index][indexves]
        #         matchingvessels.append((vessel, bravavessel))
        #         names.append((vessel.Name, bravavessel.Name))

        finalends = [i for i in matchingvessels if len(i[0].Nodes[-1].Connections) == 1]
        # for each matching vessel, find the downstream vessel in the brava network
        donorbranches = [brava.Topology.GetDownstreamVessels(vessel[1]) for index, vessel in enumerate(finalends)]
        if len(finalends) > 1:
            for i in finalends:
                print(i[1].Name)
        # Interpolation of the vessels for visualisation
        for index, vessel in enumerate(matchingvessels):
            # Interpolate the connection between the brava and the patient network
            # We only want to change the position in 3d coordinates for visualisation.
            # The actual length used in the simulation should not change!
            if vessel[0].Name in matchpoints:
                # if the vessel is one of the major vessels, we just want to extend it to the next bifurcation.
                nodebrava = vessel[1].Nodes[-1]
                # nodepatient = vessel[0].Nodes[-1]
                x = [node.Position[0] for node in vessel[0].Nodes] + [
                    nodebrava.Position[0]]
                y = [node.Position[1] for node in vessel[0].Nodes] + [
                    nodebrava.Position[1]]
                z = [node.Position[2] for node in vessel[0].Nodes] + [
                    nodebrava.Position[2]]
                s = [node.LengthAlongVessel for node in vessel[0].Nodes] + [
                    vessel[0].Length + vessel[1].Length]
            else:
                # if the vessel is not one of the major vessel, it is an added patient vessel and we want to overwrite all the positions.
                # The original positions are garbage.
                x = [node.Position[0] for node in vessel[1].Nodes]
                y = [node.Position[1] for node in vessel[1].Nodes]
                z = [node.Position[2] for node in vessel[1].Nodes]
                s = [node.LengthAlongVessel for node in vessel[1].Nodes]

            interpolation = 'linear'
            f1 = interpolate.interp1d(s, x, kind=interpolation)
            f2 = interpolate.interp1d(s, y, kind=interpolation)
            f3 = interpolate.interp1d(s, z, kind=interpolation)

            for node in vessel[0].Nodes:
                fractionalongvessellength = node.LengthAlongVessel / vessel[0].Length
                fractionalongvessellength = min(1.0, fractionalongvessellength) * s[-1]
                newx = f1(fractionalongvessellength)
                newy = f2(fractionalongvessellength)
                newz = f3(fractionalongvessellength)
                node.SetPosition([newx, newy, newz])
            # Update the vessel interpolation functions
            vessel[0].UpdateInterpolationFunctions()

        # Extend the vessels of the patient with the vessels from the brava set
        for index, vessel in enumerate(finalends):  # only end vessels!
            # calculate the scaling factor
            # apply scaling to downsteam vessels
            # add vessels to patient
            # add bifurcations to patient
            # add the connections to the bifurcations
            # NOTE: the brava set does not have the ACoA so these scaling factors are off.
            # For this reason, the length ratio is not usable
            scalingsfactor = vessel[0].MeanRadius / vessel[1].MeanRadius
            vessels, bifurcations, names, gens = donorbranches[index]
            print(f"Scaling factor {scalingsfactor}")
            connectingvessel = vessel[0]
            endvesselsforwkupdate.append(connectingvessel)
            if len(gens) <= 1:
                # check if patient network continues
                upstream = patient.Topology.GetDownstreamVessels(connectingvessel)
                if len(upstream[0]) > 1:
                    print("Error, patient network continues after the bravaset.")
                # print("No vessels to attach, updating outlet position.")
                # vessel[0].Nodes[-1].SetPosition(vessel[1].Nodes[-1].Position)
                continue

            otherconnetingvessels = gens[1]
            donorvessels = [ves for ves in list(vessels) if ves is not vessel[1]]

            # todo remove scaling?
            [vessel.ScaleRadius(scalingsfactor) for vessel in donorvessels]

            bif = [bif for bif in bifurcations if bif in otherconnetingvessels[1].Nodes[0].Connections][0]
            bif.ResetConnections()
            bif.AddConnection(connectingvessel.Nodes[-1])
            bif.AddConnection(otherconnetingvessels[1].Nodes[0])
            bif.AddConnection(otherconnetingvessels[0].Nodes[0])
            connectingvessel.Nodes[-1].AddConnection(bif)
            otherconnetingvessels[1].Nodes[0].AddConnection(bif)
            otherconnetingvessels[0].Nodes[0].AddConnection(bif)
            # transfer to patient
            for vessel_ in donorvessels:
                vesselnumbers += 1
                vessel_.SetID(vesselnumbers)
                vessel_.SetType(2)
            [patient.Topology.Vessels.append(element) for element in donorvessels]
            [patient.Topology.BifurcationNodes.append(element) for element in bifurcations]
            [patient.Topology.Nodes.append(element) for sublist in donorvessels for element in sublist.Nodes]
            [patient.Topology.Nodes.append(element) for element in bifurcations]

    # add all the other things
    patient.Topology.FindOutletNodes()
    patient.Trees = brava.Trees
    patient.Perfusion = brava.Perfusion
    # The donor networks have outlets that do not get transferred. These need to be removed.
    # If the coupling point node does not have a number, remove it from the list.
    pointstoremove = [point for point in patient.Perfusion.CouplingPoints if not (point.Node in patient.Topology.Nodes)]
    [patient.Perfusion.RemoveCouplingPoint(point) for point in pointstoremove]
    # patient.Topology.RedefineDirection()
    patient.Topology.UpdateTopology()

    # [node.ResetWK() for node in patient.Topology.OutletNodes]
    # patient.calculate_wk_parameters()

    # update the wk_parameters at the ends of the added vessels
    # taking into account that the attached trees have resistance that we have to take into account
    visc = float(patient.ModelParameters["BLOOD_VISC"])
    density = float(patient.ModelParameters["Density"])
    print("Removing resistance added by the merging of networks.")
    for name in endvesselsforwkupdate:
        patient.Topology.DownStreamResistance(name, visc, density)


def TransformFile(file, transformmatrix):
    """
    Transform all points in a vtp or ply file.
    This is not the way that paraview does its transformations.
    :param file: file to transform
    :param transformmatrix: transformation matrix, see TMatrix()
    :return: Nothing
    """

    if file[-3:] == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif file[-3:] == "ply":
        reader = vtk.vtkPLYReader()
    else:
        print("Error: unreadable file.")
        return 1
    reader.SetFileName(file)
    reader.Update()
    data = reader.GetOutput()

    pos_vtk = reader.GetOutput().GetPoints().GetData()
    pos = vtk_to_numpy(pos_vtk)
    nodes = vtk.vtkPoints()
    for point in pos:
        vec = np.array([[point[0]], [point[1]], [point[2]], [1]])
        position = np.dot(transformmatrix, vec)
        nodes.InsertNextPoint(position[:-1])
    data.SetPoints(nodes)

    # export to new file
    writer = vtk.vtkXMLPolyDataWriter()
    file = "testingtrans.vtp"
    writer.SetFileName(file)
    writer.SetInputData(data)
    writer.Write()


class MSHfile:
    """
    Class to read and write msh mesh files.
    """

    def __init__(self):
        self.MeshFormat = []
        self.PhysicalNames = []
        self.Nodes = []
        self.Elements = []

    def Loadfile(self, file):
        """
        Load the msh file into the object.
        :param file: msh file
        :return: Nothing
        """
        print("Loading MSH: %s" % file)
        mesh_raw = [i.strip('\n') for i in open(file)]
        mesh = [i.split(' ') for i in mesh_raw]

        startelementsFormat = mesh_raw.index("$MeshFormat")
        endelementsFormat = mesh_raw.index("$EndMeshFormat")
        startelementsNames = mesh_raw.index("$PhysicalNames")
        endelementsNames = mesh_raw.index("$EndPhysicalNames")
        startelementsNodes = mesh_raw.index("$Nodes")
        endelementsNodes = mesh_raw.index("$EndNodes")
        startelements = mesh_raw.index("$Elements")
        endelements = mesh_raw.index("$EndElements")

        self.MeshFormat = [mesh_raw[i] for i in range(startelementsFormat + 1, endelementsFormat)]
        self.PhysicalNames = [[int(mesh[i][0]), int(mesh[i][1]), mesh[i][2]] for i in
                              range(startelementsNames + 2, endelementsNames)]
        self.Nodes = [[int(mesh[i][0]), float(mesh[i][1]), float(mesh[i][2]), float(mesh[i][3])] for i
                      in range(startelementsNodes + 2, endelementsNodes)]
        self.Elements = [[int(x) for x in mesh[i] if x] for i in range(startelements + 2, endelements)]

    def Writefile(self, file):
        """
        Write the object to file
        :param file: file name
        :return: Nothing
        """
        print("Writing MSH: %s" % file)
        with open(file, 'w') as f:
            f.write("$MeshFormat\n")
            for i in self.MeshFormat:
                f.write(i + "\n")
            f.write("$EndMeshFormat\n")

            f.write("$PhysicalNames\n")
            f.write(str(len(self.PhysicalNames)) + "\n")
            for i in self.PhysicalNames:
                line = ' '.join(str(x) for x in i)
                f.write(line + "\n")
            f.write("$EndPhysicalNames\n")

            f.write("$Nodes\n")
            f.write(str(len(self.Nodes)) + "\n")
            for i in self.Nodes:
                line = ' '.join(str(x) for x in i)
                f.write(line + "\n")
            f.write("$EndNodes\n")

            f.write("$Elements\n")
            f.write(str(len(self.Elements)) + "\n")
            for i in self.Elements:
                line = ' '.join(str(x) for x in i)
                f.write(line + "\n")
            f.write("$EndElements\n")

    def GetElements(self, ids):
        """
        Get the index and elements of the mesh at specific regions.
        :param ids: regions ids
        :return: elements, indexes
        """
        data = [[index, element] for index, element in enumerate(self.Elements) if
                int(element[4]) in ids or int(element[3]) in ids]
        indexes = [i[0] for i in data]
        elements = [i[1] for i in data]
        return elements, indexes

    def GetSurfaceCentroids(self, ids):
        """
        Get the centroids of the elements of the mesh
        :param ids: region ids of the elements
        :return: Positions, elements, indexes
        """
        elements, indexes = self.GetElements(ids)

        triangles = [[i[-1], i[-2], i[-3]] for i in elements]

        trianglespos = [[self.Nodes[triangle[0] - 1][1:],
                         self.Nodes[triangle[1] - 1][1:],
                         self.Nodes[triangle[2] - 1][1:]
                         ] for triangle in triangles]

        positions = [meanpos(i) for i in trianglespos]
        return positions, elements, indexes

    def AreaRegion(self, regionid):
        """
        Get the area of the elements of some region.
        :param regionid: region id of the elements
        :return: total area, number of triangles.
        """
        elements, indexes = self.GetElements([regionid])
        triangles = [[i[-1], i[-2], i[-3]] for i in elements]
        trianglespos = [[self.Nodes[triangle[0] - 1][1:],
                         self.Nodes[triangle[1] - 1][1:],
                         self.Nodes[triangle[2] - 1][1:]
                         ] for triangle in triangles]
        areas = [TriangleToArea(triangle) for triangle in trianglespos]
        totalarea = sum(areas)
        return totalarea, len(triangles)


def VesselMapping(mappingfile, patient):
    """
    # For each vessel/node in patient, map them to the same vessel in the mapping file.
    # For now, only 55 vessels are included.
    :param mappingfile: vtp file containing 3d vessel network
    :param patient: patient object
    :return: Nothing
    """
    mapping = Patient.Patient()
    mapping.LoadVTPFile(mappingfile)
    mapping.Topology.UpdateVesselAtlas()
    for vessel in patient.Topology.Vessels:
        if vessel.ID < 56:  # no mapping beyond these vessels
            mappedvessel = mapping.Topology.VesselAtlas[vessel.ID]
            vessel.MajorVesselID = mappedvessel.MajorVesselID
            for node in vessel.Nodes:
                fractionalongvessellength = node.LengthAlongVessel / vessel.Length
                if vessel.ID == 15:  # starts at the vessel end
                    positioninmap = (1.0 - min(1.0, fractionalongvessellength)) * mappedvessel.Length
                else:
                    positioninmap = min(1.0, fractionalongvessellength) * mappedvessel.Length
                newposx = mappedvessel.InterpolationFunctions[0](positioninmap)
                newposy = mappedvessel.InterpolationFunctions[1](positioninmap)
                newposz = mappedvessel.InterpolationFunctions[2](positioninmap)
                node.SetPosition([newposx, newposy, newposz])
            # update the interpolation functions
            vessel.UpdateInterpolationFunctions()

    # update the bifurcation positions as well
    for bif in patient.Topology.BifurcationNodes:
        bif.SetPosition(next(iter(bif.Connections)).Position)


def MapClusteringToMSH(file, file2, datafolder):
    """
    Mapping the clustered msh file to a nother msh file.
    :param file: msh file with clustering
    :param file2: target msh
    :param datafolder: folder containing files
    :return: Nothing
    """
    print("Mapping Clustering to a MSH file.")

    outputfilemsh = datafolder + 'clustered_mesh.msh'
    regionsIDs = [4, 21, 22, 23, 24, 25, 26, 30]  # extracted from the msh file

    Surface1 = Perfusion.Perfusion()
    Surface1.PrimalGraph.LoadSurface(file)
    centers = Surface1.PrimalGraph.GetTriangleCentroids()
    sys.setrecursionlimit(10000)
    KDTree = scipy.spatial.KDTree(centers)

    mshmesh = MSHfile()
    mshmesh.Loadfile(file2)
    positions, elements, indexes = mshmesh.GetSurfaceCentroids(regionsIDs)

    # euclidean distance between outlets and surface
    MinDistance, MinDistanceIndex = KDTree.query(positions, k=1)

    # MinDistance, MinDistanceIndex = KDTree.query([i for i in meshdata2.positions], k=1)
    regiondict = Constants.MajorIDdict

    ClusterDict = {}
    for index, trianglenumber in enumerate(MinDistanceIndex):


        # major region
        # mshmesh.Elements[indexes[index]][3] = regiondict[region]

        # cluster region
        cluster = Surface1.PrimalGraph.PolygonColour[trianglenumber]
        mshmesh.Elements[indexes[index]][3] = cluster + Constants.StartClusteringIndex
        mshmesh.Elements[indexes[index]][4] = cluster + Constants.StartClusteringIndex

        # region = Surface1.PrimalGraph.map[trianglenumber]
        # ClusterDict[cluster + Constants.StartClusteringIndex] = regiondict[region]  # store ids in dict

    # regionsName = ["Cerebellum","R. ACA, A2", "R. MCA", "L. MCA", "L. ACA, A2", "R. PCA, P2", "L. PCA, P2","Brainstem"]
    # regionsIDs = [4, 21, 22, 23, 24, 25, 26, 30]  # region 0, 1, 2, 3, 4, 5

    # list of cluster names
    clusterids = [i for i in range(0, len(set(Surface1.PrimalGraph.PolygonColour)))]
    uniqueclusters = len(set(clusterids))
    namesclusters = [[2, Constants.StartClusteringIndex + i, "\"Cluster_" + str(i) + '\"'] for i in
                     range(0, uniqueclusters)]
    mshmesh.PhysicalNames += namesclusters

    indexestoremove = []
    [indexestoremove.append(i) for i, line in enumerate(mshmesh.PhysicalNames) if line[1] in regionsIDs]
    newnames = [line for i, line in enumerate(mshmesh.PhysicalNames) if not (i in indexestoremove)]
    mshmesh.PhysicalNames = newnames

    mshmesh.Writefile(outputfilemsh)


def MapPointsToMSH(points, file2, datafolder):
    print("Mapping points to a MSH file.")

    outputfilemsh = datafolder + 'clustered_mesh.msh'
    regionsIDs = [4, 21, 22, 23, 24, 25, 26, 30]  # extracted from the msh file

    sys.setrecursionlimit(10000)
    KDTree = scipy.spatial.KDTree(points)

    mshmesh = MSHfile()
    mshmesh.Loadfile(file2)
    positions, elements, indexes = mshmesh.GetSurfaceCentroids(regionsIDs)

    # euclidean distance between outlets and surface
    MinDistance, MinDistanceIndex = KDTree.query(positions, k=1)

    for index, trianglenumber in enumerate(MinDistanceIndex):
        # cluster region
        cluster = trianglenumber
        mshmesh.Elements[indexes[index]][3] = cluster + Constants.StartClusteringIndex
        mshmesh.Elements[indexes[index]][4] = cluster + Constants.StartClusteringIndex

    # list of cluster names
    clusterids = [i for i in range(0, len(points))]
    uniqueclusters = len(set(clusterids))
    namesclusters = [[2, Constants.StartClusteringIndex + i, "\"Cluster_" + str(i) + '\"'] for i in
                     range(0, uniqueclusters)]
    mshmesh.PhysicalNames += namesclusters

    indexestoremove = []
    [indexestoremove.append(i) for i, line in enumerate(mshmesh.PhysicalNames) if line[1] in regionsIDs]
    newnames = [line for i, line in enumerate(mshmesh.PhysicalNames) if not (i in indexestoremove)]
    mshmesh.PhysicalNames = newnames

    mshmesh.Writefile(outputfilemsh)


def WriteFlowFilePerfusionModel(Clusterflowdatafile, clusteringfile, datafolder):
    """
    Output pressure and flow rate data to a boundary file for the perfusion network.
    :param Clusterflowdatafile: file containing blood flow variables
    :param clusteringfile: file containing clustering
    :param datafolder: outut folder
    :return: Nothing
    """
    clusterdata = [i.strip('\n') for i in open(clusteringfile)]
    clusterdata = [i.split(',') for i in clusterdata][1:]
    MajorvesselIDs = [int(i[5]) for i in clusterdata]
    flowdata = [i.strip('\n') for i in open(Clusterflowdatafile)]
    flowdata = [i.split(',') for i in flowdata][1:]

    with open(datafolder + 'boundary_condition_file.csv', 'w') as f:
        f.write("# region I,Q [ml/s],p [Pa],feeding artery ID,BC: p->0 or Q->1\n")
        for index, i in enumerate(flowdata):
            flow_rate = float(i[1])
            pressure = float(i[3])
            boundary_type = 1 if flow_rate < 1e-6 else 0
            f.write("%d,%.16g,%.16g,%d,%d\n" % (
                Constants.StartClusteringIndex + index, flow_rate, pressure,
                Constants.MajorIDdict[MajorvesselIDs[index]], boundary_type))


def slice_by_index(lst, indexes):
    """Slice list by positional indexes.

    Adapted from https://stackoverflow.com/a/9108109/304209.

    Args:
        lst: list to slice.
        indexes: iterable of 0-based indexes of the list positions to return.

    Returns:
        a new list containing elements of lst on positions specified by indexes.
    """
    if not lst or not indexes:
        return []
    slice_ = itemgetter(*indexes)(lst)
    if len(indexes) == 1:
        return [slice_]
    return list(slice_)


def unit_normal(a, b, c):
    """
    Compute unit normal vector of plane defined by points a, b, and c
    :param a: point 1
    :param b: point 2
    :param c: point 3
    :return: vector
    """
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
    return (x / magnitude, y / magnitude, z / magnitude)


def poly_area(poly):
    """
    Calculate area of a polygon
    :param poly: polygon coordinates
    :return: area
    """
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i + 1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


def WriteEdgesToVTP(positions, edges, file="network.vtp"):
    """
    Write a set of edges to a vtp file.
    :param positions: Positions of the nodes.
    :param edges: list of edges
    :param file: file name.
    :return: Nothing
    """
    vesselspolydata = vtk.vtkPolyData()

    nodes = vtk.vtkPoints()
    for i in positions:
        nodes.InsertNextPoint(i)
    vesselspolydata.SetPoints(nodes)

    lines = vtk.vtkCellArray()
    for index, element in enumerate(edges):
        line0 = vtk.vtkLine()
        line0.GetPointIds().SetNumberOfIds(2)
        line0.GetPointIds().SetId(0, element[0])
        line0.GetPointIds().SetId(1, element[1])
        lines.InsertNextCell(line0)

    vesselspolydata.SetLines(lines)

    print("Saving to %s" % file)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(vesselspolydata)
    writer.Write()


def AddArrayToFile(file, array, name, cell):
    """
    Add an array to an existing file.
    :param file: existing file
    :param array: array to add
    :param name: name of the array
    :param cell: boolean, add array as cell or point array. True=cell, False=Point
    :return: Nothing
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file)
    reader.Update()
    data = reader.GetOutput()

    resultdata = vtk.vtkFloatArray()
    resultdata.SetNumberOfComponents(1)
    resultdata.SetName(name)

    for i in array:
        resultdata.InsertNextValue(i)

    if cell:
        data.GetCellData().AddArray(resultdata)
    else:
        data.GetPointData().AddArray(resultdata)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(data)
    writer.Write()


def CreateBoundaryFile(file, folder):
    """
    Read a msh file and write a boundary map for the perfusion model.
    Filename = 'boundary_mapper.csv'
    :param file: msh file
    :param folder: output folder
    :return: Nothing
    """
    mshmesh = MSHfile()
    mshmesh.Loadfile(file)

    with open(folder + 'boundary_mapper.csv', 'w') as f:
        # for every element
        f.write("# feeding artery ID, region ID\n")
        for i in mshmesh.Elements:
            # 102210 2 2 26 93 33181 33182 126785
            if i[1] == 2:
                feeding = i[3]
                cluster = i[4]
                f.write("%d,%d\n" % (feeding, cluster))

    for i in mshmesh.Elements:
        if i[1] == 2:
            cluster = i[4]
            i[3] = cluster
    mshmesh.Writefile(folder + "mesh.msh")
