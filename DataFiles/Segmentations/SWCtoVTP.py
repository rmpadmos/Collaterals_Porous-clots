#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
This file converts segmentation centerlines from swc to vtp.
This assumes a particular order and meaning of the columns
Note that this does not work for the brava sets.
"""
import os

import vtk


class Node:
    def __init__(self):
        self.Number = None
        """Node Number"""
        self.Position = None
        """Node position"""
        self.Radius = None
        """Node radius"""
        self.Connections = set()
        """Set of connected Nodes"""
        self.ID = None
        """Node ID"""

    def add_connection(self, node):
        """
        Add node to the list of connections

        Parameters
        ----------
        node : node object
            node to add
        """
        self.Connections.add(node)

    def set_radius(self, radius):
        """
        set the radius variable

        Parameters
        ----------
        radius : float
            radius value

        Returns
        -------

        """
        self.Radius = float(radius)

    def set_number(self, number):
        """
        set the number variable

        Parameters
        ----------
        number : int
            number value

        Returns
        -------

        """
        self.Number = int(number)

    def set_position(self, pos):
        """
        Set the position of the node

        Parameters
        ----------
        pos : list
            list of node coordinates
        """
        self.Position = pos

    def set_ID(self, id):
        """
        Set the ID of the node

        Parameters
        ----------
        id : int
            id value
        """
        self.ID = id


def SWC_Processing(filename):
    """
    Convert the file from the swc to vtp format.

    Parameters
    ----------
    filename : str
        swv file

    Returns
    -------

    """
    print("Converting swc to vtp.")
    newname = filename[:-3] + "vtp"
    dataset = [text.split() for text in open(filename)]

    number = [int(line[0]) for line in dataset]
    colour = [int(line[1]) for line in dataset]
    positions = [[float(line[2]), float(line[3]), float(line[4])] for line in dataset]
    radius = [float(line[5]) for line in dataset]
    connection = [int(line[6]) for line in dataset]

    links = [[] for i in range(0, len(number))]
    for i in range(0, len(number)):
        if connection[i] > 0:
            nodenumber = number[i] - 1
            linkedto = connection[i] - 1
            links[nodenumber].append(linkedto)
            links[linkedto].append(nodenumber)

    nodes = [Node() for _ in range(0, len(number))]
    [nodes[index].set_number(index) for index, rad in enumerate(radius)]
    [nodes[index].set_radius(rad) for index, rad in enumerate(radius)]
    [nodes[index].set_position(pos) for index, pos in enumerate(positions)]
    [nodes[index].set_ID(id) for index, id in enumerate(colour)]

    for index, link in enumerate(links):
        for con in link:
            nodes[index].add_connection(nodes[con])
            nodes[con].add_connection(nodes[index])

    outlets = [node for node in nodes if len(node.Connections) == 1]
    Vessels = []
    ProcessedNodes = []
    for BifurcationNode in outlets:
        # check connected vessels
        newvessels = list(set(BifurcationNode.Connections) - set(ProcessedNodes))
        for node in newvessels:
            vessel = []
            vessel.append(BifurcationNode)
            ProcessedNodes.append(BifurcationNode)
            currentnode = node
            while 1:
                numberconnection = len(currentnode.Connections)
                if numberconnection == 2:
                    vessel.append(currentnode)
                    ProcessedNodes.append(currentnode)
                    currentnode = list(set(currentnode.Connections) - set(vessel))[0]
                elif numberconnection == 1:
                    vessel.append(currentnode)
                    ProcessedNodes.append(currentnode)
                    Vessels.append(vessel)
                    break
                else:
                    vessel.append(currentnode)
                    ProcessedNodes.append(currentnode)
                    Vessels.append(vessel)
                    break

    nodesvtk = vtk.vtkPoints()  # pos
    vessels = vtk.vtkCellArray()  # lines
    radius = vtk.vtkFloatArray()  # radius
    nodeid = vtk.vtkIntArray()  # id

    radius.SetNumberOfComponents(1)
    radius.SetName("Radius")
    nodeid.SetNumberOfComponents(1)
    nodeid.SetName("ID")

    # Add radius and position to data array
    for i in nodes:
        nodesvtk.InsertNextPoint(i.Position)
        radius.InsertNextValue(i.Radius)
        nodeid.InsertNextValue(i.ID)

    # Add vessels to cell array
    for vessel in Vessels:
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(len(vessel))
        for i in range(0, len(vessel)):
            line.GetPointIds().SetId(i, vessel[i].Number)
        vessels.InsertNextCell(line)

    # Create a polydata to store everything in
    VesselsPolyData = vtk.vtkPolyData()

    # Add the nodes to the polydata
    VesselsPolyData.SetPoints(nodesvtk)

    # Add the vessels to the polydata
    VesselsPolyData.SetLines(vessels)

    # Assign radii and id to the nodes
    VesselsPolyData.GetPointData().SetScalars(radius)
    VesselsPolyData.GetPointData().AddArray(nodeid)

    # Write everything to a vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(newname)
    writer.SetInputData(VesselsPolyData)
    writer.Write()


def Dataset_SWC_To_VTP(path='/drive/in-silico-trial/software/1d-blood-flow/scripts/'):
    """
    Convert all swc files in a folder to .vtp files.

    Parameters
    ----------
    path : str
        path of the folder containing swc files.
    """
    print("Converting files from swc to vtp.")
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for name in filenames:
        if name[-3:] == "swc":
            SWC_Processing(path + name)


if __name__ == '__main__':
    Dataset_SWC_To_VTP("INSIST_0001/")
