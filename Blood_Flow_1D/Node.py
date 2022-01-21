#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Contains the Node object class.
"""
import math

import numpy

from Blood_Flow_1D import BloodFlowEquations


class Node:
    """
    Node Object
    """

    def __init__(self):
        self.Number = None
        """Node Number"""
        self.Position = [0, 0, 0]
        """3-D node position (mm)"""
        self.Radius = 0
        """Node radius (mm)"""
        self.Connections = set()
        """Nodes connected to this node"""
        self.R1 = None
        """Resistance 1"""
        self.R2 = None
        """Resistance 2"""
        self.C = None
        """Capacitance"""
        self.YoungsModules = 1.6e6
        """Young's modules (pa)"""
        self.Type = 0
        """node type:0=node, 1=bifurcation, 2=outlet/inlet"""
        self.VesselID = -1
        """Vessel ID of parent vessel"""
        self.MajorVesselID = -1
        """ID of major cerebral vessel"""
        self.LengthAlongVessel = 0
        """1-D node position (mm) """
        self.Thickness = None
        """Vessel wall thickness (mm)"""
        self.DirectionVector = [0, 0, 0]
        """Positive direction vector"""
        self.Pressure = 0
        """Pressure (pa)"""
        self.Velocity = 0
        """Velocity (m/s)"""
        self.FlowRate = 0
        """Volume flow rate (mL/s)"""
        self.PressureRadiusEquation = None
        """Function that describes elastic behaviour"""
        self.RefRadius = 0
        """Initial radius"""
        self.RefPressure = 0
        """Initial Pressure"""

    def flux_function(self, Area):
        """
        Flux function used in some calculations in the 1-D pulsatile blood flow model.

        Parameters
        ----------
        Area : float
            Area at the node

        Returns
        -------
        Flux : float
            flux value

        """
        # chrt_back = -4 * numpy.power(self.Lumen0, 0.25) * numpy.sqrt(self.Beta / 2.0 / self.density)
        return Area * self.charback + 4 * numpy.power(Area, 1.25) * numpy.sqrt(
            self.Beta / 2.0 / self.Density) - self.CurrentFlowrate

    def chrt_function(self, Area):
        """
        Calculate the difference between forward and backward characteristic.
        Input parameter is a guess for the new area.

        Parameters
        ----------
        Area : float
            Area at the node.

        Returns
        -------
        Result : float
            Difference between forward and backward characteristic

        """
        chrt_frw_right = self.CurrentVolumeFlowRate / Area + 4 * numpy.power(Area, 0.25) * numpy.sqrt(
            self.Beta / 2.0 / self.Density)
        return self.chrt_frw_left - chrt_frw_right

    def SetPressureAreaEquation(self):
        """
        Equation that described the pressure-area relationship at the node.

        Sets self.radius
        """
        function = lambda pressure: (pressure - self.RefPressure) * self.RefRadius * self.RefRadius * 3 / (
                4 * self.YoungsModules * self.Thickness) + self.RefRadius
        self.PressureRadiusEquation = function

    def SetPressureAreaEquation_rigid(self):
        """
               Equation that described the pressure-area relationship at the node.
               Rigid walls, no change in radius)

               Sets self.radius
               """
        function = lambda pressure: self.RefRadius
        self.PressureRadiusEquation = function

    def UpdateRadius(self):
        """
        Update the radius based on the pressure-area relationship.

        Sets self.radius
        """
        self.Radius = self.PressureRadiusEquation(self.Pressure)

    def SetNumber(self, number):
        """
        Set self.Number

        Parameters
        ----------
        number: int
            New number variable value.
        """
        self.Number = number

    def SetYoungsModules(self, modules):
        """
        Set self.YoungsModules


        Parameters
        ----------
        modules : float
            Young's modules
        """
        self.YoungsModules = modules

    def SetLengthAlongVessel(self, l):
        """
        Set self.LengthAlongVessel

        Parameters
        ----------
        l : float
            1-D position
        """
        self.LengthAlongVessel = l

    def SetRadius(self, r):
        """
        Set self.Radius and update the vessel wall thickness based on the radius.

        Parameters
        ----------
        r : float
            New radius variable
        """
        self.Radius = r
        self.CalculateThickness()

    def SetPosition(self, pos):
        """
        Set self.Position

        Parameters
        ----------
        pos : iterable object
        """
        self.Position = []
        for p in pos:
            self.Position.append(float(p))

    def SetVesselID(self, _id):
        """
        Set self.VesselID

        Parameters
        ----------
        _id : int
            Vessel ID
        """
        self.VesselID = _id

    def SetType(self, type):
        """
        Set self.Type

        0=node, 1=bifurcation, 2=outlet/inlet node.

        Parameters
        ----------
        type : int
            type
        """
        self.Type = type

    def CalculateThickness(self):
        """
        Calculate the vessel wall thinkness based on the radius.

        Returns
        -------
        Thickness : float
            Wall thickess (mm)
        """
        self.Thickness = BloodFlowEquations.thickness(self.Radius)
        return self.Thickness

    def SetMajorVesselID(self, _id):
        """
        Set self.MajorVesselID

        Parameters
        ----------
        _id : int
            New major vessel ID
        """
        self.MajorVesselID = _id

    def GetConnectedVesselIDs(self):
        """
        Get the vessel ids of the nodes connected to the node.

        Returns
        -------
        Connections : list
            List of vessel IDs of connected nodes

        """
        listIDS = []
        for node in self.Connections:
            listIDS.append(node.VesselID)
        return listIDS

    def GetConnectedMajorVesselIDs(self):
        """
        Get the major vessel ids of the nodes connected to the node.

        Returns
        -------
        Connections : list
            List of major vessel IDs of connected nodes

        """
        listIDS = []
        for node in self.Connections:
            listIDS.append(node.MajorVesselID)
        return listIDS

    def AddConnection(self, node):
        """
        Add a node to the set of connected nodes.

        Parameters
        ----------
        node : node
            node to add
        """
        self.Connections.add(node)

    def RemoveConnection(self, node):
        """
        Remove a node to the set of connected nodes.

        Parameters
        ----------
        node : node
            node to remove
        """
        self.Connections.remove(node)

    def ResetConnections(self):
        """
        Clear the list of connected nodes.
        """
        self.Connections.clear()

    def SetWK(self, r1, r2, c):
        """
        Set the windkessel parameters: self.R1, self.R2 and self.C.

        Parameters
        ----------
        r1 : float
            Resistance 1
        r2 : float
            Resistance 2
        c :float
            Capacitance
        """
        self.R1 = r1
        self.R2 = r2
        self.C = c

    def ResetWK(self):
        """
        Reset the windkessel parameters.
        """
        self.R1 = None
        self.R2 = None
        self.C = None

    def SetNodeFromTopLine(self, line):
        """
        Update node properties.

        Parameters
        ----------
        line : list of str
            Line containing node properties
        """
        self.SetNumber(int(line[0]))
        self.SetLengthAlongVessel(float(line[1][2:]))
        self.SetRadius(float(line[2][2:]))
        self.SetYoungsModules(float(line[3][2:]) * 1e6)
        self.SetType(int(line[4][2:]))

    def SetDirectionVector(self):
        """
        Calculate the direction vector of a node.
        Only valid for internal nodes.

        Set self.DirectionVector
        """
        if len(self.Connections) > 2:
            print("Error: Direction Vector not defined for bifurcations.")
            return 1
        if len(self.Connections) == 0:
            print("Error: Node has no connections.")
            return 1

        linkednode = max(list(self.Connections), key=lambda x: x.LengthAlongVessel)
        nbpos = linkednode.Position
        pos = self.Position
        direction = [nbpos[0] - pos[0], nbpos[1] - pos[1], nbpos[2] - pos[2]]
        length = math.sqrt((direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]))
        directionvector = [direction[0] / length, direction[1] / length, direction[2] / length]
        self.DirectionVector = directionvector
