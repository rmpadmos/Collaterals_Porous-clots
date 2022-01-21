#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Contains the Vessel object class.
"""
import math

import numpy
from scipy.interpolate import interpolate

from Blood_Flow_1D import BloodFlowEquations, Node


class Vessel:
    """
    Vessel Object that stores nodes and varrious parameters.
    """

    def __init__(self):
        self.Nodes = []
        """Vessel nodes"""
        self.NumberOfNodes = 0
        """Number of nodes"""
        self.Length = None
        """Vessel length"""
        self.MeanRadius = None
        """Mean vessel radius"""
        self.MeanThickness = None
        """Mean vessel thickness"""
        self.GridSize = 2.5
        """Grid size"""
        self.Name = ""
        """Vessel name"""
        self.ID = None
        """Vessel ID"""
        self.MajorVesselID = -1
        """Major vessel ID"""
        self.YoungsModules = 1.6e6
        """Young's modules (pa)"""
        self.Resistance = None
        """Vessel resistance"""
        self.Compliance = None
        """Vessel compliance"""
        self.InterpolationFunctions = []
        """Interpolation functions of position and radius"""
        self.GenerationNumber = None
        """Generation number of the vessel"""
        self.Type = 0  # 0=not defined, 1=1D network, 2=Brava, 3=Tree, 4=pialnetwork, 5=collateral in pial network
        """Vessel type"""

    def GetDistalBifurcation(self):
        """
        Bifurcation node at the distal end of the vessel.

        Returns
        -------
        Node : node or None
            Bifurcation node if it exists
        """
        nodes = self.Nodes[-1].Connections
        for node in nodes:
            if node.Type == 1:
                return node
        return None

    def GetProximalBifurcation(self):
        """
        Bifurcation node at the proximal end of the vessel.

        Returns
        -------
        Node : node or None
            Bifurcation node if it exists
        """
        nodes = self.Nodes[0].Connections
        for node in nodes:
            if node.Type == 1:
                return node
        return None

    def GetEndNodes(self):
        """
        Return list of the proximal and distal bifurcation nodes.

        Returns
        -------
        nodelist : list
            List of proximal and distal bifurcation nodes if they exist.
        """
        proximal = self.GetProximalBifurcation()
        if proximal is None:
            proximal = self.Nodes[0]

        distal = self.GetDistalBifurcation()
        if distal is None:
            distal = self.Nodes[-1]

        return [proximal, distal]

    def UpdateNodeVesselID(self):
        """
        Update all nodes of the vessel to the vessel IDs of the vessel.
        """
        [node.SetVesselID(self.ID) for node in self.Nodes]
        [node.SetMajorVesselID(self.MajorVesselID) for node in self.Nodes]

    def UpdateNodeNumber(self):
        """
        Update the number of nodes that belong to this vessel.
        """
        self.NumberOfNodes = len(self.Nodes)

    def SetType(self, type):
        """
        Set the type of the vessel.
        0=default: not defined, 1=1D network, 2=Brava, 3=Tree, 4=pialnetwork, 5=collateral in pial network

        Parameters
        ----------
        type : int
            type to set self.Type
        """
        self.Type = type

    def SetID(self, idnumber):
        """
        Set the ID of the vessel.

        Parameters
        ----------
        idnumber : int
            Value to set self.ID
        """
        self.ID = idnumber

    def SetName(self, name):
        """
        Set the name of the vessel.

        Parameters
        ----------
        name : str
            Value to set self.Name
        """
        self.Name = name

    def UpdateNodeDirectionVectors(self):
        """
        Calculate the direction vectors for every node of the vessel.
        """
        [node.SetDirectionVector() for node in self.Nodes]

    def SetNodes(self, nodes):
        """
        Set nodes of this vessel.

        Parameters
        ----------
        nodes : list
            list of nodes
        """
        self.Nodes = nodes
        self.NumberOfNodes = len(nodes)

    def SetLength(self, length):
        """
        Set the length of the vessel.

        Parameters
        ----------
        length : float
            Value to set self.Length
        """
        self.Length = length

    def SetMeanRadius(self, rad):
        """
        Set the mean radius of the vessel and calculate the mean thickness using the thickness equation.

        Parameters
        ----------
        rad : float
            Radius value
        """
        self.MeanRadius = rad
        self.MeanThickness = BloodFlowEquations.thickness(rad)

    def SetMeanThickness(self, h):
        """
        Set the mean thickness of the vessel.

        Parameters
        ----------
        h : float
            Value to set self.MeanThickness
        """
        self.MeanThickness = h

    def SetGridSize(self, gridsize):
        """
        Set the distance between nodes of the vessel.

        Parameters
        ----------
        gridsize : float
            Distance value
        """
        self.GridSize = gridsize

    def SetMajorVesselID(self, majorid):
        """
        Set the major vessel id of the vessel and update the major vessel id of the nodes.

        Parameters
        ----------
        majorid : int
            Major vessel id value
        """
        self.MajorVesselID = majorid
        for node in self.Nodes:
            node.SetMajorVesselID(majorid)

    def ScaleRadius(self, r):
        """
        Scale the radius of the vessel and the nodes.
        Also calculates new vessel thicknesses.

        Parameters
        ----------
        r : float
            scaling factor
        """
        [node.SetRadius(node.Radius * r) for node in self.Nodes]
        self.MeanRadius *= r
        self.InterpolationFunctions[3].y *= r
        self.MeanThickness = BloodFlowEquations.thickness(self.MeanRadius)

    def UpdateVessel(self):
        """
        Update the vessel ids of the vessel nodes and the number of nodes.
        """
        self.UpdateNodeVesselID()
        self.UpdateNodeNumber()

    def CalculateMeanRadius(self):
        """
        Calculate the mean radius of the vessel from the radii of the nodes.

        Returns
        -------
        MeanRadius : float
            Mean radius of the vessel
        """
        self.MeanRadius = numpy.trapz([i.Radius for i in self.Nodes],
                                      [i.LengthAlongVessel for i in self.Nodes]) / self.Length
        if self.Length == 0:
            print(1)
        return self.MeanRadius

    def CalculateMeanThickness(self):
        """
        Calculate the mean thinkness of the vessel from the thinkness of the nodes.
        """
        self.MeanThickness = numpy.trapz([i.Thickness for i in self.Nodes],
                                         [i.LengthAlongVessel for i in self.Nodes]) / self.Length

    def VesselResistance(self, bloodvisc):
        """
        Calulate the resistance of the vessel based on a blunt profile.

        Parameters
        ----------
        bloodvisc : float
            blood viscocity value

        Returns
        -------
        R : float
            Resistance of the vessel
        """
        length = self.Length * 1e-3
        radius = self.MeanRadius * 1e-3
        # R = 8 * bloodvisc * length / (numpy.pi * numpy.power(radius, 4)) # laminar
        R = 22 * bloodvisc * length / (numpy.pi * numpy.power(radius, 4))  # blunt profile
        # func = [22 * bloodvisc * self.GridSize * 1e-3 / (numpy.pi * numpy.power(n.Radius* 1e-3, 4)) for n in self.Nodes]
        # R2 = sum(func)

        self.Resistance = R
        return R

    def VesselCompliance(self):
        """
        Calculate the compliance of the vessel

        Returns
        -------
        C1D : float
            Compliance of the vessel
        """
        C1D = 2 * numpy.power(numpy.pi * self.MeanRadius * 1e-3 * self.MeanRadius * 1e-3, 1.5) * self.Length * 1e-3 / (
                (4 / 3) * numpy.sqrt(numpy.pi) * self.YoungsModules * self.MeanThickness * 1e-3)
        self.Compliance = C1D
        return C1D

    def GenerateVessel(self, length, inletradius, outletradius, Elastic):
        """
        Generate vessel nodes. The vessel is modelled as an elastic tube with a taper.

        Parameters
        ----------
        length : float
            total length of the vessel
        inletradius : float
            proximal radius
        outletradius : float
            distal radius
        Elastic : float
            Young's modules
        """
        numbernodes = int(max(3, math.ceil(length / self.GridSize) + 1))
        nodelist = [Node.Node() for _ in range(0, numbernodes)]

        dx = length / (numbernodes - 1)
        if length == 0:
            self.Length = 0
            return
        drdl = (outletradius - inletradius) / length

        for i in range(0, numbernodes):
            nodelist[i].SetLengthAlongVessel(dx * i)
            nodelist[i].SetPosition([dx * i, 0, 0])
            nodelist[i].SetRadius(inletradius + drdl * dx * i)

        for i in range(0, numbernodes - 1):
            nodelist[i].AddConnection(nodelist[i + 1])
            nodelist[i + 1].AddConnection(nodelist[i])

        [node.SetYoungsModules(Elastic) for node in nodelist]
        self.Nodes = nodelist
        self.NumberOfNodes = numbernodes
        self.YoungsModules = Elastic
        self.GridSize = length / (numbernodes - 1)
        self.Length = length
        self.MeanRadius = (inletradius + outletradius) / 2
        self.MeanThickness = BloodFlowEquations.thickness(self.MeanRadius)

    def InterpolateVessel3Dto1D(self):
        """
        Interpolate the vessels in 3D and update each node position along the vessel (1D).
        """
        # print("Interpolating Vessel.")
        x = [i.Position[0] for i in self.Nodes]
        y = [i.Position[1] for i in self.Nodes]
        z = [i.Position[2] for i in self.Nodes]
        r = [i.Radius for i in self.Nodes]
        id = self.Nodes[1].MajorVesselID
        vesselid = self.Nodes[1].VesselID
        npts = len(x)
        s = numpy.zeros(npts, dtype=float)
        for j in range(1, npts):
            dx = x[j] - x[j - 1]
            dy = y[j] - y[j - 1]
            dz = z[j] - z[j - 1]
            vec = numpy.array([dx, dy, dz])
            s[j] = s[j - 1] + numpy.linalg.norm(vec)

        interpolation = 'linear'
        # Create new interpolation function for each dimension against the norm
        f1 = interpolate.interp1d(s, x, kind=interpolation)
        f2 = interpolate.interp1d(s, y, kind=interpolation)
        f3 = interpolate.interp1d(s, z, kind=interpolation)
        f4 = interpolate.interp1d(s, r, kind=interpolation)
        meanradius = numpy.trapz(r, s) / s[-1]

        gridnodes = max(3, math.ceil(s[-1] / self.GridSize) + 1)
        xvec = numpy.linspace(s[0], s[-1], gridnodes)
        radius = f4(xvec)

        position = [xvec[i] for i in range(0, gridnodes)]
        for r in radius:
            if r < 0:
                print("Radius below zero, interpolation is off.")
        position3d = [[f1(pos), f2(pos), f3(pos)] for pos in xvec]

        nodes = [Node.Node() for _ in range(0, gridnodes)]
        [nodes[i].SetLengthAlongVessel(position[i]) for i in range(0, gridnodes)]
        [nodes[i].SetRadius(meanradius) for i in range(0, gridnodes)]
        [nodes[i].SetYoungsModules(self.YoungsModules) for i in range(0, gridnodes)]
        [nodes[i].SetPosition(position3d[i]) for i in range(0, gridnodes)]
        [nodes[i].SetMajorVesselID(id) for i in range(0, gridnodes)]
        [nodes[i].SetVesselID(vesselid) for i in range(0, gridnodes)]
        self.InterpolationFunctions = [f1, f2, f3, f4]
        self.Nodes = nodes
        self.Length = s[-1]
        self.MeanRadius = meanradius
        self.NumberOfNodes = len(nodes)
        self.GridSize = self.Length / (self.NumberOfNodes - 1)
        self.MajorVesselID = id
        self.ID = vesselid
        for i in range(0, len(self.Nodes) - 1):
            self.Nodes[i].AddConnection(self.Nodes[i + 1])
            self.Nodes[i + 1].AddConnection(self.Nodes[i])
        self.MeanThickness = BloodFlowEquations.thickness(self.MeanRadius)

    def CreateInterpolationFunctions(self):
        """
        Create interpolation functions based on the 3-D coordinates of the current nodes.
        """
        x = [i.Position[0] for i in self.Nodes]
        y = [i.Position[1] for i in self.Nodes]
        z = [i.Position[2] for i in self.Nodes]
        r = [i.Radius for i in self.Nodes]

        npts = len(x)
        s = numpy.zeros(npts, dtype=float)
        for j in range(1, npts):
            dx = x[j] - x[j - 1]
            dy = y[j] - y[j - 1]
            dz = z[j] - z[j - 1]
            vec = numpy.array([dx, dy, dz])
            s[j] = s[j - 1] + numpy.linalg.norm(vec)

        self.Length = s[-1]

        interpolation = 'linear'
        # Create new interpolation function for each dimension against the norm
        f1 = interpolate.interp1d(s, x, kind=interpolation)
        f2 = interpolate.interp1d(s, y, kind=interpolation)
        f3 = interpolate.interp1d(s, z, kind=interpolation)
        f4 = interpolate.interp1d(s, r, kind=interpolation)
        self.InterpolationFunctions = [f1, f2, f3, f4]

    def UpdateInterpolationFunctions(self):
        """
        Update the interpolation functions of the vessel.
        """
        x = [i.Position[0] for i in self.Nodes]
        y = [i.Position[1] for i in self.Nodes]
        z = [i.Position[2] for i in self.Nodes]
        r = [i.Radius for i in self.Nodes]
        s = [i.LengthAlongVessel for i in self.Nodes]

        interpolation = 'linear'
        # Create new interpolation function for each dimension against the norm
        f1 = interpolate.interp1d(s, x, kind=interpolation)
        f2 = interpolate.interp1d(s, y, kind=interpolation)
        f3 = interpolate.interp1d(s, z, kind=interpolation)
        f4 = interpolate.interp1d(s, r, kind=interpolation)
        self.InterpolationFunctions = [f1, f2, f3, f4]

    def Interpolate3D(self, lengthalongvessel):
        """
        Using the interpolation functions of the vessel, interpolate the 3-D coordinates and radii along the center line.

        Parameters
        ----------
        lengthalongvessel : float
            length along the center line.

        Returns
        -------
        functions : list
            list of 3-D coordinates and radius
        """
        if len(self.InterpolationFunctions) == 0:
            self.CreateInterpolationFunctions()

        [f1, f2, f3, f4] = self.InterpolationFunctions
        return (
            numpy.vstack((f1(lengthalongvessel), f2(lengthalongvessel), f3(lengthalongvessel))), f4(lengthalongvessel))

    def CalculateMaxWaveSpeed(self, density):
        """
        Calculate the maximum wave velocity in the vessel.

        Parameters
        ----------
        density : float
            fluid density

        Returns
        -------
        maxwavespeed : float
            maximum wave velocity
        """
        # find the largest wavespeed in the vessel
        wavespeeds = []
        for node in self.Nodes:
            A = numpy.pi * node.Radius * 1e-3 * node.Radius * 1e-3
            beta = (4 / 3) * math.sqrt(math.pi) * node.YoungsModules * node.Thickness * 1e-3 / A
            c0 = abs(math.sqrt(beta / (2 * density))) * math.pow(A, 0.25)
            wavespeeds.append(c0)
        maxwavespeed = max(wavespeeds)
        return maxwavespeed  # m/s

    def UpdateResolution(self, numbernodes):
        """
        Update the nodes of the vessel by changing the number of nodes.
        Positions and radii are interpolated from the interpolation functions of the vessel.

        Parameters
        ----------
        numbernodes : int
            new number of nodes
        """
        # note that 3 is the minimum
        # if numbernodes < 3:
        #     print("Error: Minimum number of nodes allowed per vessel is 3.")
        #     return 0

        self.GridSize = self.Length / (numbernodes - 1)
        xvec = numpy.linspace(0, self.Length, numbernodes)
        xvec = [min(i, self.Length) for i in xvec]  # 12.00000001 is above the interpolation limit if length is 12.
        Interpolationvalues = self.Interpolate3D(xvec[1:-1])

        # remove old connections
        for i in range(0, len(self.Nodes) - 1):
            self.Nodes[i].RemoveConnection(self.Nodes[i + 1])
            self.Nodes[i + 1].RemoveConnection(self.Nodes[i])

        # create new nodes
        newnodes = [self.Nodes[0]]
        for i in range(1, numbernodes - 1):
            newnode = Node.Node()
            newnode.SetRadius(Interpolationvalues[1][i - 1])
            newnode.SetLengthAlongVessel(xvec[i])
            newnode.SetPosition(Interpolationvalues[0][:, i - 1])
            newnode.SetYoungsModules(self.YoungsModules)
            newnode.SetVesselID(self.ID)
            newnode.SetMajorVesselID(self.MajorVesselID)
            newnodes.append(newnode)
        newnodes.append(self.Nodes[-1])
        self.SetNodes(newnodes)

        # add connections
        for i in range(0, len(self.Nodes) - 1):
            self.Nodes[i].AddConnection(self.Nodes[i + 1])
            self.Nodes[i + 1].AddConnection(self.Nodes[i])

    def add_node_interp_length(self, lengthalongvessel):
        """
        Add node to the vessel at the chosen length position.

        Parameters
        ----------
        lengthalongvessel: Location where along the vessel to add a node

        Returns
        -------

        """
        if lengthalongvessel > self.Length:
            print("Error: interpolation length longer than vessel length.")
            return
        pos, r = self.Interpolate3D(lengthalongvessel)
        new_node = Node.Node()
        new_node.SetRadius(float(r))
        new_node.SetPosition(pos)
        new_node.SetMajorVesselID(self.MajorVesselID)
        new_node.SetLengthAlongVessel(lengthalongvessel)
        new_node.SetType(1)
        new_node.SetVesselID(self.Nodes[0].VesselID)

        self.Nodes.append(new_node)
        self.Nodes.sort(key=lambda n: n.LengthAlongVessel)

        for index, node in enumerate(self.Nodes[1:-1]):
            node.AddConnection(self.Nodes[index])
            node.AddConnection(self.Nodes[index+2])
            self.Nodes[index].AddConnection(node)
            self.Nodes[index+2].AddConnection(node)
