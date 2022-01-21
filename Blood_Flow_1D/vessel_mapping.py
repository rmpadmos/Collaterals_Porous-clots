#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Script to output the perfusion territories of all cerebral vessels and the mapping of the vessels within the network.
Generated files are located in bf_sim: "vesselmapping.vtp" and "vesselmapping_network.vtp".
Usage:
  vessel_mapping.py <patient_folder>
"""
import os

import vtk

from Blood_Flow_1D import Patient, docopt


def mapping_function(patient):
    Major_regions_vessel = [(4, "L. MCA"),
                            (3, "R. MCA"),
                            (5, "L. ACA, A2"),
                            (2, "R. ACA, A2"),
                            (7, "L. PCA, P2"),
                            (6, "R. PCA, P2")]

    # # brain stem
    brainstem = (9, ("Pontine I", "Pontine II", "Pontine III", "Pontine IV", "Pontine V", "Pontine VI", "Pontine VII",
                     "Pontine VIII", "Pontine IX", "Pontine X", "Pontine XI", "Pontine XII"))

    # cerebellum
    cerebellum = (8, ("R. PICA", "L. PICA", "R. AICA", "L. AICA", "R. SCA", "L. SCA"))

    # for each vessel, we want to compute the perfusion territory
    # we need the mapping from one vessel to another
    # 1 get a list of vessels that we want to compute the territory for (all of them?)

    patient.Topology.UpdateVesselAtlas()
    vessel_list = set()
    [vessel_list.add(patient.Topology.VesselAtlas[vessel_name]) for _, vessel_name in Major_regions_vessel]
    [vessel_list.add(patient.Topology.VesselAtlas[vessel_name]) for vessel_name in brainstem[1]]
    [vessel_list.add(patient.Topology.VesselAtlas[vessel_name]) for vessel_name in cerebellum[1]]

    # add perfusion territory to vessels from outlet nodes!
    patient.Topology.NodeVesselDict()
    for cp in patient.Perfusion.CouplingPoints:
        outlet_vessel = patient.Topology.NodeDict[cp.Node]
        outlet_vessel.mapping = cp.SurfaceNodes

    # recursive algorithm
    def get_mapping(vessel):
        if hasattr(vessel, 'mapping'):
            return vessel.mapping
        else:
            _, _, _, gens = patient.Topology.GetDownstreamVessels(vessel)
            mapping = []
            for i in gens[1]:
                mapping += get_mapping(i)
            vessel.mapping = set(mapping)
            return vessel.mapping

    for start_vessel in vessel_list:
        start_vessel.mapping = get_mapping(start_vessel)

    # map to surface
    poly_colour = [None for _ in patient.Perfusion.PrimalGraph.map]
    patient.Topology.NodeVesselDict()
    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
        vessel = patient.Topology.NodeDict[cp.Node]
        cp.Node.SurfaceNodes = vessel.mapping
        for node in vessel.mapping:
            poly_colour[node] = index
    patient.Perfusion.PrimalGraph.PolygonColour = poly_colour

    ## output region per vessel
    filename = patient.Folders.ModellingFolder + "vesselmapping.vtp"
    nodes = vtk.vtkPoints()
    for index, item in enumerate(patient.Perfusion.PrimalGraph.PialSurface):
        nodes.InsertNextPoint(item)

    MajorArteries = vtk.vtkIntArray()
    MajorArteries.SetNumberOfComponents(1)
    MajorArteries.SetName("Major Cerebral Artery")
    for i in range(0, len(patient.Perfusion.PrimalGraph.PolygonColour)):
        regionid = patient.Perfusion.PrimalGraph.PolygonColour[i]
        couplingpoint = patient.Perfusion.CouplingPoints[regionid]
        MajorArteries.InsertNextValue(couplingpoint.Node.MajorVesselID)

    VesselsPolyData = vtk.vtkPolyData()
    VesselsPolyData.SetPoints(nodes)
    VesselsPolyData.SetPolys(patient.Perfusion.PrimalGraph.Polygons)
    VesselsPolyData.GetCellData().AddArray(MajorArteries)

    for vessel in patient.Topology.Vessels:
        if hasattr(vessel, 'mapping'):
            Colour = vtk.vtkFloatArray()
            Colour.SetNumberOfComponents(1)
            Colour.SetName(vessel.Name)
            for index, _ in enumerate(patient.Perfusion.PrimalGraph.PolygonColour):
                if index in vessel.mapping:
                    Colour.InsertNextValue(1)
                else:
                    Colour.InsertNextValue(0)
            VesselsPolyData.GetCellData().AddArray(Colour)

    writer = vtk.vtkXMLPolyDataWriter()
    print("Writing mapping to file: %s" % filename)
    writer.SetFileName(filename)
    writer.SetInputData(VesselsPolyData)
    writer.Write()

    #### vessel network
    filename = patient.Folders.ModellingFolder + "vesselmapping_network.vtp"
    nodes = vtk.vtkPoints()
    vessels = vtk.vtkCellArray()
    for i in patient.Topology.Nodes:
        nodes.InsertNextPoint(i.Position)
    for vessel in patient.Topology.Vessels:
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(len(vessel.Nodes))
        for i in range(0, len(vessel.Nodes)):
            line.GetPointIds().SetId(i, vessel.Nodes[i].Number)
        vessels.InsertNextCell(line)

    VesselsPolyData = vtk.vtkPolyData()
    VesselsPolyData.SetPoints(nodes)
    VesselsPolyData.SetLines(vessels)

    for index, vessel in enumerate(patient.Topology.Vessels):
        vessel_labelling = vtk.vtkIntArray()
        [vessel_labelling.InsertNextValue(1) if index == index2 else vessel_labelling.InsertNextValue(0)
         for index2, i in enumerate(patient.Topology.Vessels)]
        vessel_labelling.SetName(vessel.Name)
        VesselsPolyData.GetCellData().AddArray(vessel_labelling)

    writer = vtk.vtkXMLPolyDataWriter()
    print("Writing mapping to file: %s" % filename)
    writer.SetFileName(filename)
    writer.SetInputData(VesselsPolyData)
    writer.Write()


def blood_flow_script(patient_folder):
    patient = Patient.Patient(patient_folder)
    patient.LoadBFSimFiles()
    patient.LoadClusteringMapping(patient.Folders.ModellingFolder + "Clusters.csv")
    patient.LoadSurfaceMapping()
    patient.LoadPositions()

    surfacefile = patient.Folders.ModellingFolder + "PialSurface.vtp"
    patient.Perfusion.PrimalGraph.LoadSurface(surfacefile)
    return patient


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    patient_folder = arguments["<patient_folder>"]
    # patient_folder = "../Generated_Patients/patient_0/"
    patient = blood_flow_script(patient_folder)
    mapping_function(patient)

    # entire folder
    # main_folder = "/home/raymond/Desktop/2021-09-21_Brava_more-triangles/"
    # folder = [main_folder + f + "/" for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    #
    # for sim in folder:
    #     mapping_function(blood_flow_script(sim))
