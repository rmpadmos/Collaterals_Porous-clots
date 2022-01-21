#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Script to remesh a surface file with a set number of uniform triangles.

"""
import sys

import pyacvd
import pyvista
import scipy.spatial
import vtk

from Blood_Flow_1D import GeneralFunctions, Patient, Constants


def remesh(surfacefile, numbertriangles=20000, output="remeshed.vtp"):
    """
    Remesh a surface mesh using using voronoi clustering. Source and module at https://pypi.org/project/pyacvd/

    Parameters
    ----------
    surfacefile : str
        Surfacefile to be remeshed to a uniform triangulation.
    numbertriangles : int
        Number of triangles that the surface will have after the remeshing. Default:40000
    output : str
        output file name
    """
    print("Remeshing surface.")
    if surfacefile[-3:] == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif surfacefile[-3:] == "ply":
        reader = vtk.vtkPLYReader()
    else:
        print("Input is not a ply or vtp file.")
        return
    reader.SetFileName(surfacefile)
    reader.Update()

    p = reader.GetOutput()
    surf = pyvista.PolyData(p)
    clus = pyacvd.Clustering(surf)

    clus.subdivide(5)
    clus.cluster(numbertriangles, maxiter=1000, iso_try=100)
    new_mesh = clus.create_mesh()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output)
    writer.SetInputData(new_mesh)
    writer.Write()


def MapMeshtoMSH(filevtp, filemsh, output="PialSurface.vtp"):
    """
    Apply the mapping on one surface to another surface.

    Parameters
    ----------
    filevtp : str
        The remeshed surface file, see the remesh function.
    filemsh : str
        The file containing the VEASL mapping.
    output : str
        Filename of the resulting file.
    """
    print("Mapping msh to vtp.")
    regionsIDs = [4, 21, 22, 23, 24, 25, 26, 30]
    patient = Patient.Patient()
    patient.Perfusion.LoadPrimalGraph(filevtp)
    centroids = patient.Perfusion.PrimalGraph.GetTriangleCentroids()

    msh = GeneralFunctions.MSHfile()
    msh.Loadfile(filemsh)
    positions, elements, indexes = msh.GetSurfaceCentroids(regionsIDs)

    sys.setrecursionlimit(10000)
    KDTree = scipy.spatial.KDTree(positions)
    MinDistance, MinDistanceIndex = KDTree.query(centroids, k=1)

    regiondict = Constants.MajorIDdict_inv
    regionsIDs = [regiondict[elements[trianglenumber][3]] for index, trianglenumber in enumerate(MinDistanceIndex)]
    patient.Perfusion.PrimalGraph.PolygonColour = regionsIDs
    patient.Perfusion.PrimalGraph.File = output
    patient.Perfusion.PrimalGraph.GraphToVTP("")


if __name__ == '__main__':
    # remesh the mesh to a uniform triangulation.
    PAestimate = 113913
    remesh("boundary_4&21&22&23&24&25&26&30.ply", numbertriangles=PAestimate)
    # apply the same mapping to the remeshed file
    MapMeshtoMSH("remeshed.vtp", "labelled_vol_mesh.msh")

    meshpatient = Patient.Patient()
    meshpatient.Perfusion.PrimalGraph.LoadSurface("PialSurface.vtp")
    meshpatient.Perfusion.SetDualGraph(method="vertices")
    meshpatient.Perfusion.DualGraph.GraphToVTP()
    meshpatient.Perfusion.DualGraph.GraphEdgesToVTP("DualGraphEdges.vtp")
    distancemat = "Distancemat.npy"
    dualgraph = meshpatient.Perfusion.DualGraph

    #
    #
    # CompleteDijkstraMap = numpy.memmap(distancemat, dtype='float32', mode='w+',
    #                                    shape=(len(dualgraph.PialSurface), len(dualgraph.PialSurface)))
    # print("Allocated diskspace for the matrix.")
    # # CompleteDijkstraMap = scipy.sparse.csgraph.dijkstra(dualgraph.ScipyGraph, directed=False)
    # dualgraph.CalculateScipyGraph()
