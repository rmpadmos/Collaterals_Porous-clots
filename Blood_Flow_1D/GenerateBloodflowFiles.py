#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
1D Blood Flow Simulator
Script to run a blood flow simulation.
Input argument is the patient folder with a folder for input files and a folder for modelling files
The input folder contains patient data and the modelling file folder contains files for the models
such as the parameters and the surface mesh.
Usage:
  BloodFlow.py <patient_folder>
  BloodFlow.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
import shutil
from datetime import datetime
import scipy.spatial
import os
from Blood_Flow_1D import GeneralFunctions, Patient as PatientClass, Remesh, transcript, docopt, Collaterals


def generatebloodflowfiles(patient_folder):
    """
    Generate all required files for subsequent simulations and models.

    Parameters
    ----------
    patient_folder : str
        location of patient input files

    Returns
    -------
    patient : patient object
        Patient object containing all parameters and data.
    """
    start_time = datetime.now()

    # Load input files
    Patient = PatientClass.Patient(patient_folder)
    # Patient.ResetModellingFolder()
    Patient.LoadModelParameters("Model_parameters.txt")
    newnetwork = Patient.ModelParameters["NewPatientFiles"]
    # newnetwork = True

    if not GeneralFunctions.is_non_zero_file(Patient.Folders.InputFolder + "1-D_Anatomy_Patient.txt"):
        if newnetwork == "True" and Patient.ModelParameters["UsingPatientSegmentation"] == "True":
            # new network
            Patient.SelectPatient()  # randomly patient anatomy
            Patient.LoadSegmentedVessels()  # This creates a new file, this is temporary until there is a module 2.
        else:
            print("Using anatomy files as defined in modelling parameters.")
            # else use previously used anatomy from modelling parameters
            CoWConfiguration = Patient.ModelParameters["Circle_of_Willis"]
            # overwrite 1-D_Anatomy.txt
            CoWFolder = Patient.Folders.DataFilesFolder + "/CoW_Configurations/"
            anatomyfile = CoWFolder + CoWConfiguration + ".txt"
            shutil.copy(anatomyfile, Patient.Folders.InputFolder)

            if Patient.ModelParameters["UsingPatientSegmentation"] == "True":
                # copy patient segmentation
                selectedfolder = Patient.ModelParameters["Patient_Segmentation"]
                # overwrite segmentation files
                segmentationfolder = Patient.Folders.DataFilesFolder + "/Segmentations/"
                segmentationfile = segmentationfolder + selectedfolder + "/Feature_Vessel.csv"
                infofile = segmentationfolder + selectedfolder + "/Image_info.txt"
                shutil.copy(segmentationfile, Patient.Folders.InputFolder)
                shutil.copy(infofile, Patient.Folders.InputFolder)
                Patient.LoadSegmentedVessels(
                    CoWFile=CoWConfiguration + ".txt")  # This creates a new file, this is temporary until there is a module 2.
            else:
                print("Only using CoW file.")
                shutil.copy(anatomyfile, Patient.Folders.InputFolder + "1-D_Anatomy_Patient.txt")
    else:
        print("Loading preexisting patent anatomy file.")
    # load other files
    Patient.Load1DAnatomy()
    Patient.LoadPatientData()
    Patient.UpdateModelParameters()

    if newnetwork == "True" and Patient.ModelParameters["NewBravaSet"] == "True":
        Patient.SelectDonorNetwork()

    # Mapping to system vessels
    # The other vessels are mapped in the merge function
    mappingfile = Patient.Folders.DataFilesFolder + "/MappingSystemVessels.vtp"
    GeneralFunctions.VesselMapping(mappingfile, Patient)

    # Load selected donor network
    Brava = PatientClass.Patient(patient_folder)
    DonorFile = Brava.Folders.DataFilesFolder + "/Brava/" + Patient.ModelParameters["Donor_Network"]
    Brava.LoadVTPFile(DonorFile)

    # Load the pial surface
    surfacefile = Patient.Folders.ModellingFolder + "PialSurface.vtp"
    if not GeneralFunctions.is_non_zero_file(surfacefile):
        # if file does not exist, create a new one from the .ply file.
        # remesh the mesh to a uniform triangulation.
        print("Pial Surface file not found.")
        Remesh.remesh(Patient.Folders.ModellingFolder + "boundary_4&21&22&23&24&25&26&30.ply", numbertriangles=100000,
                      output=Patient.Folders.ModellingFolder + "remeshed.vtp")
        # apply the same mapping to the remeshed file
        Remesh.MapMeshtoMSH(Patient.Folders.ModellingFolder + "remeshed.vtp",
                            Patient.Folders.ModellingFolder + "labelled_vol_mesh.msh",
                            output=Patient.Folders.ModellingFolder + "PialSurface.vtp")

    Brava.Perfusion.PrimalGraph.LoadSurface(surfacefile)
    Brava.Perfusion.regionids = [2, 3, 4, 5, 6, 7, 8, 9]
    Brava.VesselToMeshAllignmentSides(Brava.Perfusion.PrimalGraph)
    # Brava.Perfusion.SetDualGraph(method="edges")
    Brava.Perfusion.SetDualGraph(method="vertices")
    Brava.Perfusion.RemapMajorRegions()

    GeneralFunctions.Merge(Patient, Brava)
    del Brava
    Patient.Topology.UpdateM2Names()
    Patient.ImportClots(Patient.Folders.InputFolder+Patient.ModelParameters["ClotFile"] + ".txt")
    Patient.TopologyToVTP()
    Patient.CerebellumBrainstemMapping()

    # Calculate mapping
    # Patient.MajorArteriesToPialSurfaceNN(Patient.Perfusion.DualGraph)
    # or mapping from VEASL data
    Patient.Perfusion.DualGraph.MajorVesselID = Patient.Perfusion.PrimalGraph.map

    Patient.FindCouplingPoints()
    # Patient.GenerateTreesAtCps(0.20)  # cutoff strongly scales the number of coupling points.
    # Patient.PerfusionEndsNumber()
    # Patient.MappingSixMajorArteries()

    # clustering of the pial surface, pick one of three methods
    # Patient.Perfusion.ClusteringMetis(Patient.Perfusion.DualGraph,distancemat=Patient.Folders.ModellingFolder+"Distancemat.npy",maxiter=10, useregion=False)
    # Patient.Perfusion.Clustering(Patient.Perfusion.DualGraph,distancemat=Patient.Folders.ModellingFolder+"Distancemat.npy", method="", fractiontriangles=1, debug=False,maxiter=10,useregion=True)  # default is the Dijkstra distance

    ClusteringCompute = Patient.ModelParameters["ClusteringCompute"]
    # ClusteringCompute = "False"
    if ClusteringCompute == "True":
        Patient.Perfusion.ClusteringByRegion(Patient.Perfusion.DualGraph,
                                             distancemat=Patient.Folders.ModellingFolder + "Distancemat.npy", method="",
                                             fractiontriangles=(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5), debug=False,
                                             maxiter=20, useregion=True)  # default is the Dijkstra distance

        Patient.Perfusion.OutputClusteringSteps(file=Patient.Folders.ModellingFolder + "OutputSteps.vtp",
                                                rootsfile=Patient.Folders.ModellingFolder + "RootsOutputSteps.vtp")  # disable this to speed up generation
        Patient.Perfusion.MapDualGraphToPrimalGraph()
    else:
        # load pre-computed clustering
        # surfacefile = "/home/raymond/Desktop/Clustering.vtp"
        # surfacefile = Patient.Folders.ModellingFolder + "Clustering.vtp"
        surfacefile = Patient.Folders.DataFilesFolder + "/Clustering_Lib/" + Patient.ModelParameters["Donor_Network"][:-4] + "/Clustering.vtp"
        Patient.Perfusion.PrimalGraph.LoadSurface(surfacefile)
        Patient.Perfusion.DualGraph.NodeColour = Patient.Perfusion.PrimalGraph.PolygonColour

        # identify and reorder the coupling points
        # file = "/home/raymond/Desktop/Clusters.csv"
        # file = Patient.Folders.ModellingFolder + "/Clusters.csv"
        file = Patient.Folders.DataFilesFolder + "/Clustering_Lib/" + Patient.ModelParameters["Donor_Network"][:-4] + "/Clusters.csv"
        clusters = [i.strip('\n').split(',') for i in open(file)]
        Position = [i[1].split(' ') for i in clusters[1:]]
        Position = [[float(i[0]), float(i[1]), float(i[2])] for i in Position]

        # find the closest outlet nodes in relation to the positions
        pos = [i.Node.Position for i in Patient.Perfusion.CouplingPoints]
        KDTree = scipy.spatial.KDTree(pos)
        _, MinDistanceIndex = KDTree.query(Position, k=1)

        if len(set(MinDistanceIndex)) is not len(Patient.Perfusion.CouplingPoints):
            print("Error in mapping the coupling points.")
        Patient.Perfusion.CouplingPoints = [Patient.Perfusion.CouplingPoints[i] for i in MinDistanceIndex]
        Patient.Perfusion.ClusterArea(Patient.Perfusion.PrimalGraph)

    Patient.Perfusion.DualGraph.IdentifyNeighbouringRegions()
    for out in Patient.Topology.OutletNodes:
        out.connected_cp = []
    for NN, cp in zip(Patient.Perfusion.DualGraph.connected_regions, Patient.Perfusion.CouplingPoints):
        cp.Node.connected_cp = [Patient.Perfusion.CouplingPoints[connected_region].Node for connected_region in NN]

    # write all files needed for the BF simulation
    Patient.CalculateMaximumTimestep()
    Patient.WriteModelParameters()
    Patient.WriteSimFiles()
    Patient.Perfusion.PrimalGraph.ExportTriangleColour(Patient.Folders.ModellingFolder)
    Patient.Perfusion.WriteClusteringMapping(Patient.Folders.ModellingFolder)
    Patient.ExportSurface(Patient.Folders.ModellingFolder + "Clustering.vtp", Patient.Perfusion.PrimalGraph)

    # export functions
    Patient.Perfusion.PrimalGraph.GraphToVTP(Patient.Folders.ModellingFolder)
    Patient.Perfusion.DualGraph.GraphToVTP(Patient.Folders.ModellingFolder)
    Patient.Perfusion.DualGraph.GraphEdgesToVTP(Patient.Folders.ModellingFolder + "DualGraphEdges.vtp")
    # export system
    Patient.Topology.WriteNodesCSV(Patient.Folders.ModellingFolder + "Nodes.csv")
    Patient.Topology.WriteVesselCSV(Patient.Folders.ModellingFolder + "Vessels.csv")

    # map the clustering back to the msh file.
    file1 = Patient.Folders.ModellingFolder + "Clustering.vtp"
    filehighres = Patient.Folders.ModellingFolder + "labelled_vol_mesh.msh"
    GeneralFunctions.MapClusteringToMSH(file1, filehighres, Patient.Folders.ModellingFolder)

    # generate collaterals
    if Patient.ModelParameters["CollateralPatientGen"] == "True":
        Collaterals.add_network_pa_nodes(Patient)

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    Distancemat = patient_folder + "/bf_sim/Distancemat.npy"
    if os.path.exists(Distancemat):
        os.remove(Distancemat)
    return Patient


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    patient_folder = arguments["<patient_folder>"]
    # patient_folder = "../Generated_Patients/patient_0/"

    try:
        shutil.rmtree(patient_folder + "/logfile.log")
    except:
        print("Error while deleting file ", patient_folder + "/logfile.log")

    transcript.start(patient_folder + 'bf_generation.log')
    patient = generatebloodflowfiles(patient_folder)
    transcript.stop()
