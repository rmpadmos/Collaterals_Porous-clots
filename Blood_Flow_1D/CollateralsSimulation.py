#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Collateral Simulator

Script to run a simulation of the pial network.
Input argument is the patient folder with a folder for input files and a folder for modelling files
The input folder contains patient data and the modelling file folder contains files for the models
such as the parameters and the surface mesh.

Usage:
  tavernaBloodFlow.py <patient_folder>
  tavernaBloodFlow.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from datetime import datetime

import numpy as np
import scipy.spatial
import scipy.stats
import contextlib
from Blood_Flow_1D import Collaterals, ContrastGraphModel, GeneralFunctions, Metadata, Patient, docopt, transcript


def collateral_simulation(patient, clot_active=False, lower_bound_mmhg=10, upper_bound_mmhg=120):
    if clot_active:
        print("\033[96mSimulating Collaterals with autoregulation: Stroke scenario. \033[m")
    else:
        print("\033[96mSimulating Collaterals with autoregulation: Healthy scenario. \033[m")
    brain_perfusion_estimate = patient.ModelParameters["brainperfusionestimate"]
    venous_pressure = patient.ModelParameters["OUT_PRESSURE"]
    target_flow = brain_perfusion_estimate / patient.Topology.NumberOfPenetratingArteries
    lower_bound = 133.322 * lower_bound_mmhg / target_flow
    upper_bound = 133.322 * upper_bound_mmhg / target_flow
    patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clot_active, PressureInlets=True,
                                  coarseCollaterals=False, frictionconstant=patient.ModelParameters["FRICTION_C"],
                                  scale_resistance=False)

    # stop updating resistance if max change in less than rel_tol
    print("\tAdjusting resistances.")
    rel_tol = 1e-4
    relative_residual = 1
    while relative_residual > rel_tol:
        oldR = np.array([node.R1 + node.R2 for node in patient.Topology.PAnodes])
        for node in patient.Topology.PAnodes:
            resistance = (node.Pressure - venous_pressure) / target_flow
            minr = min(resistance, upper_bound)
            resistance = max(minr, lower_bound)
            node.R2 = resistance
            node.R1 = 0
        with contextlib.redirect_stdout(None):
            patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clot_active, PressureInlets=True,
                                          coarseCollaterals=False,
                                          frictionconstant=patient.ModelParameters["FRICTION_C"],
                                          scale_resistance=False)
        # relative_residual = max(
        #     [abs(node.R1 + node.R2 - oldR[index]) / (node.R1 + node.R2) for index, node in
        #      enumerate(patient.Topology.PAnodes)])
        solutionnew = np.array([abs(node.R1 + node.R2) for node in patient.Topology.PAnodes])
        diff = oldR-solutionnew
        relative_residual = np.linalg.norm(diff, np.inf) / np.linalg.norm(solutionnew, np.inf)
        print(f'\tMax relative residual: {relative_residual}')
    print("Convergence!")

    # while 1:
    #     # Update resistances to get uniform flow (similar to autoregulation)
    #     # keep updating the outlet resistances until flow does not change.
    #     # update PA resistances
    #     old_flows = [node.FlowRate for node in patient.Topology.PAnodes]
    #     for node in patient.Topology.PAnodes:
    #         resistance = (node.Pressure - venous_pressure) / target_flow
    #         minr = min(resistance, upper_bound)
    #         resistance = max(minr, lower_bound)
    #         node.R2 = resistance
    #         # if resistance >= upperbound or resistance <= lower_bound:
    #         #     print("Resistance outside range! %f [%f-%f]" % (resistance, lower_bound, upper_bound))
    #     patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clot_active, PressureInlets=True,
    #                                   FlowRateOutlets=False, coarseCollaterals=False,
    #                                   frictionconstant=patient.ModelParameters["FRICTION_C"], scale_resistance=True)
    #     current_flow = [node.FlowRate for node in patient.Topology.PAnodes]
    #     difference = np.linalg.norm([flow - old_flow for flow, old_flow in
    #                                  zip(current_flow, old_flows)]) / np.linalg.norm(current_flow)
    #     print("Autoregulation Convergence criteria %f" % difference)
    #     if difference < 1e-3:
    #         break


def CollateralSimulation(patient_folder):
    """
    This function loads the model, adds the collaterals, and runs both healthy and stroke simulations.

    Parameters
    ----------
    patient_folder : str
        Patient folder location

    Returns
    -------

    """
    start_time = datetime.now()
    transcript.start(patient_folder + 'logfile.log')

    # load parameters from input file
    ModelParameters = Metadata.ModelParameter()
    ModelParameters.LoadModelParameters(patient_folder + "/bf_sim/Model_parameters.txt")
    UsingPAfile = ModelParameters["UsingPAfile"]
    frictionconstant = ModelParameters["FRICTION_C"]

    # if GenerateNewFiles == "True":
    #     GenerateBloodflowFiles.generatebloodflowfiles(patient_folder)

    # Load files
    patient = Patient.Patient(patient_folder)
    patient.LoadBFSimFiles()
    patient.LoadClusteringMapping(patient.Folders.ModellingFolder + "Clusters.csv")
    patient.LoadSurfaceMapping()
    patient.LoadPositions()
    patient.Perfusion.PrimalGraph.LoadSurface(patient.Folders.ModellingFolder + "Clustering.vtp")
    patient.Perfusion.SetDualGraph(method="vertices")
    patient.Perfusion.DualGraph.NodeColour = patient.Perfusion.PrimalGraph.map

    # Collaterals.add_network_pa_nodes(patient)
    PAfile = patient.Folders.ModellingFolder + "PA.vtp"
    Collaterals.ImportPAnodes(patient)

    # Healthy case
    print("Solving healthy scenario.")
    patient.Initiate1DSteadyStateModel()
    patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=False, PressureInlets=True,
                                  coarseCollaterals=False, frictionconstant=patient.ModelParameters["FRICTION_C"],
                                  scale_resistance=True)
    # auto regulation for healthy case
    collateral_simulation(patient, clot_active=False)

    # export results for the healthy case
    Collaterals.ExportSimulationResults(patient)
    Collaterals.CalculatePressureDrop(patient)
    # contrast model
    patient.Topology.TopologyToGraph()
    ContrastGraphModel.CalculateTimeDelay(patient)
    ContrastGraphModel.WriteTimeDelays(patient)
    ContrastGraphModel.AddTimeDelaysToTopologyFile(patient)
    # if using PAfile, map results to the Pa file
    if UsingPAfile == "True":
        Collaterals.ExportResultsToPAfile(patient, PAfile, healthy=True)
        Collaterals.ExportResultsToPAfile(patient, patient.Folders.ModellingFolder + "DualPA.vtp", healthy=True,
                                          cellArray=True)
        resistances = [node.R2 for node in patient.Topology.PAnodes]
        GeneralFunctions.AddArrayToFile(patient.Folders.ModellingFolder + "DualPA.vtp", resistances,
                                        "Resistance Healthy", True)
        GeneralFunctions.AddArrayToFile(PAfile, resistances, "Resistance Healthy", False)

        KDTree = scipy.spatial.KDTree(patient.Perfusion.DualGraph.PialSurface)
        pos = [node.Position for node in patient.Topology.PAnodes]
        # euclidean distance between outlets and surface
        _, MinDistanceIndex = KDTree.query(pos, k=1)
        MajorVesselIDPA = [patient.Perfusion.PrimalGraph.map[MinDistanceIndex[index]] for index, node in
                           enumerate(patient.Topology.PAnodes)]
        GeneralFunctions.AddArrayToFile(patient.Folders.ModellingFolder + "DualPA.vtp", MajorVesselIDPA,
                                        "Major vessel ID", True)

        # ContrastGraphModel.CalculatePerfusionTerritories(patient, PAfile)

    Collaterals.ExportSystemResult(patient, patient.Folders.ModellingFolder, file="ContrastMeasurements.csv")

    mean = np.mean([node.Pressure for node in patient.Topology.PAnodes])
    se = scipy.stats.sem([node.Pressure for node in patient.Topology.PAnodes])
    sd = np.std([node.Pressure for node in patient.Topology.PAnodes])
    print(f"Pressure stats (Healthy), mean: {mean} sem: {se} sd: {sd}")

    # Clot case
    print("Solving stroke scenario.")
    # create new topology file for the clot case
    patient.TopologyToVTP(filename="TopologyClot.vtp")
    # auto regulation for clot case
    collateral_simulation(patient, clot_active=True)

    # export results for the clot case
    Collaterals.ExportSimulationResults(patient, healthy=False)
    Collaterals.CalculatePressureDrop(patient)
    # contrast model
    ContrastGraphModel.CalculateTimeDelay(patient)
    ContrastGraphModel.WriteTimeDelays(patient, file="GraphTimeDelayClot.csv")
    ContrastGraphModel.AddTimeDelaysToTopologyFile(patient, file="TopologyClot.vtp")
    # if using PA file, map results to the Pa file
    if UsingPAfile == "True":
        Collaterals.ExportResultsToPAfile(patient, PAfile, healthy=False)
        Collaterals.ExportResultsToPAfile(patient, patient.Folders.ModellingFolder + "DualPA.vtp", healthy=False,
                                          cellArray=True)
        resistances = [node.R2 for node in patient.Topology.PAnodes]
        GeneralFunctions.AddArrayToFile(patient.Folders.ModellingFolder + "DualPA.vtp", resistances, "Resistance Clot",
                                        True)
        GeneralFunctions.AddArrayToFile(PAfile, resistances, "Resistance Clot", False)
    Collaterals.ExportSystemResult(patient, patient.Folders.ModellingFolder, file="ContrastMeasurementsClot.csv",
                                   figname="TimedelayGraphClot.png", figname2="PressureClot.png")

    mean = np.mean([node.Pressure for node in patient.Topology.PAnodes])
    se = scipy.stats.sem([node.Pressure for node in patient.Topology.PAnodes])
    sd = np.std([node.Pressure for node in patient.Topology.PAnodes])
    print(f"Pressure stats (Stroke), mean: {mean} sem: {se} sd: {sd}")

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    transcript.stop()


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    patient_folder = arguments["<patient_folder>"]
    # patient_folder = "../Generated_Patients/patient_0_collaterals_run/"
    CollateralSimulation(patient_folder)
