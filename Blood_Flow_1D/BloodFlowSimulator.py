#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""1D Blood Flow Simulator
Script to run a blood flow simulation using a previously generated patient.
The blood flow model can be pulsatile or steady state.
Clots can be included in the blood flow model.

Three inputs are needed.
<executable_location> is the location of the executable for the pulsatile blood flow simulations.
If the steady state model is used, the argument can be anything.

<patient_folder> argument is the patient folder with a folder for input files and a folder for modelling files
The input folder contains patient data.
The modelling file folder contains files for the models such as the parameters and the surface mesh.

--clot_present is a flag that indicates clots are to be simulated by the models.

Usage:
  BloodFlow.py <executable_location> <patient_folder> [--clot_present]
  BloodFlow.py (-h | --help)

Options:
  -h --help     Show this screen.
  --clot_present        Simulations consider clots.
"""

import subprocess
from datetime import datetime

from Blood_Flow_1D import transcript, Patient, Collaterals, GeneralFunctions, Results, docopt


def blood_flow_script(executable, patient_folder, clot_present):
    """
    Function to run a blood flow simulation on a previously generated patient.
    Option to choose between pulsatile and steady state models.

    Parameters
    ----------
    executable : str
        Location of BloodflowModel.exe
    patient_folder : str
        Location of patient folder
    clot_present : boolean
        True or False if clot should be included in the simulations

    Returns
    -------
    patient : patient object
        patient object containing network and results.
    """
    patient = Patient.Patient(patient_folder)
    # remove result files if these were created during a previous run
    patient.RemoveOldSimFiles()
    patient.LoadBFSimFiles()

    model = patient.ModelParameters["Model"]
    friction_constant = patient.ModelParameters["FRICTION_C"]
    # model = "Pulsatile"
    # model = "Steady"
    results_folder = patient.Folders.ModellingFolder
    if model == "Pulsatile":
        if clot_present:
            subprocess.call(["mono", executable, results_folder + "Run.txt",
                             results_folder + "Results.dyn", results_folder + "Clots.txt"])
        else:
            subprocess.call(["mono", executable, results_folder + "Run.txt",
                             results_folder + "Results.dyn"])
        patient.LoadResults("Results.dyn", correct=False)
    elif model == "Steady":
        patient.Initiate1DSteadyStateModel()
        if clot_present:
            patient.Run1DSteadyStateModel(model="Linear", clotactive=True, frictionconstant=friction_constant)
        else:
            patient.Run1DSteadyStateModel(model="Linear", clotactive=False, frictionconstant=friction_constant)

        # export data in same format at the 1-D pulsatile model
        # start point t=0
        time_point = Results.TimePoint(0)
        time_point.Flow = [node.FlowRate for node in patient.Topology.Nodes]
        time_point.Pressure = [node.Pressure for node in patient.Topology.Nodes]
        time_point.Radius = [node.Radius for node in patient.Topology.Nodes]
        # end point, t=duration of a single heart beat
        time_point2 = Results.TimePoint(patient.ModelParameters['Beat_Duration'])
        time_point2.Flow = time_point.Flow
        time_point2.Pressure = time_point.Pressure
        time_point2.Radius = time_point.Radius

        patient.Results.TimePoints = [time_point, time_point2]
        patient.Results.ExportResults(results_folder + "Results.dyn")

        patient.LoadResults("Results.dyn")
    else:
        print("Error: Model not defined.")
        exit()
    patient.GetMeanResults()
    patient.ExportMeanResults()

    Collaterals.estimate_perfusion(patient)

    patient.DistributeFlowTriangles()
    patient.ExportTriangleFlowData()

    patient.WriteTimeseriesVessels()
    patient.Results.AddResultsPerNodeToFile(patient.Folders.ModellingFolder + "Topology.vtp")
    patient.Results.AddResultsPerVesselToFile(patient.Folders.ModellingFolder + "Topology.vtp")

    # update the flow to the perfusion model
    clustering_flow_data = patient.Folders.ModellingFolder + "ClusterFlowData.csv"
    clustering_file = patient.Folders.ModellingFolder + "Clusters.csv"
    data_folder = patient.Folders.ModellingFolder
    GeneralFunctions.WriteFlowFilePerfusionModel(clustering_flow_data, clustering_file, data_folder)

    # export bf values for the clots
    patient.ExportClotBFValues()

    # if clot_present:
    #     patient.Topology.NodeVesselDict()
    #     clot_vessels = set()
    #     clot_nodes = [node for clot in patient.Topology.Clots for node in clot[0]]
    #     for clot_node in clot_nodes:
    #         vessel = patient.Topology.NodeDict[clot_node]
    #         clot_vessels.add(vessel)
    #
    #     with open(patient.Folders.ModellingFolder + "/Pressure_Drop.csv", 'w') as f:
    #         f.write(
    #             "VesselName,Pressure Difference (pa),Pressure Difference (mmHg)\n")
    #
    #         # calculate pressure drop as abs(p1-p2)
    #         for vessel in list(clot_vessels):
    #             dp = abs(vessel.Nodes[0].Pressure - vessel.Nodes[-1].Pressure)
    #             vessel.PressureDrop = dp
    #             print(f"Pressure drop across the clot: {dp}")
    #             f.write("%s,%f,%f\n" % (vessel.Name, vessel.PressureDrop, vessel.PressureDrop * 0.007500617))

    return patient


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__, version='0.1')
    executable = arguments["<executable_location>"]
    patient_folder = arguments["<patient_folder>"]
    clot_present = arguments["--clot_present"]

    start_time = datetime.now()
    transcript.start(patient_folder + 'bf_simulation.log')
    blood_flow_script(executable, patient_folder, clot_present)
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    transcript.stop()
