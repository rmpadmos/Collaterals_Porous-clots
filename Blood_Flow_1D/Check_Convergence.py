#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Script to calculate whether the 1-D pulsatile model has converged.
"""
import sys

import numpy

from Blood_Flow_1D import GeneralFunctions
from Blood_Flow_1D import Results as ResultClass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        Folder = str(sys.argv[1])
    else:
        Folder = "../Generated_Patients/patient_0/Modelling_files/"

    ResultsFile = "Results.dyn"
    ResultsFile2 = "ResultsPrev.dyn"
    ResultsFile3 = "ResultsTotal.dyn"
    Conv = 0

    if GeneralFunctions.is_non_zero_file(Folder + ResultsFile) and GeneralFunctions.is_non_zero_file(
            Folder + ResultsFile2):
        Results = ResultClass.Results()
        Results.LoadResults(Folder + ResultsFile)

        Results2 = ResultClass.Results()
        Results2.LoadResults(Folder + ResultsFile2)

        p1 = numpy.array([item for sublist in Results.Pressure[-1] for item in sublist])
        p2 = numpy.array([item for sublist in Results2.Pressure[-1] for item in sublist])
        p1_norm = numpy.linalg.norm(p1)
        pressure_norm = numpy.linalg.norm(p1 - p2) / p1_norm

        v1 = numpy.array([item for sublist in Results.VolumeFlowRate[-1] for item in sublist])
        v2 = numpy.array([item for sublist in Results2.VolumeFlowRate[-1] for item in sublist])
        v1_norm = numpy.linalg.norm(v1)
        flow_norm = numpy.linalg.norm(v1 - v2) / v1_norm

        r1 = numpy.array([item for sublist in Results.Radius[-1] for item in sublist])
        r2 = numpy.array([item for sublist in Results2.Radius[-1] for item in sublist])
        r1_norm = numpy.linalg.norm(r1)
        radius_norm = numpy.linalg.norm(r1 - r2) / r1_norm

        print("Max Residual: Pressure: %f  Flow rate: %f Radius: %f" % (pressure_norm, flow_norm, radius_norm))

        # if pressure residual  < 1e-3 and flow residual < 1e-3 and velocity residual < 1e-3:
        with open(Folder + "Convergence.csv", "a") as f:
            f.write("%f,%f,%f\n" % (pressure_norm, flow_norm, radius_norm))

        if pressure_norm < 1e-3:
            Conv = 1

    with open(Folder + ResultsFile) as f:
        with open(Folder + ResultsFile3, "a") as f1:
            for line in f:
                f1.write(line)

    if Conv == 0:
        with open(Folder + ResultsFile) as f:
            with open(Folder + ResultsFile2, "w") as f1:
                for line in f:
                    f1.write(line)

    with open(Folder + "Conv.txt", "w") as f:
        f.write(str(Conv))
    exit()
