#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Classes that store patient and model parameters.
"""
import os
from os.path import dirname, realpath
import yaml
from lxml import etree


class Folders:
    def __init__(self, patientfolder):
        """
        Object containing paths.

        Initiate the object and set up the folder structure.

        Parameters
        ----------
        patientfolder : str
            Main folder containing all files
        """
        self.PatientFolder = patientfolder
        """Main folder containg all files"""
        self.InputFolder = patientfolder
        """Path to location with input files"""
        self.ModellingFolder = patientfolder + "bf_sim/"
        """Path to store bf simulation files"""
        self.DataFilesFolder = os.path.dirname(realpath(dirname(__file__))) + "/DataFiles/"
        """Path to folder containing model code"""

    def SetPatientFolder(self, patientfolder):
        """
        Update the folder structure

        Parameters
        ----------
        patientfolder : str
            Main folder containing all files
        """
        self.PatientFolder = patientfolder
        self.InputFolder = patientfolder
        self.ModellingFolder = patientfolder + "bf_sim/"


class PatientData(dict):
    def __init__(self):
        """
        Dictionary of patient parameters.

        Initiate the object with some default parameters
        """
        dict.__init__(self)
        self.File = ""
        self["Age"] = 50  # years
        self["Sex"] = "Male"
        self["HeartRate"] = 80  # per minute
        self["SystolePressure"] = 17300  # Pa
        self["DiastolePressure"] = 10100  # Pa
        self["MeanRightAtrialPressure"] = 0  # Pa
        self["StrokeVolume"] = 100  # mL/s
        self["CollateralScore"] = int(0)  # default collateral score

    def LoadPatientData(self, file):
        """
        Load patient parameters and add them to the internal dict.

        Parameters
        ----------
        file : str
            patient parameter txt file
        """
        print("Loading %s" % file)
        self.File = file
        data = [line.strip('\n').split('=') for line in open(file)]
        for line in data:
            try:
                datavalue = int(line[1])
            except ValueError:
                try:
                    datavalue = float(line[1])
                except ValueError:
                    datavalue = line[1]
            self[line[0]] = datavalue

    def LoadPatientDataXML(self, xml_file):
        """
        Load patient parameters and add them to the internal dict.

        Parameters
        ----------
        xml_file : str
            xml file
        """
        print("Loading %s" % xml_file)
        try:
            f = open(xml_file, "rb+")
        except IOError:
            print("File open error")
            exit(1)

        root_xml = etree.parse(f)
        patient_xml = root_xml.find("Patient")
        self.File = xml_file
        # (Age, Sex) appear with and without capital..
        for key in ["Age", "Sex"]:
            if patient_xml.find(key) is not None:
                self[key] = patient_xml.find(key)
            else:
                self[key] = patient_xml.find(key.lower())
        self["HeartRate"] = float(patient_xml.find("HeartRate").text)  # per minute
        self["SystolePressure"] = float(patient_xml.find("SystolePressure").text)  # Pa
        self["DiastolePressure"] = float(patient_xml.find("DiastolePressure").text)  # Pa
        self["MeanRightAtrialPressure"] = float(patient_xml.find("MeanRightAtrialPressure").text)  # Pa
        self["StrokeVolume"] = float(patient_xml.find("StrokeVolume").text)  # mL/s
        try:
            self["CollateralScore"] = int(patient_xml.find("collaterals").text)  # score
        except:
            self["CollateralScore"] = int(0)

    def LoadPatientDataYML(self, yml_file):
        """
        Load patient parameters and add them to the internal dict.

        Parameters
        ----------
        yml_file : str
            yml file
        """
        print("Loading %s" % yml_file)
        self.File = yml_file
        with open(yml_file, "r") as configfile:
            config = yaml.load(configfile, yaml.SafeLoader)

        for key, value in config.items():
            self[key] = value

    def WritePatientData(self, file="Patient_parameters.txt"):
        """
        Export patient parameters to txt file.

        Parameters
        ----------
        file : str
             file name
        """
        print("Writing: %s" % file)
        with open(file, 'w') as f:
            for key, val in self.items():
                f.write(key + "=" + str(val) + "\n")


class ModelParameter(dict):
    def __init__(self):
        """
        Dictionary of model parameters.
        """
        dict.__init__(self)
        self["RTotal"] = 1.19e8  # N S^-1 m^-5
        self["CTotal"] = 2.38e-9  # m^5 N^-1
        self["Density"] = 1040  # kg M^-3
        self["YoungsModules"] = 225e3  # Pa
        self["TimeConstantDiastolicDecay"] = 1.34
        self["coarse_collaterals_number"] = 0.0

    def LoadModelParameters(self, file):
        """
        Load model parameters and add them to the internal dict.

        Parameters
        ----------
        file : str
            file containing model parameters
        """
        print("Loading %s" % file)
        data = [line.strip('\n').split('=') for line in open(file)]
        for line in data:
            try:
                datavalue = int(line[1])
            except ValueError:
                try:
                    datavalue = float(line[1])
                except ValueError:
                    datavalue = line[1]
            self[line[0]] = datavalue

    def WriteModelParameters(self, file="Model_parameters.txt"):
        """
        Export model parameters to txt file.

        Parameters
        ----------
        file : str
            file name
        """
        print("Writing: %s" % file)
        with open(file, 'w') as f:
            for key, val in self.items():
                f.write(key + "=" + str(val) + "\n")
