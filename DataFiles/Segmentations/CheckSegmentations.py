#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Check a folder of segmentations if there are segmentations lacking key vessels.
"""
import csv
import os

def CheckSegmentationFolder(folder):
    """
    Function to check whether every segmentation has no missing vital vessels.

    Parameters
    ----------
    folder : str
        folder of segmentation folders
    """
    segmentations = sorted(os.listdir(folder), key=lambda f: int(f[-4:]))
    for segmentation in segmentations:
        file = folder + segmentation + "/Feature_Vessel.csv"
        CheckSegmentation(file)


def CheckSegmentation(file):
    """
    We need at least the six major branches and the three major vessels (ICAs, BA).
    if any vessel is missing, output the folder name

    Parameters
    ----------
    file : str
        segmentation csv file.
    """
    requiredvesselids = [1, 2, 3, 4, 9, 10, 16, 19, 20]
    vesselscsv = []
    with open(file) as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        for row in readcsv:
            vesselscsv.append(row)
    vesselids = [int(i[10]) for i in vesselscsv[1:]]
    for vesselid in requiredvesselids:
        if vesselid not in vesselids:
            print("Segmentation is missing one or more key vessels: %s" % file)
            break


if __name__ == '__main__':
    folder = "Segmentations/"
    CheckSegmentationFolder(folder)
