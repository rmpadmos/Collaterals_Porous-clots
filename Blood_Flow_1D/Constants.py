#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Contains constants and dictionaries used in the model.
"""
# Used in Patient
fractions = [0.70, 0.05, 0.05, 0.20]
bodyparts = ['Thoracic aorta', 'R. brachial', 'L. brachial']

# Used in GeneralFunctionss
# Dictionary that maps major vessel ID to a name.
MajorIDNames = {0: "CoW and Other",
                2: "R. ACA",
                3: "R. MCA",
                4: "L. MCA",
                5: "L. ACA",
                6: "R. PCA",
                7: "L. PCA",
                8: "Cerebellum",
                9: "Brainstem",
                -1: "Undefined"}
# Dictionary that maps major vessel ID between the blood flow and perfusion models.
MajorIDdict = {
    5: 21,
    4: 22,
    7: 23,
    2: 24,
    3: 25,
    6: 26,
    8: 4,
    9: 30}
# Inverse dictionary that maps major vessel ID between the perfusion and blood flow models.
MajorIDdict_inv = {v: k for k, v in MajorIDdict.items()}

# starting value for the cluster IDs
StartClusteringIndex = 20
