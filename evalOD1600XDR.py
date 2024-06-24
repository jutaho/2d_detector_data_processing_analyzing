#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description -     
# Last change -     08.06.2023
# Author -          J. Horn

# Import modules
import os
from fnmatch import fnmatch
import readDatOD1600XDR as proc
import analyzeFieldFlatness as field

# Create variables and search pattern
pattern = ["*.mcc"]
root = "r:\\Therapie_QA\\2024\\H1\\E-2-2-1_Homogenität\\06.Juni"
pathAndFile = []
file =[]
folder = []

# Look for relevant ".mcc"-files
for elements in range(len(pattern)):
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern[elements]):
                folder.append(path)
                file.append(name)
                pathAndFile.append(os.path.join(path, name))
            
# Process frames
for elements in range(len(pathAndFile)):
        print(pathAndFile[elements])
        dose1, dose2 = proc.readRawDoseData(pathAndFile[elements])
        field.analyze(dose2, 2.5 ,pathAndFile[elements] + '.pdf', 'OD1600XDR')
        