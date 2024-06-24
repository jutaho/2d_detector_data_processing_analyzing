#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description - aaa
# Last change -     29.05.2023     
# Author -          J. Horn

# Import modules
import os
import numpy as np
from fnmatch import fnmatch
import analyzeFieldFlatness as field
import processImgXRD0822 as proc

# Create variables and search pattern

root = "E:\\_rawData\\_rawDataFlatpanel\\20240618_H2_E221Check\\" # hit PC

pattern = ["2004*.his"]
kernelSize4MedianFilter = 5 # 1 = no median filter
pathAndFile = []
file =[]
folder = []
imgContainer = []

# Look for relevant ".his"-files
for elements in range(len(pattern)):
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern[elements]):
                folder.append(path)
                file.append(name)
                pathAndFile.append(os.path.join(path, name))
            
# Process frames
for elements in range(len(pathAndFile)):
        img, imgQual, pxlShiftInMm, imgs2Corr, bg = proc.processRawFrames(
            pathAndFile[elements], kernelSize4MedianFilter)
        imgContainer.append(img)
        if np.sum(img) == 0 :
            print("image contains no information")
        else:
            #imgi = proc.writeTIFF(img, imgQual[0], imgQual[1],
            #pathAndFile[elements])
            field.analyze(img, 0.2, pathAndFile[elements] + '.pdf', 'flatpanel')
            