#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description -  
# Last change -     08.06.2023  
# Author -          J. Horn

# Import modules
import os
from fnmatch import fnmatch
from PIL import Image
import analyzeFieldFlatness as field
import numpy as np
from scipy import signal

# Info: Either flat-panel or calculated reference

# Create variables and search pattern
pattern = ["*.tif*"]
root = "E:\\_untersuchungen\\_untersuchungenOctavius1600XDR\\062023_PTCOG61\\data\\refFluenceDist\\" # hit PC

pathAndFile = []
file =[]
folder = []

# Look for .tif-files
for elements in range(len(pattern)):
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern[elements]):
                folder.append(path)
                file.append(name)
                pathAndFile.append(os.path.join(path, name))
 
# Evaluate all data
for elements in range(len(pathAndFile)):
        print(pathAndFile[elements])
        img = Image.open(pathAndFile[elements])
        #img = signal.medfilt2d(img,7)
        img = np.array(img)
        img = abs((2**16-1)) - img
        #img = np.rot90(img,3)
        field.analyze(img, 0.2 ,pathAndFile[elements] + '.pdf', 'flatpanel')