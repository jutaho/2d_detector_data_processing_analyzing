#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description -     processes raw data from file ".mcc"-format of
#                   Octavius 1600 XDR
# Author -          J. Horn

# Last change -     29.05.2023

# Import modules
import re
import numpy as np
from scipy.interpolate import griddata

# Open file, read data       
def readRawDoseData(file):
    # Define variables 
    doseValues = []
    chamberNb = []
  
    # Load chamber layout OD1600XDR 
    layout = np.load('layout_OD1600XDR.npy')
    
    # Extract only dose values and chamber indices
    strPattern = re.compile('#') 
    with open(file) as rawData:
        for line in rawData:
            if strPattern.search(line):
                values = line.split('\t')
                doseValues.append(float(values[5]))
                chambValue = values[7].strip('#\n')
                chamberNb.append(int(chambValue))
        doseValues = np.array(doseValues)
        chamberNb = np.array(chamberNb)
        
        # cut last three blocks = central profile, first and second diagonal
        doseValues = doseValues[0:-135]
        chamberNb = chamberNb[0:-135]
        mccDataContainer = np.array([chamberNb,doseValues])
        
    # Copy of layout, store dose values, keep nan
    doseMatrix = layout.copy()
    for row in range(61):
        for col in range(61):
            if np.isnan(doseMatrix[row,col]) == False:
                buffer = np.where(mccDataContainer == doseMatrix[row,col])
                buffer = buffer[1][0]
                dose = mccDataContainer[1][buffer]
                doseMatrix[row,col] = dose
                
    #doseArray = np.rot90(doseMatrix,3)
    doseArray = doseMatrix
    
    # Interpolate NANs
    validRows, validCols = np.where(np.isnan(doseArray) == False) 
    validDose = doseArray[np.isnan(doseArray) == False]
    
    # New grid to interpolate
    gridRows, gridCols = np.meshgrid(np.arange(0,61),np.arange(0,61))
    
    # Use griddata to interpolate
    interpDoseArray = griddata((validRows,validCols), validDose, (gridRows, gridCols), method='linear')
    
    return doseArray, interpDoseArray
    
        