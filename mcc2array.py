# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:33:42 2024

@author: hornjulian
"""

import re
import numpy as np

# Open file, read data       
def readRawDoseData(file):
    # Define variables 
    doseValues = []
    chamberNb = []
  
    # Load chamber layout OD1600XDR 
    layout = np.load('LayoutOD1600XDR.npy')
    
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
    for col in range(doseMatrix.shape[0]):
        for row in range(doseMatrix.shape[1]):
            if not np.isnan(doseMatrix[row, col]):
                # Find the index in mccDataContainer where the chamber number matches doseMatrix[row, col]
                buffer = np.where(mccDataContainer[0] == doseMatrix[row, col])
                if buffer[0].size > 0:  # Check if there is at least one match
                    # Extract the first match index
                    buffer = buffer[0][0]
                    # Get the corresponding dose value
                    dose = mccDataContainer[1][buffer]
                    # Assign the dose value to doseMatrix
                    doseMatrix[row, col] = dose
                
    doseArray = doseMatrix
    
    return doseArray, layout

# Example usage
file_path = "r:\\Therapie_QA\\2024\\H2\\E-2-2-1_Homogenität\\06.Juni\\testPython\\original.mcc"
doseArray, layout = readRawDoseData(file_path)
