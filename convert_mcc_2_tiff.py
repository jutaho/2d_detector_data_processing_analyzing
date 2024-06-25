# -*- coding: utf-8 -*-
"""
Read mcc-file generated by software "BeamAdjust" type "Octavius 1600 XDR"
Create an interpolated mcc-file 

J. Horn
"""

import re
import numpy as np
from scipy.interpolate import griddata
from tifffile import imwrite
       
def read_1600xdr_mcc(file):
    """
    read mcc-file type Octavius 1600 XDR

    Parameters
    ----------
    file : mcc-file generated by software BeamAdjust - only type Octavius 1600 XDR 

    Returns
    -------
    dose_array : float64 numpy array with all dose values inlcuding nans
    
    """
    # Define empty lists 
    dose_values = []
    chamber_number = []
  
    # Load chamber layout Octavois 1600 XDR 
    detector_layout = np.load('layout_OD1600XDR.npy')
    
    # Extract only dose values and chamber indices
    string_pattern = re.compile('#') 
    with open(file) as raw_data:
        for line in raw_data:
            if string_pattern.search(line):
                relevant_values = line.split('\t')
                dose_values.append(float(relevant_values[5]))
                chamber = relevant_values[7].strip('#\n')
                chamber_number.append(int(chamber))
        dose_values = np.array(dose_values)
        chamber_number = np.array(chamber_number)
        # Cut last three blocks = central profile, first and second diagonal
        dose_values = dose_values[0:-135]
        chamber_number = chamber_number[0:-135]
        mcc_data_container = np.array([chamber_number, dose_values])
        
    # Create copy of layout, store dose values, keep nan
    dose_matrix = detector_layout.copy()
    for rows in range(dose_matrix.shape[0]):
        for cols in range(dose_matrix.shape[1]):
            if not np.isnan(dose_matrix[rows, cols]):
                # Find indices in mccDataContainer where the chamber number matches dose_matrix[rows, cols]
                indices = np.where(mcc_data_container[0] == dose_matrix[rows, cols])
                if indices[0].size == 1:  # Check if there is only one corresponding match
                    # Extract the matching indices
                    indices = indices[0][0]
                    # Get the corresponding dose value for this indices
                    dose = mcc_data_container[1][indices]
                    # Assign the dose value to dose_matrix
                    dose_matrix[rows, cols] = dose
                    
    dose_array = dose_matrix
    return dose_array

def interpolate_nan_values(dose_array):
    """
    Parameters
    ----------
    dose_array : float64 numpy array with all dose values inlcuding nans 

    Returns
    -------
    dose_array : float64 numpy array with interpolated dose values
    
    """
    # Create a grid for the data points
    y, x = np.indices(dose_array.shape)

    # Identify the valid (non-NaN) and invalid (NaN) points
    valid_mask = ~np.isnan(dose_array)
    invalid_mask = np.isnan(dose_array)

    # Extract the coordinates and values of the valid points
    # .T transposes the array to have shape (N, 2) instead of (2, N)
    valid_points = np.array((x[valid_mask], y[valid_mask])).T
    valid_values = dose_array[valid_mask]

    # Extract the coordinates of the invalid points
    invalid_points = np.array((x[invalid_mask], y[invalid_mask])).T

    # Perform interpolation
    interpolated_values = griddata(valid_points, valid_values, invalid_points, method='linear')

    # Fill the NaN values in the original array with the interpolated values
    dose_array[invalid_mask] = interpolated_values
    dose_interp_nan_array = np.copy(dose_array)

    return dose_interp_nan_array

def interpolate_finer(dose_interp_nan_array, scale=10):
    """
    Parameters
    ----------
    dose_interp_nan_array : TYPE
        DESCRIPTION.
    scale : TYPE, optional

    Returns
    -------
    finer_dose_array :

    """
    # Original grid
    y, x = np.indices(dose_interp_nan_array.shape)
    
    # New finer grid
    new_shape = (dose_interp_nan_array.shape[0] * scale, dose_interp_nan_array.shape[1] * scale)
    new_y, new_x = np.linspace(0, dose_array.shape[0]-1, new_shape[0]), np.linspace(0, dose_array.shape[1]-1, new_shape[1])
    new_y, new_x = np.meshgrid(new_y, new_x)
    
    # Flatten the grids for interpolation
    original_points = np.array((x.ravel(), y.ravel())).T
    new_points = np.array((new_x.ravel(), new_y.ravel())).T
    
    # Perform interpolation
    finer_dose_array = griddata(original_points, dose_array.ravel(), new_points, method='linear')
    
    # Reshape back to the new grid shape
    finer_dose_array = finer_dose_array.reshape(new_shape)
    
    return finer_dose_array

def write_tiff_file(finer_dose_array):
    """
    Parameters
    ----------
    finer_dose_array : TYPE
        DESCRIPTION.

    Returns
    -------
    tiff

    """
    # convert to 16 bit
    pixel_spacing_mm = 0.25
    pixel_spacing_inch = pixel_spacing_mm / 25.4
    resolution_ppi = 1 / pixel_spacing_inch
    normalize_dose = np.max(finer_dose_array)
    normalized_dose_array = finer_dose_array / normalize_dose
    dose_array_16bit = (normalized_dose_array * (2**16 - 1)).astype(np.uint16)
    dose_array_16bit_inverted = (2**16 - 1) - dose_array_16bit
    imwrite("N:\Desktop\original.tiff", dose_array_16bit_inverted, resolution=(resolution_ppi, resolution_ppi))  # 1/0.25 = 4 pixels/mm
    
# Usage
if __name__ == "__main__":
    file = "N:\Desktop\original.mcc"
    dose_array = read_1600xdr_mcc(file)
    # Interpolate the NaN values
    interpolated_dose_array = interpolate_nan_values(dose_array)
    finer_dose_array = interpolate_finer(interpolated_dose_array)
    write_tiff_file(finer_dose_array)