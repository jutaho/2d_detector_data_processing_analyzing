# -*- coding: utf-8 -*-
"""
Read mcc-file generated by software "BeamAdjust" type "Octavius 1600 XDR"
Create an interpolated mcc-file 

juh
"""

import re
import numpy as np
from tifffile import imwrite
       
def read_1600xdr_mcc(file):
    """
    read mcc-file type Octavius 1600 XDR

    Parameters
    ----------
    file : mcc-file generated by software BeamAdjust - only type Octavius 1600 XDR 

    Returns
    -------
    dose_array : float64 numpy array with all dose values inlcuding NaNs
    
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

def bilinear_interpolate_nan(dose_array):
    """
    Interpolates NaN values in the dose array using bilinear interpolation.

    Parameters
    ----------
    dose_array : np.ndarray
        2D array with NaN values to interpolate.

    Returns
    -------
    np.ndarray
        Array with NaN values interpolated.
    """
    y, x = np.indices(dose_array.shape)
    
    # Create masks for valid and invalid points
    valid_mask = ~np.isnan(dose_array)
    invalid_mask = np.isnan(dose_array)

    # Create arrays to store interpolated values
    interpolated_values = np.copy(dose_array)

    # For each NaN value, interpolate using bilinear interpolation
    for i in range(dose_array.shape[0]):
        for j in range(dose_array.shape[1]):
            if invalid_mask[i, j]:
                # Find the nearest non-NaN neighbors
                x0, x1 = max(0, j-1), min(dose_array.shape[1]-1, j+1)
                y0, y1 = max(0, i-1), min(dose_array.shape[0]-1, i+1)

                # Get the values and weights for the neighbors
                neighbors = []
                weights = []

                if valid_mask[y0, x0]:
                    neighbors.append(dose_array[y0, x0])
                    weights.append((x1 - j) * (y1 - i))
                if valid_mask[y0, x1]:
                    neighbors.append(dose_array[y0, x1])
                    weights.append((j - x0) * (y1 - i))
                if valid_mask[y1, x0]:
                    neighbors.append(dose_array[y1, x0])
                    weights.append((x1 - j) * (i - y0))
                if valid_mask[y1, x1]:
                    neighbors.append(dose_array[y1, x1])
                    weights.append((j - x0) * (i - y0))

                if neighbors:
                    interpolated_values[i, j] = np.average(neighbors, weights=weights)

    return interpolated_values

def bilinear_interpolate(source, scale):
    """
    Perform bilinear interpolation on a 2D array to scale up the resolution.

    Parameters
    ----------
    src : np.ndarray
        2D array to be interpolated.
    scale : int
        Factor by which to increase the grid resolution.

    Returns
    -------
    np.ndarray
        Interpolated 2D array with increased resolution.
    """
    src_h, src_w = source.shape
    dst_h, dst_w = (src_h * scale) +1, (src_w * scale) +1

    # Create coordinate grid for the destination
    dst_y, dst_x = np.mgrid[0:dst_h, 0:dst_w]

    # Calculate the corresponding coordinates in the source array
    src_x = dst_x / scale
    src_y = dst_y / scale

    # Calculate the coordinates of the four neighbors
    x0 = np.floor(src_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(np.int32)
    y1 = y0 + 1

    # Clip coordinates to be within the valid range
    x0 = np.clip(x0, 0, src_w - 1)
    x1 = np.clip(x1, 0, src_w - 1)
    y0 = np.clip(y0, 0, src_h - 1)
    y1 = np.clip(y1, 0, src_h - 1)

    # Retrieve values at the four neighbors
    Ia = source[y0, x0]
    Ib = source[y1, x0]
    Ic = source[y0, x1]
    Id = source[y1, x1]

    # Calculate the weights
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (x1 - src_x) * (src_y - y0)
    wc = (src_x - x0) * (y1 - src_y)
    wd = (src_x - x0) * (src_y - y0)

    # Compute the interpolated values
    dst = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return dst

def write_tiff_file(fine_dose_array):
    """
    Parameters
    ----------
    finer_dose_array : uint16 numpy array

    Returns
    -------
    16 bit tiff-file

    """
    # convert to 16 bit
    pixel_spacing_mm = 0.1
    pixel_spacing_inch = pixel_spacing_mm / 25.4
    resolution_ppi = 1 / pixel_spacing_inch
    normalize_dose = np.max(fine_dose_array)
    normalized_dose_array = fine_dose_array / normalize_dose
    dose_array_16bit = (normalized_dose_array * (2**16 - 1)).astype(np.uint16)
    dose_array_16bit_inv = (2**16 - 1) - dose_array_16bit
    imwrite("original.tif", dose_array_16bit_inv, resolution=(resolution_ppi, resolution_ppi))
    return dose_array_16bit_inv
    
# Usage
if __name__ == "__main__":
    file = "original.mcc"
    doses = read_1600xdr_mcc(file)
    dose_array = bilinear_interpolate_nan(doses)
    fine_dose_array = bilinear_interpolate(dose_array, 10)
    dose_array_16bit = write_tiff_file(fine_dose_array)
