# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:35:18 2024

@author: hornjulian
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

class VoxelDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.root = self.read_xml_file()
        self.x_mwpc = []
        self.y_mwpc = []
        self.x_scanner = []
        self.y_scanner = []

    def read_xml_file(self):
        """Parse the content of the given XML file."""
        try:
            # Parse the XML file
            tree = ET.parse(self.file_path)
            return tree.getroot()
        except Exception as e:
            print(f"Error reading XML file: {e}")
            return None

    def extract_voxel_data(self):
        """Extract voxel data from the XML root."""
        if self.root is None:
            print("No XML root to extract data from.")
            return

        for ies in self.root.findall(".//IES"):
            for voxel in ies.findall('Voxel'):
                self.x_mwpc.append(float(voxel.get('MwpcPosX')))
                self.y_mwpc.append(float(voxel.get('MwpcPosY')))
                self.x_scanner.append(float(voxel.get('ScannerX')))
                self.y_scanner.append(float(voxel.get('ScannerY')))

    def plot_voxel_data(self):
        """Plot voxel data."""
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(self.x_scanner, self.y_scanner, cmap='viridis', s=20, alpha=0.7, edgecolors='w', linewidth=0.5)
        plt.colorbar(scatter, label='Scanner Current')
        plt.title('Scanner Current')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(self.x_mwpc, self.y_mwpc, cmap='viridis', s=20, alpha=0.7, edgecolors='w', linewidth=0.5)
        plt.colorbar(scatter, label='Mwpc Position')
        plt.title('Mwpc Positions')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def process_and_plot(self):
        """Process the XML data and plot the voxel positions."""
        self.extract_voxel_data()
        self.plot_voxel_data()
        return np.array(self.x_mwpc), np.array(self.x_scanner), np.array(self.y_mwpc), np.array(self.y_scanner)

if __name__ == "__main__":
    """ Execute when called directly """
    file_path = "r://MPA-Export_H1//autoexport//PROC_ManualSpillRequest//2024.06.23.16.24.59//MachineBeamRecord_TCU3_20240623162613-IES2.xml"
    processor = VoxelDataProcessor(file_path)
    x_mw, x_sc, y_mw, y_sc = processor.process_and_plot()
