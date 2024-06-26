#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:43:18 2024

juh
"""

from analyze_field_flatness import FlatnessAnalyzer
from conv_mcc_2_tiff import MCCProcessor

layout_file = 'layout_OD1600XDR.npy'
default_folder = "//test_data//"

processor = MCCProcessor(layout_file)
dose_image = processor.select_folder_and_process(default_folder)
