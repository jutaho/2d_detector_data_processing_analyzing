#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:43:18 2024

juh
"""

from analyze_field_flatness import FlatnessAnalyzer
from read_octavius1600xrd_mcc import MCCProcessor

layout_file = 'layout_octavius1600xdr.npy'
default_folder = ".//test_data//"

processor = MCCProcessor(layout_file)
dose_image = processor.process_folder(default_folder)
