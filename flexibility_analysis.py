#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:26:09 2023

@author: Claire Halloran, University of Oxford

Analyze flexible network results
"""

import pypsa


results_folder = '/Users/mans3904/Library/CloudStorage/OneDrive-Nexus365/DPhil/GeoHeat-GB-private/results/networks/'

no_flex = pypsa.Network(results_folder + 'elec_s_39_ec_flex0.0_lv1.15_Co2L0.1-EQ0.95c.nc')
flex_1 = pypsa.Network(results_folder + 'elec_s_39_ec_flex0.01_lv1.15_Co2L0.1-EQ0.95c.nc')
flex_2 = pypsa.Network(results_folder + 'elec_s_39_ec_flex0.02_lv1.15_Co2L0.1-EQ0.95c.nc')
flex_3 = pypsa.Network(results_folder + 'elec_s_39_ec_flex0.03_lv1.15_Co2L0.1-EQ0.95c.nc')
flex_4 = pypsa.Network(results_folder + 'elec_s_39_ec_flex0.04_lv1.15_Co2L0.1-EQ0.95c.nc')
flex_5 = pypsa.Network(results_folder + 'elec_s_39_ec_flex0.05_lv1.15_Co2L0.1-EQ0.95c.nc')