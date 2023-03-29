# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:35:35 2023

@author: Claire Halloran, University of Oxford

plot_heating.py

This script plots information about heating demand
"""
import xarray as xr
import geopandas as gpd

#%%
ASHP_heating_profiles = xr.open_dataset('resources/profile_air_source_heating.nc')

GSHP_heating_profiles = xr.open_dataset('resources/profile_ground_source_heating.nc')

regions = gpd.read_file('resources/regions_onshore.geojson')
regions = regions.set_index("name").rename_axis("bus")



GSHP_heating_profiles['demand'].sel(bus='6005').plot() # I think this is the London bus

GSHP_heating_profiles['cop'].sel(bus='6005').plot()

#%% calculating this here mainly for plotting and stats

GSHP_heating_elec_demand = GSHP_heating_profiles['demand']/GSHP_heating_profiles['cop']
GSHP_heating_elec_demand.sel(bus='6005').plot()
#%% plot peak heating demand in each bus region
# !!! would be cool to have an animation of heating demand at each bus throughout the year
