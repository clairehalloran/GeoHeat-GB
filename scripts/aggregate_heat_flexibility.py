#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:24:08 2023

@author: Claire Halloran, University of Oxford

Aggregates heating flexibility potential to bus regions.

"""

import geopandas as gpd
import numpy as np

def aggregate_heat_flexibility(flexibility_potential, bus_regions):

    join_left_df = flexibility_potential.sjoin(bus_regions, how='left')
    
    bus_regions['Thermal time constant [h]'] = np.nan
    bus_regions['Thermal capacity [kWh/C]'] = np.nan
    bus_regions['Households'] = np.nan
    
    regions_list = join_left_df['index_right'].unique()
    
    for region in regions_list:
        region_df = join_left_df[join_left_df['index_right']==region]
        region_households = region_df['Households'].sum()
        time_constant_households = region_df[~region_df['Thermal time constant [h]'].isna()]['Households'].sum()
        # how to handle regions with NaNs?
        mean_time_constant = (region_df['Thermal time constant [h]']*region_df['Households']).sum()/time_constant_households
        mean_thermal_capacity = (region_df['Thermal capacity [kWh/C]']*region_df['Households']).sum()/region_households
        bus_regions.at[region,'Thermal time constant [h]'] = mean_time_constant
        bus_regions.at[region,'Thermal capacity [kWh/C]'] = mean_thermal_capacity
        bus_regions.at[region, 'Households'] = region_households
        if isinstance(region, str) and 'GB2' in region:
            bus_regions.at[region,'Thermal time constant [h]'] = np.nan
            bus_regions.at[region,'Thermal capacity [kWh/C]'] = np.nan
            bus_regions.at[region, 'Households'] = np.nan

    bus_regions['Thermal time constant [h]'] = bus_regions['Thermal time constant [h]'].fillna(0)
    bus_regions['Thermal capacity [kWh/C]'] = bus_regions['Thermal capacity [kWh/C]'].fillna(0)
    bus_regions['Households'] = bus_regions['Households'].fillna(0)
    
    bus_regions.dropna(inplace = True)
    bus_regions.drop('geometry', axis=1)

    return bus_regions

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "aggregate_heat_flexibility",
            simpl="",
            clusters=39,
        )
    bus_regions = gpd.read_file(snakemake.input['regions'])
    bus_regions.set_index('name',inplace=True)
    
    # import a single geojson with time constants for each LSOA and number of households
    flexibility_potential = gpd.read_file(snakemake.input['flexibility_potential'])
    flexibility_potential.set_index('index',inplace=True)
    flexibility_potential.to_crs('EPSG:4326',inplace=True)
    
    bus_flexibility = aggregate_heat_flexibility(flexibility_potential, bus_regions)
    
    bus_flexibility.to_csv(snakemake.output['aggregated_flexibility_potential'])
