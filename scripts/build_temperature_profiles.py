# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:21:03 2023

@author: Claire Halloran, University of Oxford

This script builds hourly air and soil temperature profiles 
for each clustered network region. Inspired by build_temperature_profiles.py
in PyPSA-Eur v. 0.8.0.
"""

# import functools
import logging
# import time

import atlite
import geopandas as gpd
import numpy as np
# import progressbar as pgb
# import xarray as xr
# from _helpers import configure_logging
# from dask.distributed import Client, LocalCluster
# from pypsa.geo import haversine
# from shapely.geometry import LineString
# import pandas as pd
import rioxarray as rio
import rasterio
# from dask.distributed import Client, LocalCluster
logger = logging.getLogger(__name__)


#%% testing
# load cutout
cutout = atlite.Cutout('cutouts/europe-2019-era5.nc')

# build layout DataArray-- how many households in each area will use a heat pump?
# in future work, this can be used to reflect adoption trends and differing shares of ground vs. air source

# import population raster
population = rio.open_rasterio('data/population_layout/GB_residential_population_2011_1_km.tif')
population.rio.set_spatial_dims(x_dim='x',y_dim='y')

cutout_rio = cutout.data
cutout_rio = cutout_rio.rio.write_crs('EPSG:4326')
# transform to same CRS and resolution as cutout
population_match = population.rio.reproject_match(cutout_rio,
                                                  resampling = rasterio.enums.Resampling.sum)

# change large negative values to nan
population_match = population_match.where(population_match>0.)


regions = gpd.read_file('resources/regions_onshore_elec_s_39.geojson')
regions = regions.set_index("name").rename_axis("bus")
buses = regions.index

#%% check alignment of regions and population by plotting
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
ax = plt.subplot(projection=ccrs.PlateCarree())
population_match.plot(ax=ax)
regions.plot(ax=ax)

#%% from 0.8.0 build_temperature_profiles.py

I = cutout.indicatormatrix(regions)

stacked_pop = population_match.stack(spatial=("y", "x"))

M = I.T.dot(np.diag(I.dot(stacked_pop[0])))

nonzero_sum = M.sum(axis=0, keepdims=True)
nonzero_sum[nonzero_sum == 0.0] = 1.0
M_tilde = M / nonzero_sum
# what if I replace nan with zero
M_tilde = np.nan_to_num(M_tilde,0.)

# population-weighted average temperature
temp_air = cutout.temperature(matrix=M_tilde.T,
                              index=regions.index,
                              )

temp_air.to_netcdf('resources/temp_air_elec_s_39.nc')

temp_soil = cutout.soil_temperature(matrix=M_tilde.T,
                              index=regions.index,
                              )

temp_air.to_netcdf('resources/temp_ground_elec_s_39.nc')




# #%% build temperature
# ASHP_heating_demand, units = temperature(cutout,
#                                            'air',
#                                            layout = households_air,
#                                            index=buses,
#                                            shapes = regions,
#                                            per_unit=False,
#                                            return_capacity=True,
#                                            )
