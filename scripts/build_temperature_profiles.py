# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:21:03 2023

@author: Claire Halloran, University of Oxford

This script builds hourly air and soil temperature profiles 
for each clustered network region. Inspired by build_temperature_profiles.py
in PyPSA-Eur v. 0.8.0.
"""

import logging
import atlite
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import numpy as np
import pandas as pd
import progressbar as pgb
import rioxarray as rio
import rasterio
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_temperature_profiles",
            simpl='',
            clusters=48
            )
    configure_logging(snakemake)
    pgb.streams.wrap_stderr()

    nprocesses = int(snakemake.threads)
    noprogress = not snakemake.config["atlite"].get("show_progress", False)
    config = snakemake.config["heating"][snakemake.wildcards.source]
    resource = config["source"]  # pv panel config / wind turbine config

    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)
    
    time = pd.date_range(freq="h", **snakemake.config["snapshots"])
    cutout = atlite.Cutout(snakemake.input.cutout).sel(time=time)

    regions = gpd.read_file(snakemake.input.regions)
    assert not regions.empty, (
        f"List of regions in {snakemake.input.regions} is empty, please "
        "disable the corresponding renewable technology"
    )
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")
    buses = regions.index
    # import population raster
    population = rio.open_rasterio(snakemake.population)
    population.rio.set_spatial_dims(x_dim='x',y_dim='y')
    cutout_rio = cutout.data
    cutout_rio = cutout_rio.rio.write_crs('EPSG:4326')
    # transform to same CRS and resolution as cutout
    population_match = population.rio.reproject_match(cutout_rio,
                                                      resampling = rasterio.enums.Resampling.sum)
    # change large negative values to nan
    population_match = population_match.where(population_match>0.)
    
   # from 0.8.0 build_temperature_profiles.py

    I = cutout.indicatormatrix(regions)

    stacked_pop = population_match.stack(spatial=("y", "x"))

    M = I.T.dot(np.diag(I.dot(stacked_pop[0])))

    nonzero_sum = M.sum(axis=0, keepdims=True)
    nonzero_sum[nonzero_sum == 0.0] = 1.0
    M_tilde = M / nonzero_sum
    M_tilde = np.nan_to_num(M_tilde,0.)
    # population-weighted average temperature
    temp_air = cutout.temperature(matrix=M_tilde.T,
                                  index=regions.index,
                                  )
    temp_air.to_netcdf(snakemake.output.temp_air)
    
    temp_ground = cutout.soil_temperature(matrix=M_tilde.T,
                                  index=regions.index,
                                  )
  
    temp_ground.to_netcdf(snakemake.output.temp_ground)
