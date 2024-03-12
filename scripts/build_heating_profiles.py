# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:53:07 2023

@author: Claire Halloran, University of Oxford

Calculates for each network node the (i) thermal heat demand hourly time series based
on temperature data and (ii) the coefficient of performance (COP) hourly time series
based on temperature data for both air-source and ground-source heat pumps.

If the country is GB, calculates demand profiles based on Watson et al. 2021.
Currently heating demand for other countries is not implemented.

Relevant settings
-----------------

.. code:: yaml

    snapshots:

    atlite:
        nprocesses:

    renewable:
        {technology}:
            cutout:
            corine:
            grid_codes:
            distance:
            natura:
            max_depth:
            max_shore_distance:
            min_shore_distance:
            capacity_per_sqkm:
            correction_factor:
            potential:
            min_p_max_pu:
            clip_p_max_pu:
            resource:
    heating:
        share_air:
        share_ground:
        

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`atlite_cf`, :ref:`renewable_cf`

Inputs
------
- cutout
- UK population raster


- ``resources/natura.tiff``: confer :ref:`natura`
- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``"cutouts/" + config["renewable"][{technology}]['cutout']``: :ref:`cutout`
- ``networks/base.nc``: :ref:`base`

Outputs
-------

- ``resources/load_{source}_source_heating.nc`` with the following structure

    ===================  ==========  =========================================================
    Field                Dimensions  Description
    ===================  ==========  =========================================================
    demand [MW]          bus, time   hourly heating thermal demand for each node
    -------------------  ----------  ---------------------------------------------------------
    cop                  bus, time   hourly heat pump coefficient of performance for each node
    -------------------  ----------  ---------------------------------------------------------


Description
-----------

This script functions at two main spatial resolutions: the resolution of the
network nodes and their `Voronoi cells
<https://en.wikipedia.org/wiki/Voronoi_diagram>`_, and the resolution of the
cutout grid cells for the weather data. Typically the weather data grid is
finer than the network nodes, so we have to work out the distribution of
generators across the grid cells within each Voronoi cell. This is done by
taking account of a combination of the available land at each grid cell and the
capacity factor there.

To compute the layout of heat pumps in each node's Voronoi cell, the
total number of households in each grid cell is multiplied by the share of 
households with each type of heat pumps (ground source or air source). 

This layout is then used to compute the heating demand time series
from the weather data cutout from ``atlite`` based on Watson et al. 2021.

S. D. Watson, K. J. Lomas, and R. A. Buswell, “How will heat pumps alter 
national half-hourly heat demands? Empirical modelling based on GB field trials,” 
Energy and Buildings, vol. 238, p. 110777, 2021, doi: 10.1016/j.enbuild.2021.110777.


"""
import logging

import atlite
from dask.distributed import Client, LocalCluster
import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rio
import rasterio
logger = logging.getLogger(__name__)

# define function for Watson et al. 2021 heating demand applicable to GB
def convert_watson_heat_demand(ds, heating_profiles, source, end_use = 'combined', num_dwellings = 1):
    # watson_filepath = f'data/{source}_source_{end_use}_heat_profiles.csv'
    T_hourly = ds['temperature']
    # note this isn't population-weighted right now
    # calculate population-weighted temperature as part of clustering
    T = T_hourly.resample(time="1D").mean(dim="time")-273.15 # convert to Celsius
    T.load()
    if source != 'air' and source != 'ground':
        raise NotImplementedError(f'Source {source} has not been implemented yet, only ground- and air-source allowed.')
    #read in watson data
    if end_use == 'DHW':
        sheet = 'Normalised DHW profile'
    elif end_use == 'space' and source == 'air':
        sheet = 'Space heat 100% ASHP'
    elif end_use == 'space' and source == 'ground':
        sheet = 'Space heat 100% GSHP'
    elif end_use == 'combined' and source == 'air':
        sheet = 'Total heat 100% ASHP'
    elif end_use == 'combined' and source == 'ground':
        sheet = 'Total heat 100% GSHP'
    
    watson_data = pd.read_excel(heating_profiles, sheet, index_col = 0)
    watson_data.index = pd.to_datetime(watson_data.index, format= '%H:%M:%S')
    watson_data = watson_data.resample('1H').max()
    time_resolution_watson_hour = 0.5 # resolution of input data
    #find which normalized profile to use for each day    
    def _heating_demand_profile(daily_temp_array):
        # takes an array of daily temperatures and returns a half-hourly array of heating demand
        heat_demand_list = []
        for i in range(len(daily_temp_array)):
            temp = daily_temp_array[i] 
            if end_use != 'DHW':
                #add heat demand profiles if mean daily temp is more extreme
                watson_data['>19.5°C'] = watson_data['16.5°C to 19.5°C'] #repeat hottest
                watson_data['<-4.5°C'] = watson_data['-4.5°C to -1.5°C'] #repeat coldest
                if temp > 19.5:
                    col = '>19.5°C'
                elif temp > 16.5:
                    col  = '16.5°C to 19.5°C'
                elif temp > 13.5:
                    col = '13.5°C to 16.5°C'
                elif temp > 10.5:
                    col = '10.5°C to 13.5°C'
                elif temp > 7.5:
                    col = '7.5°C to 10.5°C'
                elif temp > 4.5:
                    col = '4.5°C to 7.5°C'
                elif temp > 1.5:
                    col = '1.5°C to 4.5°C'
                elif temp > -1.5:
                    col = '-1.5°C to 1.5°C'
                elif temp > -4.5:
                    col = '-4.5°C to -1.5°C'
                elif temp <= -4.5: 
                    col = '<-4.5°C'
                daily_profile_norm = watson_data[col].values
            elif end_use == 'DHW':
                col = 'All outdoor temperatures'
                # DHW input data not normalized
                daily_profile = watson_data[col].values
                # divide by 2 to account for half-hourly normalization to 1
                # and hourly resampling
                daily_profile_norm = daily_profile/(2*daily_profile.sum())

            # get total daily heat demand 
            # discontinuous linear relationship between temperature and kWh heat demand per dwelling
            if end_use == 'combined':
                if source == 'air':
                    if temp < 14.3:
                        gradient = -5.85
                        y_int = 96.7
                    else:
                        gradient = -1.1
                        y_int = 28.7
                elif source == 'ground':
                    if temp < 14.3:
                        gradient = -5.96
                        y_int = 98.7
                    else:
                        gradient = -1.15
                        y_int = 29.8
            elif end_use == 'space':
                if source == 'air':
                    if temp < 14.3:
                        gradient = -5.53
                        y_int = 83.8
                    else:
                        gradient = -0.77
                        y_int = 15.8
                elif source == 'ground':
                    if temp < 14.3:
                        gradient = -5.64
                        y_int = 85.8
                    else:
                        gradient = -0.82
                        y_int = 16.9
            elif end_use == 'DHW':
                gradient = -0.324
                y_int = 12.9
            else:
                raise NotImplementedError(
                    f'End-use {end_use} has not been implemented yet, only space, DHW, or combined end use allowed.')

            daily_heat_demand = temp * gradient + y_int
            
            # multiply profile by daily heat demand and 2 to convert kWh to KW on half hour basis
            day_profile = daily_heat_demand * daily_profile_norm * 1/time_resolution_watson_hour
    
            heat_demand_list.append(day_profile)
        heat_demand_profile = np.array(heat_demand_list).flatten()
        
        return heat_demand_profile
    
    heat_demand = xr.apply_ufunc(
        _heating_demand_profile,
        T,
        input_core_dims = [['time']],
        output_core_dims = [['time']],
        exclude_dims=set(('time',)),
        vectorize=True,
        dask = 'parallelized',
        output_dtypes=[T.dtype],
        dask_gufunc_kwargs = dict(
            output_sizes = dict(time=24*T.time.size),
            allow_rechunk=True
            )
        )
    heat_demand = heat_demand.rename('heat demand')
    heat_demand['time']=T_hourly.time
    heat_demand = heat_demand.clip(min=0.0)
    return heat_demand/1000 # convert from kW to MW-- units: MW per household

def heat_demand_watson(cutout, heating_profiles, source, end_use, **params):
    """
    Creates GB heat demand profile for an average dwelling based on Watson et al. 2021 and input temperature data
    
    cutout: atlite cutout
        outdoor temperature at daily or higher temporal resolution
    heating_profiles: pandas dataframe
        Watson et al. heating profiles
    source: string
        air or ground source
    end_use: string, default 'combined'
        end use, either 'space', 'DHW' for hot water, or 'combined' for both.
    
    
    Input temperature profile should be
        -at least daily time resolution
        -population weighted
        -in deg K not deg C
        -in a pandas dataframe with index as timestamp
        
    Output profile is kW heat demand at HALF HOURLY time resolution (kW/HH)
    If want less fine (e.g. hourly), can resample and ensure adjust units appropriately
        
    num_dwellings = number of residential households so can scale up to right level of total demand
    default = 1 household
    Watson et al use num_dwellings = 25.8 * 1e6 for 2010
    """
    return cutout.convert_and_aggregate(
        convert_func = convert_watson_heat_demand,
        heating_profiles = heating_profiles,
        source = source,
        end_use = end_use,
        **params,
        )

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_heating_profiles",
            simpl="",
            clusters=39,
        )
    nprocesses = int(snakemake.threads)
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    time = pd.date_range(freq="h", **snakemake.config["snapshots"])
    cutout = atlite.Cutout(snakemake.input.cutout).sel(time=time)
    share_air = snakemake.config['heating']['air']['share']
    share_ground = snakemake.config['heating']['ground']['share']

    regions = gpd.read_file(snakemake.input.regions)
    regions = regions.set_index("name").rename_axis("bus")
    buses = regions.index

    population = rio.open_rasterio(snakemake.input.population)
    population.rio.set_spatial_dims(x_dim='x',y_dim='y')
    
    cutout_rio = cutout.data
    cutout_rio = cutout_rio.rio.write_crs('EPSG:4326')
    # transform to same CRS and resolution as cutout
    population_match = population.rio.reproject_match(cutout_rio,
                                                      resampling = rasterio.enums.Resampling.sum)
    population_match = population_match.squeeze().drop_vars('band')
    population_match = population_match.where(population_match>0.)
    # population_match = population_match.fillna(0.)
    households = population_match/2.4 # England and Wales average household size
    # change large negative values to NaN-- may need to change to 0
    households = households.where(households>0.)
    households = households.fillna(0.)
    households_air = households * share_air
    households_ground = households * share_ground

    if snakemake.config['heating']['single_GB_temperature']==True:
        # calculate population-weighted national average hourly air and soil temperature
        total_population = population_match.sum(dim=['x', 'y'])
        weighted_temperature = (cutout.data['temperature'] * population_match).sum(dim=['x', 'y']) / total_population
        # use mask of population to replace temperature within Britain with average
        cutout.data['temperature'] = cutout.data['temperature'].where(population_match.isnull(),weighted_temperature)
    
    heating_profiles = pd.ExcelFile(snakemake.input.heating_profiles)
    
    ASHP_space_heating_demand = heat_demand_watson(cutout,
                                             heating_profiles,
                                                'air',
                                                'space',
                                                layout = households_air,
                                                index = buses,
                                                shapes = regions,
                                                )
    ASHP_space_heating_demand = ASHP_space_heating_demand.rename('demand')
    ASHP_space_heating_demand.to_netcdf(snakemake.output.profile_air_source_space_heating)

    ASHP_DHW_heating_demand = heat_demand_watson(cutout,
                                             heating_profiles,
                                                'air',
                                                'DHW',
                                                layout = households_air,
                                                index = buses,
                                                shapes = regions,
                                                )
    ASHP_DHW_heating_demand = ASHP_DHW_heating_demand.rename('demand')
    ASHP_DHW_heating_demand.to_netcdf(snakemake.output.profile_air_source_DHW_heating)


    GSHP_space_heating_demand = heat_demand_watson(cutout,
                                             heating_profiles,
                                                'ground',
                                                'space',
                                                layout = households_ground,
                                                index = buses,
                                                shapes = regions,
                                                )
    GSHP_space_heating_demand = GSHP_space_heating_demand.rename('demand')
    GSHP_space_heating_demand.to_netcdf(snakemake.output.profile_ground_source_space_heating)

    GSHP_DHW_heating_demand = heat_demand_watson(cutout,
                                             heating_profiles,
                                                'ground',
                                                'DHW',
                                                layout = households_ground,
                                                index = buses,
                                                shapes = regions,
                                                )
    GSHP_DHW_heating_demand = GSHP_DHW_heating_demand.rename('demand')
    GSHP_DHW_heating_demand.to_netcdf(snakemake.output.profile_ground_source_DHW_heating)