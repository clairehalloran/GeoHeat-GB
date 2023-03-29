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
import geopandas as gpd
import numpy as np
# import progressbar as pgb
import xarray as xr
import pandas as pd
import rioxarray as rio
import rasterio
logger = logging.getLogger(__name__)
#%%
# define function for Watson et al. 2021 heating demand applicable to GB
def convert_watson_heat_demand(ds, source, num_dwellings = 1):
    watson_filepath = f'data/{source}_source_heat_profiles.csv'
    T_hourly = ds['temperature']
    #!!! note this isn't population-weighted right now
    # calculate population-weighted temperature as part of clustering
    T = T_hourly.resample(time="1D").mean(dim="time")-273.15 # convert to Celsius
    T.load()
    
    #read in watson data
    watson_data = pd.read_csv(watson_filepath, index_col = 0, parse_dates = [0])
    #add heat demand profiles if mean daily temp is more extreme
    watson_data['temp>19.5C'] = watson_data['temp16.5C to 19.5C'] #repeat hottest
    watson_data['temp<-4.5C'] = watson_data['temp-4.5C to -1.5C'] #repeat coldest
    watson_data = watson_data.resample('1H').max()
    time_resolution_watson_hour = 0.5 # resolution of input data

    #find which normalized profile to use for each day    
    def _heating_demand_profile(daily_temp_array):
        # takes an array of daily temperatures and returns a half-hourly array of heating demand
        heat_demand_list = []
        for i in range(len(daily_temp_array)):
            temp = daily_temp_array[i] 
            if temp > 19.5:
                col = 'temp>19.5C'
                # print('Hotter than Watson et al 2021 accounted for')
            elif temp > 16.5:
                col  = 'temp16.5C to 19.5C'
            elif temp > 13.5:
                col = 'temp13.5C to 16.5C'
            elif temp > 10.5:
                col = 'temp10.5C to 13.5C'
            elif temp > 7.5:
                col = 'temp7.5C to 10.5C'
            elif temp > 4.5:
                col = 'temp4.5C to 7.5C'
            elif temp > 1.5:
                col = 'temp1.5C to 4.5C'
            elif temp > -1.5:
                col = 'temp-1.5C to 1.5C'
            elif temp > -4.5:
                col = 'temp-4.5C to -1.5C'
            else: 
                col = 'temp<-4.5C'
                # print('Colder than Watson et al 2021 accounted for')
                 
            daily_profile_norm = watson_data[col].values
            
            # get total daily heat demand 
            # linear relationship between temperature and kWh heat demand per dwelling
            # from Watson based on 75% air source 25% ground source heat pumps
            # they observe that roughly similar numbers of households in each heating pattern
            # for ASHP and GSHP, so assume that mix is representative
            if temp < 14.3:
                gradient = -5.88
                y_int = 97.2
            else:
                gradient = -1.11
                y_int = 30
                
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

def heat_demand_watson(cutout, source, **params):
    """
    Creates GB heat demand profile for an average dwelling based on Watson et al. 2021 and input temperature data
    
    Input temperature profile should be
        -at least daily time resolution
        -population weighted
        -in deg K not deg C
        -in a pandas dataframe with index as timestamp
        
    Output profile is kW heat demand at HALF HOURLY time resolution (kW/HH)
    If want less fine (e.g. hourly), can resample and ensure adjust units appropriately
    
    From Watson et al 2021 - update parameters if use different mix
    75% ASHP 25% GSHP
    For T(eff)<14.3 gradient -5.88 yint 97.2
    For T(eff)>14.3 gradient -1.11 yint 30
    
    num_dwellings = number of residential households so can scale up to right level of total demand
    default = 1 household
    Watson et al use num_dwellings = 25.8 * 1e6 for 2010
    """
    return cutout.convert_and_aggregate(
        convert_func = convert_watson_heat_demand,
        source = source,
        **params,
        )

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
#!!! check household size           
households = population_match.sel(band=1)/2.4
# change large negative values to NaN-- may need to change to 0
households = households.where(households>0.)
households = households.fillna(0.)
# set share of ASHP and GSHP across whole of UK-- if it sums to 1, 100% penetration
share_air = 0.75
share_ground = 0.25

households_air = households * share_air
households_ground = households * share_ground

regions = gpd.read_file('resources/regions_onshore_elec_s_39.geojson')
regions = regions.set_index("name").rename_axis("bus")
buses = regions.index

#%% ASHP vs GSHP
ASHP_heating_demand, units = heat_demand_watson(cutout,
                                            'air',
                                            layout = households_air,
                                            index = buses,
                                            shapes = regions,
                                            per_unit=False,
                                            return_capacity=True,
                                            )
# ASHP_heating_demand.sel(bus='6005').plot() # I think this is the London bus

GSHP_heating_demand, units = heat_demand_watson(cutout,
                                            'ground',
                                            layout = households_ground,
                                            index=buses,
                                            shapes = regions,
                                            per_unit=False,
                                            return_capacity=True,
                                            )
# GSHP_heating_demand.sel(bus='6005').plot() # I think this is the London bus

#%% outputs: heating demand at each bus
# save to dataset as netcdf

ASHP_heating_demand = ASHP_heating_demand.rename('demand')

ASHP_heating_demand.to_netcdf('resources/load_air_source_heating.nc')

ASHP_heating_demand = GSHP_heating_demand.rename('demand')

ASHP_heating_demand.to_netcdf('resources/load_ground_source_heating.nc')

#%% example code for implementing this with snakemake from build_renewable_profiles.py
# if __name__ == "__main__":
#     if "snakemake" not in globals():
#         from _helpers import mock_snakemake

#         snakemake = mock_snakemake("build_renewable_profiles", technology="solar")
#     configure_logging(snakemake)
#     pgb.streams.wrap_stderr()

#     nprocesses = int(snakemake.threads)
#     noprogress = not snakemake.config["atlite"].get("show_progress", False)
#     config = snakemake.config["renewable"][snakemake.wildcards.technology]
#     resource = config["resource"]  # pv panel config / wind turbine config
#     correction_factor = config.get("correction_factor", 1.0)
#     capacity_per_sqkm = config["capacity_per_sqkm"]
#     p_nom_max_meth = config.get("potential", "conservative")

#     if isinstance(config.get("corine", {}), list):
#         config["corine"] = {"grid_codes": config["corine"]}

#     if correction_factor != 1.0:
#         logger.info(f"correction_factor is set as {correction_factor}")

#     cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
#     client = Client(cluster, asynchronous=True)

#     cutout = atlite.Cutout(snakemake.input["cutout"])
#     regions = gpd.read_file(snakemake.input.regions)
#     assert not regions.empty, (
#         f"List of regions in {snakemake.input.regions} is empty, please "
#         "disable the corresponding renewable technology"
#     )
#     # do not pull up, set_index does not work if geo dataframe is empty
#     regions = regions.set_index("name").rename_axis("bus")
#     buses = regions.index

#     res = config.get("excluder_resolution", 100)
#     excluder = atlite.ExclusionContainer(crs=3035, res=res)

#     if config["natura"]:
#         excluder.add_raster(snakemake.input.natura, nodata=0, allow_no_overlap=True)

#     corine = config.get("corine", {})
#     if "grid_codes" in corine:
#         codes = corine["grid_codes"]
#         excluder.add_raster(snakemake.input.corine, codes=codes, invert=True, crs=3035)
#     if corine.get("distance", 0.0) > 0.0:
#         codes = corine["distance_grid_codes"]
#         buffer = corine["distance"]
#         excluder.add_raster(
#             snakemake.input.corine, codes=codes, buffer=buffer, crs=3035
#         )

#     # if "ship_threshold" in config:
#     #     shipping_threshold = (
#     #         config["ship_threshold"] * 8760 * 6
#     #     )  # approximation because 6 years of data which is hourly collected
#     #     func = functools.partial(np.less, shipping_threshold)
#     #     excluder.add_raster(
#     #         snakemake.input.ship_density, codes=func, crs=4326, allow_no_overlap=True
#     #     )

#     # if "max_depth" in config:
#     #     # lambda not supported for atlite + multiprocessing
#     #     # use named function np.greater with partially frozen argument instead
#     #     # and exclude areas where: -max_depth > grid cell depth
#     #     func = functools.partial(np.greater, -config["max_depth"])
#     #     excluder.add_raster(snakemake.input.gebco, codes=func, crs=4326, nodata=-1000)

#     # if "min_shore_distance" in config:
#     #     buffer = config["min_shore_distance"]
#     #     excluder.add_geometry(snakemake.input.country_shapes, buffer=buffer)

#     # if "max_shore_distance" in config:
#     #     buffer = config["max_shore_distance"]
#     #     excluder.add_geometry(
#     #         snakemake.input.country_shapes, buffer=buffer, invert=True
#     #     )

#     kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
#     if noprogress:
#         logger.info("Calculate landuse availabilities...")
#         start = time.time()
#         availability = cutout.availabilitymatrix(regions, excluder, **kwargs)
#         duration = time.time() - start
#         logger.info(f"Completed availability calculation ({duration:2.2f}s)")
#     else:
#         availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

#     area = cutout.grid.to_crs(3035).area / 1e6
#     area = xr.DataArray(
#         area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
#     )

#     potential = capacity_per_sqkm * availability.sum("bus") * area
#     func = getattr(cutout, resource.pop("method"))
#     #!!! here I'll need to define my own function(s), rather than pulling from atlite
#     resource["dask_kwargs"] = {"scheduler": client}
#     capacity_factor = correction_factor * func(capacity_factor=True, **resource)
#     layout = capacity_factor * area * capacity_per_sqkm
#     #!!! heating profile here
#     profile, capacities = func(
#         matrix=availability.stack(spatial=["y", "x"]),
#         layout=layout,
#         index=buses,
#         per_unit=True,
#         return_capacity=True,
#         **resource,
#     )

#     logger.info(f"Calculating maximal capacity per bus (method '{p_nom_max_meth}')")
#     if p_nom_max_meth == "simple":
#         p_nom_max = capacity_per_sqkm * availability @ area
#     elif p_nom_max_meth == "conservative":
#         max_cap_factor = capacity_factor.where(availability != 0).max(["x", "y"])
#         p_nom_max = capacities / max_cap_factor
#     else:
#         raise AssertionError(
#             'Config key `potential` should be one of "simple" '
#             f'(default) or "conservative", not "{p_nom_max_meth}"'
#         )

#     logger.info("Calculate average distances.")
#     layoutmatrix = (layout * availability).stack(spatial=["y", "x"])

#     coords = cutout.grid[["x", "y"]]
#     bus_coords = regions[["x", "y"]]

#     average_distance = []
#     centre_of_mass = []
#     for bus in buses:
#         row = layoutmatrix.sel(bus=bus).data
#         nz_b = row != 0
#         row = row[nz_b]
#         co = coords[nz_b]
#         distances = haversine(bus_coords.loc[bus], co)
#         average_distance.append((distances * (row / row.sum())).sum())
#         centre_of_mass.append(co.values.T @ (row / row.sum()))

#     average_distance = xr.DataArray(average_distance, [buses])
#     centre_of_mass = xr.DataArray(centre_of_mass, [buses, ("spatial", ["x", "y"])])

#     ds = xr.merge(
#         [
#             (correction_factor * profile).rename("profile"),
#             capacities.rename("weight"),
#             p_nom_max.rename("p_nom_max"),
#             potential.rename("potential"),
#             average_distance.rename("average_distance"),
#         ]
#     )

#     if snakemake.wildcards.technology.startswith("offwind"):
#         logger.info("Calculate underwater fraction of connections.")
#         offshore_shape = gpd.read_file(snakemake.input["offshore_shapes"]).unary_union
#         underwater_fraction = []
#         for bus in buses:
#             p = centre_of_mass.sel(bus=bus).data
#             line = LineString([p, regions.loc[bus, ["x", "y"]]])
#             frac = line.intersection(offshore_shape).length / line.length
#             underwater_fraction.append(frac)

#         ds["underwater_fraction"] = xr.DataArray(underwater_fraction, [buses])

#     # select only buses with some capacity and minimal capacity factor
#     ds = ds.sel(
#         bus=(
#             (ds["profile"].mean("time") > config.get("min_p_max_pu", 0.0))
#             & (ds["p_nom_max"] > config.get("min_p_nom_max", 0.0))
#         )
#     )

#     if "clip_p_max_pu" in config:
#         min_p_max_pu = config["clip_p_max_pu"]
#         ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

#     ds.to_netcdf(snakemake.output.profile)
