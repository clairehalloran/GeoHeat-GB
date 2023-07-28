# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:02:29 2023

@author: Claire Halloran, University of Oxford

Calculates the coefficient of performance (COP) time series of 
air-source and ground-source heat pumps at each clustered network node.

Based on build_cop_profiles in PyPSA-Eur v. 0.8.0
"""


"""
Build coefficient of performance (COP) time series for air- or ground-sourced
heat pumps.
The COP is a function of the temperature difference between
source and sink.
The quadratic regression used is based on Staffell et al. (2012)
https://doi.org/10.1039/C2EE22653G.
"""

import xarray as xr


def coefficient_of_performance(delta_T, source="air"):
    if source == "air":
        return 6.81 - 0.121 * delta_T + 0.000630 * delta_T**2
    elif source == "ground":
        return 8.77 - 0.150 * delta_T + 0.000734 * delta_T**2
    else:
        raise NotImplementedError("'source' must be one of  ['air', 'ground']")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_cop_profiles",
            simpl="",
            clusters=48,
        )
    for source in snakemake.config['heating']['heat_sources']:
        source_T = xr.open_dataarray(snakemake.input[f"temp_{source}"])

        delta_T = snakemake.config['heating']["heat_pump_sink_T"] - source_T

        cop = coefficient_of_performance(delta_T, source)
        cop = cop.rename('cop')

        cop.to_netcdf(snakemake.output[f"cop_{source}"])