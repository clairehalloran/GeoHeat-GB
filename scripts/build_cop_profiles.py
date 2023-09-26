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
The quadratic regression used is based on Ruhnau et al. (2019)
http://dx.doi.org/10.1038/s41597-019-0199-y
"""

import xarray as xr


def coefficient_of_performance(delta_T, source="air"):
    if source == "air":
        return 6.08 - 0.09 * delta_T + 0.0005 * delta_T**2
    elif source == "ground":
        return 10.29 - 0.21 * delta_T + 0.0012 * delta_T**2
    else:
        raise NotImplementedError("'source' must be one of  ['air', 'ground']")

def calculate_sink_T(amb_T, sink_type = 'radiator'):
    if sink_type == 'radiator':
        return 40. - 1.0 * amb_T
    elif sink_type =='floor':
        return 30 - 0.5 * amb_T
    else:
        raise NotImplementedError("'sink_type' must be one of ['radiator', 'floor']")
    
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_cop_profiles",
            simpl="",
            clusters=48,
        )

    for source in ["air", "ground"]:
        source_T = xr.open_dataarray(snakemake.input[f"temp_{source}"])
        amb_T = xr.open_dataarray(snakemake.input["temp_air"])
        sink_T = calculate_sink_T(amb_T)
        # minimum delta T of 15 C for warm 
        delta_T = xr.where(sink_T - source_T > 15.0, sink_T - source_T, 15.0)
        cop = coefficient_of_performance(delta_T, source)
        cop = cop.rename('cop')

        cop.to_netcdf(snakemake.output[f"cop_{source}"])