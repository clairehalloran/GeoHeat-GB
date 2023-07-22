# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule :mod:`solve_network`.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
        solver:
            name:
            (solveroptions):

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`solving_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`
- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`solve`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op.nc``: Solved PyPSA network for optimal dispatch including optimisation results

Description
-----------
"""

import logging
from pathlib import Path

import numpy as np
import pypsa
import xarray as xr
from _helpers import configure_logging
from solve_network import prepare_network, solve_network
from add_extra_components import attach_heat_demand
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)


def set_parameters_from_optimized(n, n_optim):
    lines_typed_i = n.lines.index[n.lines.type != ""]
    n.lines.loc[lines_typed_i, "num_parallel"] = n_optim.lines["num_parallel"].reindex(
        lines_typed_i, fill_value=0.0
    )
    n.lines.loc[lines_typed_i, "s_nom"] = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
        * n.lines.num_parallel
    )

    lines_untyped_i = n.lines.index[n.lines.type == ""]
    for attr in ("s_nom", "r", "x"):
        n.lines.loc[lines_untyped_i, attr] = n_optim.lines[attr].reindex(
            lines_untyped_i, fill_value=0.0
        )
    n.lines["s_nom_extendable"] = False

    links_dc_i = n.links.index[n.links.p_nom_extendable]
    n.links.loc[links_dc_i, "p_nom"] = n_optim.links["p_nom_opt"].reindex(
        links_dc_i, fill_value=0.0
    )
    n.links.loc[links_dc_i, "p_nom_extendable"] = False

    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    n.generators.loc[gen_extend_i, "p_nom"] = n_optim.generators["p_nom_opt"].reindex(
        gen_extend_i, fill_value=0.0
    )
    n.generators.loc[gen_extend_i, "p_nom_extendable"] = False

    stor_units_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    n.storage_units.loc[stor_units_extend_i, "p_nom"] = n_optim.storage_units[
        "p_nom_opt"
    ].reindex(stor_units_extend_i, fill_value=0.0)
    n.storage_units.loc[stor_units_extend_i, "p_nom_extendable"] = False

    stor_extend_i = n.stores.index[n.stores.e_nom_extendable]
    n.stores.loc[stor_extend_i, "e_nom"] = n_optim.stores["e_nom_opt"].reindex(
        stor_extend_i, fill_value=0.0
    )
    n.stores.loc[stor_extend_i, "e_nom_extendable"] = False
    return n

def remove_heat_demand(n, sources):
    buses_AC = n.buses[n.buses.carrier== 'AC']
    buses_i = buses_AC.index
    logger.info('Removing heating buses, loads, and links.')
    for source in sources:
        n.mremove('Bus', buses_i+f'_heat_{source}')
        # remove heating load from heating buses
        n.mremove('Load', buses_i + f'_{source}_heat')
        # remove heating links with COP
        n.mremove('Link', buses_i + f' {source} heat pump')
        return n



def attach_hi_res_heat_demand(n, full_res_path, sources):
    buses_AC = n.buses[n.buses.carrier== 'AC']
    buses_i = buses_AC.index
    bus_sub_dict = {k: buses_AC[k].values for k in ["x", "y", "country"]}

    for source in sources:
        with xr.open_dataset(full_res_path + 'load_' + source + '_source_heating_elec_s_39.nc') as ds:
            if ds.indexes["bus"].empty:
                continue
            # create heat buses
            heat_buses_i = n.madd(
                'Bus',
                #!!! seem to be having a problem with the way buses are named... should I use suffix?
                names = buses_i + f'_heat_{source}',
                carrier = 'heat',
                **bus_sub_dict
                )  
            # add heating demand to buses
            heating_demand = ds['demand'].to_pandas().T
            
            n.madd(
                'Load',
                names = buses_i,
                suffix = f'_{source}_heat',
                carrier = 'heat',
                bus = heat_buses_i,
                p_set = heating_demand,
                )
            with xr.open_dataset(full_res_path + 'cop_' + source + '_elec_s_39.nc') as cop:

                cop = cop['cop'].to_pandas()
            
                # add heat pump links to heat buses
                n.madd(
                    "Link",
                    names = buses_i,
                    suffix =f' {source} heat pump',
                    bus0 = buses_i,
                    bus1 = heat_buses_i,
                    carrier = f'{source} heat pump',
                    efficiency=cop,
                    p_nom_extendable=True,
                    capital_cost = 0 
                    )
            

# create a function to calculate full-resolution heating demand? need to redo several steps
# from snakemake workflow


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_network",
            simpl="",
            clusters="39",
            ll="v1.15",
            opts="Co2L0.1-EQ0.95c",
        )
    configure_logging(snakemake)

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(snakemake.input.unprepared)
    n_optim = pypsa.Network(snakemake.input.optimized)
    n = set_parameters_from_optimized(n, n_optim)
    
    del n_optim

    # if single GB temeprature is True, need to remove extra components for heating
    if snakemake.config['heating']['single_GB_temperature']==True:
        # need to remove extra components for heating
        heat_sources = snakemake.config['electricity']['heat_sources']
        remove_heat_demand(n, heat_sources)
        # and add new ones based on high-resolution temperature
        # !!! in the future, create a Snakemake rule that will generate full-resolution profiles
        full_res_path = '/Users/mans3904/Library/CloudStorage/OneDrive-Nexus365/DPhil/GeoHeat-GB-private/resources/full-resolution/'
        attach_hi_res_heat_demand(n, 
                                  full_res_path, 
                                  heat_sources
                                  )
    opts = snakemake.wildcards.opts.split("-")
    snakemake.config["solving"]["options"]["skip_iterations"] = False

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        # add load shedding based on UK VoLL of £6000/MWh
        snakemake.config["solving"]["options"]['load_shedding']=6.9
        n = prepare_network(n, snakemake.config["solving"]["options"])
        n = solve_network(
            n,
            snakemake.config,
            opts,
            solver_dir=tmpdir,
            solver_logfile=snakemake.log.solver,
        )
        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
