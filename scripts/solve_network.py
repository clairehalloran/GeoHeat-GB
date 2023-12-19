# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Solves linear optimal power flow for a network iteratively while updating
reactances.

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
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: ../img/results.png
        :scale: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the ``pyomo=False`` setting in the :func:`network.lopf` and  :func:`pypsa.linopf.ilopf` function.
Additionally, some extra constraints specified in :mod:`prepare_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors made through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
    
Updated to use linopy backend based on PyPSA-Eur 0.8.0, which is under an MIT License:
    
MIT License

Copyright 2017-2023 The PyPSA-Eur Authors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    

"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):
    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            load_shedding = 1e2  # Eur/kWh
        # intersect between macroeconomic and surveybased
        # willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full)
        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components(n.one_port_components):
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def add_CCL_constraints(n, config):

    agg_p_nom_minmax = pd.read_csv(
        config["electricity"]["agg_p_nom_limits"], index_col=[0, 1]
    )
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier])
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")
    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")
    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )



def add_EQ_constraints(n, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    index = mincaps.index.intersection(lhs.indexes["carrier"])
    rhs = mincaps[index].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")

def add_SAFE_constraints(n, config):
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")

def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (p_nom_vres * (-EPSILON_VRES * capacity_factor)).sum(
            "Generator"
        )

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"].rename(
        {"Generator-ext": "Generator"}
    )
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * p_max_pu[ext_i]

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")
    
def add_battery_constraints(n):
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")

def add_flexibility_constraints(n, config, o):
    # add constraints on flexibility participation
    
    float_regex = "[0-9]*\.?[0-9]+"
    participation = float(re.findall(float_regex, o)[0])
    
    buses_i = n.buses.query("carrier == 'AC'").index
    sources = config['heating']['heat_sources']
    heat_pump_capacity = config['heating']['heat_pump_capacity']/1e3 # kW to MW
    temperature_window = config['heating']['temperature_window']
    flexibility_potential = pd.read_csv(snakemake.input.flexibility_potential, index_col='name')

    for source in sources:
        source_share = config['heating'][f'{source}']['share']
        households = pd.Series(flexibility_potential['Households'].values, index = [buses_i + f" {source} building envelope"])
        # decision variable-- defined for each source
        flexible_households = n.model.add_variables(lower = 0.0, 
                                                  upper = households*snakemake.config['heating'][f'{source}']['share'], 
                                                  coords = [buses_i + f" {source} building envelope"],
                                                  name = f'Bus-{source}_flexible_households')
        
        # add a global-style constraint limiting the total number of air- and ground-source households participating in flexibility
        n.model.add_constraints(flexible_households.sum()
                              == participation * sum(households) * source_share,
                              name = f'Total-{source}-flexible-households'
                              )
        # add dropped heat load with p_set = 0
        for bus in (buses_i.values + f'_{source}_heat'):
            if bus not in n.loads_t['p_set'].columns:
                n.loads_t['p_set'][bus] = 0.
        
        # limit p_nom as the maximum of the flexible households times the peak heat demand in that area
        peak_heat_demand = n.loads_t['p_set'][buses_i.values + f'_{source}_heat'].max()
        # update index of peak heat demand to match households dischargers
        peak_heat_demand.index = ([buses_i + f" {source} building envelope"])
        
        flexibility_discharge_power = n.model.variables['Link-p_nom'].loc[buses_i.values + f" {source} building envelope discharger"]
        # replace 
        flexible_households_share = flexible_households / (households).replace(0,np.inf)
        
        links = buses_i.values + f" {source} building envelope discharger"
        buses = buses_i + f" {source} building envelope"
        discharge_power_mask = xr.DataArray(
            np.eye(peak_heat_demand.size,flexible_households_share.size,dtype=bool),
            dims = ['Link-ext','Bus'],
            coords={
                'Link-ext' : links,
                'Bus' : buses
                }
            )
        
        n.model.add_constraints(flexibility_discharge_power
                              == flexible_households_share * peak_heat_demand,
                              name = f'{source}-flexibility-discharge-limit',
                              # mask to only apply on diagonal
                               mask = discharge_power_mask
                              )
        
        # update index of peak heat demand to match load
        peak_heat_demand.index = (buses_i.values + f'_{source}_heat')
        
        # set p_nom_max as normalized heat demand for each time step 
        normalized_heat_demand = n.loads_t['p_set'][buses_i.values + f'_{source}_heat']/peak_heat_demand
        # reindex to links instead of buses
        normalized_heat_demand.columns = (buses_i.values + f" {source} building envelope discharger")
        normalized_heat_demand.fillna(0,inplace=True)
        
        n.links_t['p_max_pu'] = normalized_heat_demand
        
        
        # limit flexibility charging based on mean thermal capacity of heat pumps in Watson et al.
        
        flexibility_charge_power = n.model.variables['Link-p_nom'].loc[buses_i.values + f" {source} building envelope charger"]

        links = buses_i.values + f" {source} building envelope charger"
        buses = buses_i + f" {source} building envelope"
        charge_power_mask = xr.DataArray(
            np.eye(peak_heat_demand.size,flexible_households_share.size,dtype=bool),
            dims = ['Link-ext','Bus'],
            coords={
                'Link-ext' : links,
                'Bus' : buses
                }
            )

        n.model.add_constraints(flexibility_charge_power
                              == flexible_households * heat_pump_capacity,
                              name = f'{source}-flexibility-charge-limit',
                              # mask to only apply on diagonal
                               mask = charge_power_mask
                              )
        
        # limit e_nom based on mean thermal capacity in each region, number of flexible households, and temperature window
        thermal_capacity = flexibility_potential['Thermal capacity [kWh/C]']/1000 # convert from kWh to MWh
        thermal_capacity.index = (buses_i.values + f" {source} building envelope")
        
        flexibility_energy_capacity = n.model.variables['Store-e_nom'].loc[buses_i.values + f" {source} building envelope"]
        
        
        stores = buses_i.values + f" {source} building envelope"
        energy_mask  = xr.DataArray(
            np.eye(flexibility_energy_capacity.size,flexible_households_share.size,dtype=bool),
            dims = ['Store-ext','Bus'],
            coords={
                'Store-ext' : stores,
                'Bus' : buses
                }
            )
        
        
        n.model.add_constraints(flexibility_energy_capacity
                              == flexible_households * thermal_capacity * temperature_window,
                              name = f'{source}-flexibility-energy-limit',
                              mask = energy_mask
                              )
        

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
        if 'flex' in o:
            add_flexibility_constraints(n, config, o)
    # if no flexibility opt wildcard given, assume 0% flexibility
    if sum('flex' in o for o in opts) == 0:
       add_flexibility_constraints(n, config, 'flex0.0')
    add_battery_constraints(n)

def solve_network(n, config, opts="", **kwargs):
    solver_options = config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    if skip_iterations:
        n.optimize(
            n, solver_name=solver_name, solver_options=solver_options, **kwargs
        )
    else:
        n.optimize.optimize_transmission_expansion_iteratively(
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            **kwargs
        )
    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network", simpl="", clusters="39", ll="v1.15", opts="Co2L0.1-EQ0.95c-flex0.01"
        )
    configure_logging(snakemake)

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        n = pypsa.Network(snakemake.input[0])
        n = prepare_network(n, solve_opts)
        n = solve_network(
            n,
            snakemake.config,
            opts,
            extra_functionality=extra_functionality,
            solver_dir=tmpdir,
            solver_logfile=snakemake.log.solver,
        )
            
        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
