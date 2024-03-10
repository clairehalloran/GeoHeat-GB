# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging
from add_electricity import (
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    load_costs,
)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def attach_storageunits(n, costs, elec_opts):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    max_hours = elec_opts["max_hours"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        roundtrip_correction = 0.5 if carrier == "battery" else 1

        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_store[carrier], "efficiency"]
            ** roundtrip_correction,
            efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"]
            ** roundtrip_correction,
            max_hours=max_hours[carrier],
            cyclic_state_of_charge=True,
        )


def attach_stores(n, costs, elec_opts):
    carriers = elec_opts["extendable_carriers"]["Store"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Electrolysis",
            bus0=buses_i,
            bus1=h2_buses_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Fuel Cell",
            bus0=h2_buses_i,
            bus1=buses_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            # NB: fixed cost is per MWel
            capital_cost=costs.at["fuel cell", "capital_cost"]
            * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
        )

    if "battery" in carriers:
        b_buses_i = n.madd(
            "Bus", buses_i + " battery", carrier="battery", **bus_sub_dict
        )

        n.madd(
            "Store",
            b_buses_i,
            bus=b_buses_i,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["battery storage", "capital_cost"],
            marginal_cost=costs.at["battery", "marginal_cost"],
        )

        n.madd(
            "Link",
            b_buses_i + " charger",
            bus0=buses_i,
            bus1=b_buses_i,
            carrier="battery charger",
            # the efficiencies are "round trip efficiencies"
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

        n.madd(
            "Link",
            b_buses_i + " discharger",
            bus0=b_buses_i,
            bus1=buses_i,
            carrier="battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

def add_heat(n, heat_profiles, cop_profiles, flexibility_potential, heating_config):
    logger.info("Add heat sector")
    
    heat_sources = heating_config['heat_sources']
    heat_pump_capacity = heating_config['heat_pump_capacity']/1e3 # kW to MW

    buses_AC = n.buses[n.buses.carrier== 'AC']
    buses_i = buses_AC.index
    bus_sub_dict = {k: buses_AC[k].values for k in ["x", "y", "country"]}
    
    for source in heat_sources:
        source_share = heating_config[f'{source}']['share']
        with xr.open_dataset(getattr(heat_profiles, 'profile_' + source + '_source_heating')) as ds:
            if ds.indexes["bus"].empty:
                continue
            # create heat buses
            heat_buses_i = n.madd(
                'Bus',
                names = buses_i + f'_heat_{source}',
                carrier = 'heat',
                **bus_sub_dict
                )  
            # add heating demand to buses
            heating_demand = ds['demand'].to_pandas().T
            # remove small heating demands outside GB
            for column in heating_demand.columns:
                if 'GB1' not in column:
                    heating_demand[column] = 0.
            n.madd(
                'Load',
                names = buses_i,
                suffix = f'_{source}_heat',
                carrier = 'heat',
                bus = heat_buses_i,
                p_set = heating_demand,
                )
            with xr.open_dataset(getattr(cop_profiles, 'profile_' + source + '_cop')) as cop:

                cop = cop['cop'].to_pandas()
            
                households = flexibility_potential['Households']
                # add heat pump links to heat buses
                n.madd(
                    "Link",
                    names = buses_i,
                    suffix =f' {source} heat pump',
                    # bus 0 and 1 flipped to set p_nom in terms of thermal capacity
                    bus0 = heat_buses_i,
                    bus1 = buses_i,
                    p_max_pu = 0,
                    p_min_pu=-1,
                    carrier = f'{source} heat pump',
                    efficiency=1/cop,
                    p_nom = source_share * households * heat_pump_capacity,
                    capital_cost = 0 
                    )
            # add flexibility from building envelope thermal inertia
            n.add("Carrier", source + " building envelope")
            
            n.madd(
                "Bus",
                names = buses_i + f" {source} building envelope",
                location = buses_i,
                carrier = source + " building envelope",
                unit="MWh_th",
            )
            
            n.madd(
                "Link",
                buses_i + f" {source} building envelope charger",
                bus0 = heat_buses_i,
                bus1 = buses_i + f" {source} building envelope",
                carrier=source + " building envelope charger",
                p_nom_extendable=True
                )

            n.madd(
                "Link",
                buses_i + f" {source} building envelope discharger",
                bus0 = buses_i + f" {source} building envelope",
                bus1 = heat_buses_i,
                carrier = source + " building envelope discharger",
                p_nom_extendable=True,
            )
            
            tes_time_constant = flexibility_potential['Thermal time constant [h]']
            tes_time_constant.index = buses_i + f" {source} building envelope"
            
            n.madd(
                "Store",
                buses_i + f" {source} building envelope",
                bus = buses_i + f" {source} building envelope",
                e_cyclic=True,
                e_nom_extendable=True,
                carrier=source + " building envelope",
                standing_loss=1 - np.exp(-1 / tes_time_constant),
            )

def attach_hydrogen_pipelines(n, costs, elec_opts):
    ext_carriers = elec_opts["extendable_carriers"]
    as_stores = ext_carriers.get("Store", [])

    if "H2 pipeline" not in ext_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]]
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.madd(
        "Link",
        h2_links.index,
        bus0=h2_links.bus0.values + " H2",
        bus1=h2_links.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * h2_links.length,
        efficiency=costs.at["H2 pipeline", "efficiency"],
        carrier="H2 pipeline",
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_extra_components", simpl="", clusters=39)
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config["electricity"]

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs, snakemake.config["costs"], elec_config, Nyears
    )

    attach_storageunits(n, costs, elec_config)
    attach_stores(n, costs, elec_config)
    
    heating_config = snakemake.config['heating']

    flexibility_potential = pd.read_csv(snakemake.input.flexibility_potential, index_col='name')
    add_heat(
        n,
        snakemake.input,
        snakemake.input,
        flexibility_potential,
        heating_config
        )
    
    attach_hydrogen_pipelines(n, costs, elec_config)
    
    add_nice_carrier_names(n, snakemake.config)
    loads = n.loads_t.p_set
    links = n.links.p_nom
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
