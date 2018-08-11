# coding: utf-8

import logging
logger = logging.getLogger(__name__)
import pandas as pd
idx = pd.IndexSlice

import numpy as np
import scipy as sp
import xarray as xr
import re

from six import iteritems
import geopandas as gpd

import pypsa
from add_electricity import load_costs, update_transmission_costs

def add_co2limit(n, Nyears=1.):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=snakemake.config['electricity']['co2limit'] * Nyears)

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    assert False, "Needs to be fixed, adds NAN"

    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def set_line_s_max_pu(n):
    # set n-1 security margin to 0.5 for 37 clusters and to 0.7 from 200 clusters
    n_clusters = len(n.buses)
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 37) / (200 - 37), 0.5, 0.7)
    n.lines['s_max_pu'] = s_max_pu

def set_line_cost_limit(n, lc, Nyears=1.):
    links_dc_b = n.links.carrier == 'DC' if not n.links.empty else pd.Series()

    lines_s_nom = n.lines.s_nom.where(
        n.lines.type == '',
        np.sqrt(3) * n.lines.num_parallel *
        n.lines.type.map(n.line_types.i_nom) *
        n.lines.bus0.map(n.buses.v_nom)
    )

    n.lines['capital_cost_lc'] = n.lines['capital_cost']
    n.links['capital_cost_lc'] = n.links['capital_cost']
    total_line_cost = ((lines_s_nom * n.lines['capital_cost_lc']).sum() +
                       n.links.loc[links_dc_b].eval('p_nom * capital_cost_lc').sum())

    if lc == 'opt':
        costs = load_costs(Nyears, snakemake.input.tech_costs,
                           snakemake.config['costs'], snakemake.config['electricity'])
        update_transmission_costs(n, costs, simple_hvdc_costs=False)
    else:
        # Either line_volume cap or cost
        n.lines['capital_cost'] = 0.
        n.links.loc[links_dc_b, 'capital_cost'] = 0.

    if lc == 'opt' or float(lc) > 1.0:
        n.lines['s_nom_min'] = lines_s_nom
        n.lines['s_nom_extendable'] = True

        n.links.loc[links_dc_b, 'p_nom_min'] = n.links.loc[links_dc_b, 'p_nom']
        n.links.loc[links_dc_b, 'p_nom_extendable'] = True

        if lc != 'opt':
            n.line_cost_limit = float(lc) * total_line_cost

    return n

def set_line_volume_limit(n, lv, Nyears=1.):
    links_dc_b = n.links.carrier == 'DC' if not n.links.empty else pd.Series()

    lines_s_nom = n.lines.s_nom.where(
        n.lines.type == '',
        np.sqrt(3) * n.lines.num_parallel *
        n.lines.type.map(n.line_types.i_nom) *
        n.lines.bus0.map(n.buses.v_nom)
    )

    total_line_volume = ((lines_s_nom * n.lines['length']).sum() +
                         n.links.loc[links_dc_b].eval('p_nom * length').sum())

    if lv == 'opt':
        costs = load_costs(Nyears, snakemake.input.tech_costs,
                           snakemake.config['costs'], snakemake.config['electricity'])
        update_transmission_costs(n, costs, simple_hvdc_costs=True)
    else:
        # Either line_volume cap or cost
        n.lines['capital_cost'] = 0.
        n.links.loc[links_dc_b, 'capital_cost'] = 0.

    if lv == 'opt' or float(lv) > 1.0:
        n.lines['s_nom_min'] = lines_s_nom
        n.lines['s_nom_extendable'] = True

        n.links.loc[links_dc_b, 'p_nom_min'] = n.links.loc[links_dc_b, 'p_nom']
        n.links.loc[links_dc_b, 'p_nom_extendable'] = True

        if lv != 'opt':
            n.line_volume_limit = float(lv) * total_line_volume

    return n

def average_every_nhours(n, offset):
    logger.info('Resampling the network to {}'.format(offset))
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='37', ll='v2', opts='Co2L-3H'),
            input=['networks/{network}_s{simpl}_{clusters}.nc'],
            output=['networks/{network}_s{simpl}_{clusters}_l{ll}_{opts}.nc']
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    opts = snakemake.wildcards.opts.split('-')

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.sum()/8760.

    set_line_s_max_pu(n)

    for o in opts:
        m = re.match(r'^\d+h$', o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
    else:
        logger.info("No resampling")

    if 'Co2L' in opts:
        add_co2limit(n, Nyears)
        # add_emission_prices(n, exclude_co2=True)

    # if 'Ep' in opts:
    #     add_emission_prices(n)

    # ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    # if ll_type == 'v':
    #     set_line_volume_limit(n, factor, Nyears)
    # elif ll_type == 'c':
    #     set_line_cost_limit(n, factor, Nyears)

    solve_opts = snakemake.config['solving']['options']

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               marginal_cost=1.0e5,
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e6)

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            t.df['capital_cost'] += (1e-1 + 2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    n.generators['class'] = n.generators['carrier']
    for carrier in n.generators.carrier.unique():
        ind = n.generators.index[n.generators['carrier'] == carrier]
        ind_t = n.generators_t.p_max_pu.columns.intersection(ind)
        if len(ind_t) != 0 and len(ind) != len(ind_t):
            n.generators.loc[ind_t, 'class'] = n.generators.loc[ind_t, 'carrier'] + '_t'

    n.storage_units['class'] = n.storage_units['carrier']
    for carrier in n.storage_units.carrier.unique():
        ind = n.storage_units.index[n.storage_units['carrier'] == carrier]
        ind_t = n.storage_units_t.inflow.columns.intersection(ind)
        if len(ind_t) != 0 and len(ind) != len(ind_t):
            n.storage_units.loc[ind_t, 'class'] = n.storage_units.loc[ind_t, 'carrier'] + '_t'

    n.export_to_netcdf(snakemake.output[0])
