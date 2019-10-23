import numpy as np
import logging
logger = logging.getLogger(__name__)

import pypsa

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger
from vresutils.snakemake import MockSnakemake


def prepare_network(n, solve_opts=None):
    if solve_opts is None:
        solve_opts = snakemake.config['solving']['options']

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components(n.one_port_components):
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            t.df['capital_cost'] += (1e-1 +
                2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    return n


#def add_opts_constraints(n, opts=None):
#    if opts is None:
#        opts = snakemake.wildcards.opts.split('-')
#
#    if 'BAU' in opts:
#        mincaps = snakemake.config['electricity']['BAU_mincapacities']
#        def bau_mincapacities_rule(model, carrier):
#            gens = n.generators.index[n.generators.p_nom_extendable & (n.generators.carrier == carrier)]
#            return sum(model.generator_p_nom[gen] for gen in gens) >= mincaps[carrier]
#        n.model.bau_mincapacities = pypsa.opt.Constraint(list(mincaps), rule=bau_mincapacities_rule)
#
#    if 'SAFE' in opts:
#        peakdemand = (1. + snakemake.config['electricity']['SAFE_reservemargin']) * n.loads_t.p_set.sum(axis=1).max()
#        conv_techs = snakemake.config['plotting']['conv_techs']
#        exist_conv_caps = n.generators.loc[n.generators.carrier.isin(conv_techs) & ~n.generators.p_nom_extendable, 'p_nom'].sum()
#        ext_gens_i = n.generators.index[n.generators.carrier.isin(conv_techs) & n.generators.p_nom_extendable]
#        n.model.safe_peakdemand = pypsa.opt.Constraint(expr=sum(n.model.generator_p_nom[gen] for gen in ext_gens_i) >= peakdemand - exist_conv_caps)
#
#    # Add constraints on the per-carrier capacity in each country
#    if 'CCL' in opts:
#        agg_p_nom_limits = snakemake.config['electricity'].get('agg_p_nom_limits')
#
#        try:
#            agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
#        except IOError:
#            logger.exception("Need to specify the path to a .csv file containing aggregate capacity limits per country in config['electricity']['agg_p_nom_limit'].")
#
#        logger.info("Adding per carrier generation capacity constraints for individual countries")
#
#        gen_country = n.generators.bus.map(n.buses.country)
#
#        def agg_p_nom_min_rule(model, country, carrier):
#            min = agg_p_nom_minmax.at[(country, carrier), 'min']
#            return ((sum(model.generator_p_nom[gen]
#                         for gen in n.generators.index[(gen_country == country) & (n.generators.carrier == carrier)])
#                    >= min)
#                    if np.isfinite(min) else pypsa.opt.Constraint.Skip)
#
#        def agg_p_nom_max_rule(model, country, carrier):
#            max = agg_p_nom_minmax.at[(country, carrier), 'max']
#            return ((sum(model.generator_p_nom[gen]
#                         for gen in n.generators.index[(gen_country == country) & (n.generators.carrier == carrier)])
#                    <= max)
#                    if np.isfinite(max) else pypsa.opt.Constraint.Skip)
#
#        n.model.agg_p_nom_min = pypsa.opt.Constraint(list(agg_p_nom_minmax.index), rule=agg_p_nom_min_rule)
#        n.model.agg_p_nom_max = pypsa.opt.Constraint(list(agg_p_nom_minmax.index), rule=agg_p_nom_max_rule)


def add_lv_constraint(n):
    line_volume = getattr(n, 'line_volume_limit', None)
    if line_volume is not None:
        n.add('GlobalConstraint', 'lv_limit',
              type='transmission_volume_expansion_limit',
              sense='<=', constant=line_volume, carrier_attribute='AC, DC')


def add_lc_constraint(n):
    line_cost = getattr(n, 'line_cost_limit', None)
    if line_cost is not None:
        n.add('GlobalConstraint', 'lv_limit',
              type='transmission_expansion_cost_limit',
              sense='<=', constant=line_cost, carrier_attribute='AC, DC')

#def add_eps_storage_constraint(n):
#    if not hasattr(n, 'epsilon'):
#        n.epsilon = 1e-5
#    fix_sus_i = n.storage_units.index[~ n.storage_units.p_nom_extendable]
#    n.model.objective.expr += sum(n.epsilon * n.model.state_of_charge[su, n.snapshots[0]] for su in fix_sus_i)


def solve_network(n, config=None, solver_log=None, opts=None, callback=None):
    if config is None:
        config = snakemake.config['solving']
#    solve_opts = config['options']

    solver_options = config['solver'].copy()
    if solver_log is None:
        solver_log = snakemake.log.solver
    solver_name = solver_options.pop('name')
    pypsa.linopf.ilopf(n, solver_name=solver_name, solver_options=solver_options)


    return n

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='',
                           clusters='45', lv='1.0', opts='Co2L-3H'),
            input=["networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"],
            output=["results/networks/s{simpl}_{clusters}_lv{lv}_{opts}.nc"],
            log=dict(solver="logs/{network}_s{simpl}_{clusters}_lv{lv}_"
                             "{opts}_solver.log",
                     python="logs/{network}_s{simpl}_{clusters}_lv{lv}_"
                             "{opts}_python.log"))


    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    with memory_logger(filename=getattr(snakemake.log, 'memory', None),
                       interval=30.) as mem:
        n = pypsa.Network(snakemake.input[0])
        add_lc_constraint(n)
        add_lv_constraint(n)
        n = prepare_network(n)
        n = solve_network(n)

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
