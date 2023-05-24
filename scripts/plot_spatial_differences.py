# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:37:00 2023

@author: Claire Halloran, University of Oxford

Plot comparison between runs with different temperature resolutions
"""

import logging
import pypsa
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# from _helpers import (
#     aggregate_costs,
#     aggregate_p,
#     configure_logging,
#     load_network_for_plots,
# )
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
import matplotlib as mpl
from matplotlib import colors
to_rgba = mpl.colors.colorConverter.to_rgba

logger = logging.getLogger(__name__)

# het_base = r'C:\Users\clair\OneDrive - Nexus365\DPhil\Spatiotemporal heating results\20230429 39 temperature regions'
# homo_base = r'C:\Users\clair\OneDrive - Nexus365\DPhil\Spatiotemporal heating results\20230128 single temperature region'

het_base = r'C:\Users\clair\OneDrive - Nexus365\DPhil\Spatiotemporal heating results\20230506 39 temperature regions'
homo_base = r'C:\Users\clair\OneDrive - Nexus365\DPhil\Spatiotemporal heating results\20230506 single temperature region'

#%% check differences in heating demand-- map demand at peak time using local temp and national temp

homo_heat_air = xr.open_dataset(homo_base + '\heating demand\load_air_source_heating_elec_s_39.nc')
# homo_heat_air.sel(bus='GB1 18')['demand'].plot()
# homo_heat_air.sel(bus='GB1 12')['demand'].plot()

homo_cop_air = xr.open_dataset(homo_base + '\heating demand\cop_air_elec_s_39.nc')
# homo_cop_air.sel(bus='GB1 18')['cop'].plot()

homo_heat_ground = xr.open_dataset(homo_base + '\heating demand\load_ground_source_heating_elec_s_39.nc')
# homo_heat_ground.sel(bus='GB1 12')['demand'].plot()
homo_cop_ground = xr.open_dataset(homo_base + '\heating demand\cop_ground_elec_s_39.nc')
# homo_cop_ground.sel(bus='GB1 18')['cop'].plot()

# total homogeneous thermal demand

homo_total_therm_demand = homo_heat_air['demand'].sum() + homo_heat_ground['demand'].sum()

homo_heat_elec = homo_heat_air['demand']/homo_cop_air['cop'] + homo_heat_ground['demand']/homo_cop_ground['cop']
homo_heat_therm = homo_heat_air['demand'] + homo_heat_ground['demand']
# plot time series of total electric demand
# identify time when maximum demand occurs
homo_max_time = homo_heat_elec.sum(dim='bus').idxmax()

het_heat_air = xr.open_dataset(het_base + '\heating demand\load_air_source_heating_elec_s_39.nc')
# het_heat_air.sel(bus='GB1 18')['demand'].plot()
# het_heat_air.sel(bus='GB1 12')['demand'].plot()

het_cop_air = xr.open_dataset(het_base + '\heating demand\cop_air_elec_s_39.nc')
# het_cop_air.sel(bus='GB1 18')['cop'].plot()


het_heat_ground = xr.open_dataset(het_base + '\heating demand\load_ground_source_heating_elec_s_39.nc')
# het_heat_ground.sel(bus='GB1 12')['demand'].plot()
het_cop_ground = xr.open_dataset(het_base + '\heating demand\cop_ground_elec_s_39.nc')
# het_cop_ground.sel(bus='GB1 12')['cop'].plot()
het_total_therm_demand = het_heat_air['demand'].sum() + het_heat_ground['demand'].sum()

# make a map of demand at when system-wide national demand occurs


het_heat_elec = het_heat_air['demand']/het_cop_air['cop'] + het_heat_ground['demand']/het_cop_ground['cop']
het_heat_therm = het_heat_air['demand'] + het_heat_ground['demand']
# plot time series of total electric demand

# plt.figure()
# homo_heat_elec.sum(dim='bus').plot(alpha=0.5)

# het_heat_elec.sum(dim='bus').plot(alpha=0.5)
# # violin plot
# fig, ax = plt.subplots()
# ax.violinplot(homo_heat_elec.sum(dim='bus').to_pandas())
# ax.violinplot(het_heat_elec.sum(dim='bus').to_pandas())
# # duration curve for whole system
# fig, ax = plt.subplots()
# ax.plot(homo_heat_elec.sum(dim='bus').to_pandas().sort_values(ascending=False,ignore_index=True))
# ax.plot(het_heat_elec.sum(dim='bus').to_pandas().sort_values(ascending=False,ignore_index=True))

#!!! can also plot load duration curves for het vs homo
plt.figure()
# identify time when maximum demand occurs
het_max_time = het_heat_elec.sum(dim='bus').idxmax()
homo_max_time_demand = homo_heat_elec.sel(time=homo_max_time).to_pandas()

# make a map of demand at when system-wide national demand occurs

het_max_time_demand = het_heat_elec.sel(time=het_max_time).to_pandas()

regions = gpd.read_file(homo_base + r'\regions_onshore_elec_s_39.geojson')
regions = regions.set_index('name')

regions_gb = regions[regions.index.str.contains("GB1")]

# regional demand at system peak
regions_gb = regions_gb.assign(peak_demand_homo=homo_max_time_demand)
regions_gb = regions_gb.assign(peak_demand_het=het_max_time_demand)
regions_gb = regions_gb.assign(peak_demand_ratio = homo_max_time_demand/het_max_time_demand)

divnorm = colors.TwoSlopeNorm(vmin=regions_gb['peak_demand_ratio'].min(), vcenter=1., vmax=regions_gb['peak_demand_ratio'].max())
ax = regions_gb.plot('peak_demand_ratio',cmap='RdBu_r',norm = divnorm, legend=True, edgecolor='k')
ax.set_axis_off()
ax.set_title('Peak demand ratio')
plt.savefig(het_base +'/plots/peak demand ratio.png', bbox_inches = 'tight', dpi=300)

# also show difference in total heating electricity demand at each bus
plt.figure()
het_total_demand = het_heat_elec.sum(dim='time').to_pandas()
homo_total_demand = homo_heat_elec.sum(dim='time').to_pandas()

demand_ratio = homo_total_demand/het_total_demand

regions_gb = regions_gb.assign(total_demand_ratio = demand_ratio)
divnorm = colors.TwoSlopeNorm(vmin=regions_gb['total_demand_ratio'].min(), vcenter=1., vmax=regions_gb['total_demand_ratio'].max())
# divnorm = colors.TwoSlopeNorm(vmin=regions_gb['total_demand_ratio'].min(), vcenter=1.)
ax = regions_gb.plot('total_demand_ratio',cmap='RdBu_r',norm = divnorm, legend=True, edgecolor='k')

ax.set_axis_off()
ax.set_title('Total heating demand ratio')
plt.savefig(het_base+'/plots/total heating demand ratio.png',bbox_inches = 'tight', dpi=300)

# map total thermal demand in each region

# het_therm_total_demand = het_heat_elec.sum(dim='time').to_pandas()
# homo_therm_total_demand = homo_heat_elec.sum(dim='time').to_pandas()

# therm_demand_ratio = homo_therm_total_demand/het_therm_total_demand


# regions_gb = regions_gb.assign(therm_demand_ratio = therm_demand_ratio)
# divnorm = colors.TwoSlopeNorm(vmin=regions_gb['therm_demand_ratio'].min(), vcenter=1., vmax=regions_gb['therm_demand_ratio'].max())

# ax = regions_gb.plot('therm_demand_ratio',cmap='RdBu_r', norm = divnorm, legend=True, edgecolor='k')
# ax.set_axis_off()
# ax.set_title('Total thermal heating demand ratio')

# ax = regions_gb.plot('homo_therm_total_demand',cmap='RdBu_r', legend=True, edgecolor='k')


# Check differences in daily heating demand

# het_monthly_therm_demand = het_heat_elec.resample(time='1D').sum().sum(dim='bus').to_pandas()
# homo_monthly_therm_demand = homo_heat_elec.resample(time='1D').sum().sum(dim='bus').to_pandas()

# het_monthly_therm_demand.plot()
# homo_monthly_therm_demand.plot()

# (homo_monthly_therm_demand-het_monthly_therm_demand).plot()
#%% temperature comparison

# local_temp_air = xr.open_dataset(het_base + '/heating demand/temp_air_elec_s_39.nc')

# local_temp_air['temperature'].sel(bus = 'GB1 18').resample(time='1D').mean().plot()

# # try plotting mean temperature-- maybe London or another populous area is often close to breakpoint

# mean_temp = local_temp_air['temperature'].mean(dim='time').to_pandas()

# min_mean_temp_bus = local_temp_air['temperature'].mean(dim='time').idxmin(dim='bus')

# regions_gb = regions_gb.assign(mean_temp=mean_temp)

# ax = regions_gb.plot('mean_temp',cmap='RdBu_r', legend=True, edgecolor='k')

# ax.set_axis_off()

# # something weird at GB1 1 and GB1 18

# local_temp_air['temperature'].sel(bus = 'GB1 1').plot()
# local_temp_air['temperature'].sel(bus = 'GB1 18').plot()

# national_temp_air = xr.open_dataset(homo_base + '/heating demand/temp_air_elec_s_39.nc')
# national_temp_air['temperature'].sel(bus = 'GB1 18',time = '2019-03').plot.hist()
# national_temp_air['temperature'].sel(bus='GB1 5').resample(time='1D').mean().plot()

# # violin plots of daily mean temperatures? compare with GB1 14 and GB1 1
# national_temp = national_temp_air['temperature'].sel(bus='GB1 1').resample(time='1D').mean().to_pandas()
# low_temp = local_temp_air['temperature'].sel(bus='GB1 1').resample(time='1D').mean().to_pandas()
# high_temp = local_temp_air['temperature'].sel(bus='GB1 14').resample(time='1D').mean().to_pandas()

# fig, ax = plt.subplots(1,3,sharey=True)

# ax[0].violinplot(national_temp)
# ax[1].violinplot(low_temp)
# ax[2].violinplot(high_temp)


#%%% differences in network design

# central scenario

het = pypsa.Network(het_base + r"\networks\elec_s_39_ec_lv1.15_Co2L0.1-EQ0.95c.nc")

homo = pypsa.Network(homo_base + r"\networks\elec_s_39_ec_lv1.15_Co2L0.1-EQ0.95c.nc")

het_cost = het.objective/1e9 # billion euros p.a.
homo_cost = homo.objective/1e9 # billion euros p.a.

# het.plot()

# het.generators.p_nom

# het_generation_capacity = het.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()
 
# homo_generation_capacity = homo.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()

# gen_diffs = homo_generation_capacity - het_generation_capacity
# gen_pct_diffs = homo_generation_capacity/het_generation_capacity


# # try plotting generation capacity differences in each region to get an idea...
# regions_gb = regions_gb.assign(gen_diffs=gen_diffs)

# divnorm = colors.TwoSlopeNorm(vmin=regions_gb['gen_diffs'].min(), vcenter=0., vmax=regions_gb['gen_diffs'].max())
# ax = regions_gb.plot('gen_diffs',cmap='RdBu_r',norm = divnorm, legend=True, edgecolor='k')
# ax.set_axis_off()
# ax.set_title('Generation capacity difference')

# # percent difference in generation capacity-- circles on buses are probably clearer than colored regions

# regions_gb = regions_gb.assign(gen_pct_diffs=gen_pct_diffs)

# # divnorm = colors.TwoSlopeNorm(vmin=regions_gb['gen_pct_diffs'].min(), vcenter=1., vmax=regions_gb['gen_pct_diffs'].max())
# divnorm = colors.TwoSlopeNorm(vmin=0.95, vcenter=1., vmax=1.05)

# ax = regions_gb.plot('gen_pct_diffs',cmap='RdBu_r',norm = divnorm, legend=True, edgecolor='k')
# ax.set_axis_off()
# ax.set_title('Generation capacity ratio')

# # violin plot of differences
# fig,ax = plt.subplots()
# ax.violinplot(gen_pct_diffs)


# (het_generation_capacity[het_generation_capacity.index.str.contains('GB1')]-
# homo_generation_capacity[homo_generation_capacity.index.str.contains('GB1')]).sum()/homo_generation_capacity[homo_generation_capacity.index.str.contains('GB1')].sum()

# (het_generation_capacity[~het_generation_capacity.index.str.contains('GB1')]-
# homo_generation_capacity[~homo_generation_capacity.index.str.contains('GB1')])/homo_generation_capacity[~homo_generation_capacity.index.str.contains('GB1')]


# britain_gen_diff = gen_diffs[gen_diffs.index.str.contains('GB1')]
# other_gen_diff = gen_diffs[~gen_diffs.index.str.contains('GB1')]
# # plot percent differences?


#%% basic plot of network 

def plot_network(n, ax = None):
    if ax is None:
        ax = plt.gca()
    
    
    bus_sizes = pd.concat(
        (
            n.generators.query('carrier != "load"')
            .groupby(["bus", "carrier"])
            .p_nom_opt.sum(),
            n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum(),
        )
        )
    bus_size_factor = 1.3e+5
    
    line_colors = {
        "cur": "purple",
        "exp": mpl.colors.rgb2hex(to_rgba("red", 0.7), True),
    }
    
    line_widths_exp = n.lines.s_nom_opt
    line_widths_cur = n.lines.s_nom_min
    link_widths_exp = n.links.p_nom_opt
    link_widths_cur = n.links.p_nom_min
    line_colors_with_alpha = (line_widths_cur / n.lines.s_nom > 1e-3).map(
        {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
    )
    link_colors_with_alpha = (link_widths_cur / n.links.p_nom > 1e-3).map(
        {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
    )
    linewidth_factor = 2.e+3
    gb_boundaries = [-6., 2, 50, 60.5]
    
    n.plot(
            line_widths=line_widths_exp / linewidth_factor,
            link_widths=link_widths_exp / linewidth_factor,
            line_colors=line_colors["exp"],
            link_colors=line_colors["exp"],
            bus_sizes=bus_sizes/bus_size_factor,
            boundaries = gb_boundaries,
            geomap=True,
            color_geomap = {'ocean':'whitesmoke','land':'lightslategray'},
            ax=ax)
    
    n.plot(
        line_widths=line_widths_cur / linewidth_factor,
        link_widths=link_widths_cur / linewidth_factor,
        line_colors=line_colors_with_alpha,
        link_colors=link_colors_with_alpha,
            bus_sizes=bus_sizes/bus_size_factor,
            boundaries = gb_boundaries,
            geomap=True,
            color_geomap = {'ocean':'whitesmoke','land':'lightslategray'},
            ax=ax)

fig, ax = plt.subplots(1,2,
    figsize = [7,7],
    subplot_kw={"projection": ccrs.EuroPP()}
)

plot_network(homo,ax[0])
plot_network(het,ax[1])
#%% network plot based on differences in generation capacity and transmission capacity
# probably also need to change scaling factors
def map_differences(n_national, n_local, ax = None):
    if ax is None:
        ax = plt.gca()
    
    
    
    # !!! change this based on differences in generation and storage capacity
    bus_sizes = n_national.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()-\
                n_local.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()
    bus_size_factor = 2e+4

    line_widths_exp = n_national.lines.s_nom_opt-n_local.lines.s_nom_opt
    link_widths_exp = n_national.links.p_nom_opt-n_local.links.p_nom_opt
    
    linewidth_factor = 5.e+1
    gb_boundaries = [-6., 2, 50, 60.5]
    
    n_national.plot(
            line_widths=abs(line_widths_exp) / linewidth_factor,
            link_widths=abs(link_widths_exp) / linewidth_factor,
            line_colors=line_widths_exp,
            link_colors=link_widths_exp,
            line_cmap='RdBu_r',
            link_cmap='RdBu_r',
            bus_sizes=abs(bus_sizes)/bus_size_factor,
            bus_colors = bus_sizes,
            bus_cmap='RdBu_r',
            boundaries = gb_boundaries,
            geomap=True,
            color_geomap = {'ocean':'whitesmoke','land':'lightslategray'},
            ax=ax)

###

fig, ax = plt.subplots(
    figsize = [7,7],
    subplot_kw={"projection": ccrs.EuroPP()}
)

map_differences(homo, het, ax)

#%% some plot of storage capacity differences


#%%
def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        # return np.diff(ax.transData.transform([(0, 0), (1, 1)]))[0] * (
        return np.diff(ax.transData.transform([(0, 0), (1, 1)])) * (
            72.0 / fig.dpi
        )

    ellipses = []
    if not dont_resize_actively:

        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2.0 * radius * dist

        fig.canvas.mpl_connect("resize_event", update_width_height)
        ax.callbacks.connect("xlim_changed", update_width_height)
        ax.callbacks.connect("ylim_changed", update_width_height)

    def legend_circle_handler(
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        # w,h = 2.0 * orig_handle.get_radius() * axes2pt()
        w,h = orig_handle.get_radius() * axes2pt()
        e = Ellipse(
            xy=(0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
            width=w,
            height=w,
            )
        #!!!
        # center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        # w = 2.0 * orig_handle.get_radius() * axes2pt()
        # e = Ellipse(
        #     xy=center,
        #     width=w,
        #     height=w,
        # )
        ellipses.append((e, orig_handle.get_radius()))
        return e

    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

def make_legend_circles_for(sizes, scale=1.0, **kw):
    #!!! changing power
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]


def set_plot_style():
    plt.style.use(
        [
            "classic",
            "seaborn-white",
            {
                "axes.grid": False,
                "grid.linestyle": "--",
                "grid.color": "0.6",
                "hatch.color": "white",
                "patch.linewidth": 0.5,
                "font.size": 12,
                "legend.fontsize": "medium",
                "lines.linewidth": 1.5,
                "pdf.fonttype": 42,
            },
        ]
    )


def plot_map(n, opts, ax=None, attribute="p_nom"):
    if ax is None:
        ax = plt.gca()

    ## DATA
    line_colors = {
        "cur": "purple",
        "exp": mpl.colors.rgb2hex(to_rgba("red", 0.7), True),
    }
    tech_colors = opts["tech_colors"]

    if attribute == "p_nom":
        # bus_sizes = n.generators_t.p.sum().loc[n.generators.carrier == "load"].groupby(n.generators.bus).sum()
        bus_sizes = pd.concat(
            (
                n.generators.query('carrier != "load"')
                .groupby(["bus", "carrier"])
                .p_nom_opt.sum(),
                n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum(),
            )
        )
        line_widths_exp = n.lines.s_nom_opt
        line_widths_cur = n.lines.s_nom_min
        link_widths_exp = n.links.p_nom_opt
        link_widths_cur = n.links.p_nom_min
    else:
        raise "plotting of {} has not been implemented yet".format(attribute)

    line_colors_with_alpha = (line_widths_cur / n.lines.s_nom > 1e-3).map(
        {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
    )
    link_colors_with_alpha = (link_widths_cur / n.links.p_nom > 1e-3).map(
        {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
    )

    ## FORMAT
    linewidth_factor = opts["map"][attribute]["linewidth_factor"]
    bus_size_factor = opts["map"][attribute]["bus_size_factor"]

    ## PLOT
    n.plot(
        line_widths=line_widths_exp / linewidth_factor,
        link_widths=link_widths_exp / linewidth_factor,
        line_colors=line_colors["exp"],
        link_colors=line_colors["exp"],
        bus_sizes=bus_sizes / bus_size_factor,
        bus_colors=tech_colors,
        boundaries=map_boundaries,
        color_geomap=True,
        geomap=True,
        ax=ax,
    )
    n.plot(
        line_widths=line_widths_cur / linewidth_factor,
        link_widths=link_widths_cur / linewidth_factor,
        line_colors=line_colors_with_alpha,
        link_colors=link_colors_with_alpha,
        bus_sizes=0,
        boundaries=map_boundaries,
        color_geomap=True,
        geomap=True,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Rasterize basemap
    # TODO : Check if this also works with cartopy
    for c in ax.collections[:2]:
        c.set_rasterized(True)

    # LEGEND
    handles = []
    labels = []

    for s in (10, 5):
        handles.append(
            plt.Line2D(
                [0], [0], color=line_colors["exp"], linewidth=s * 1e3 / linewidth_factor
            )
        )
        labels.append("{} GW".format(s))
    l1_1 = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.66),
        frameon=False,
        labelspacing=0.8,
        # handletextpad=1.5,
        handletextpad=4.5,
        title="Transmission Exp./Exist.",
    )
    ax.add_artist(l1_1)

    handles = []
    labels = []
    for s in (10, 5):
        handles.append(
            plt.Line2D(
                [0], [0], color=line_colors["cur"], linewidth=s * 1e3 / linewidth_factor
            )
        )
        labels.append(" ")
    l1_2 = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.18, 0.66),
        frameon=False,
        labelspacing=0.8,
        handletextpad=0.5,
        title=" ",
    )
    ax.add_artist(l1_2)

    handles = make_legend_circles_for(
        #!!!
        [10, 5, 3],
        # [10e3, 5e3, 1e3],
        scale=bus_size_factor, 
        # facecolor="w",
        #!!!
        facecolor = 'None',
        edgecolor='b'
    )
    labels = ["{} GW".format(s) for s in (10, 5, 3)]
    l2 = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        labelspacing=1.7,
        title="Generation",
        handler_map=make_handler_map_to_scale_circles_as_in(ax),
    )
    ax.add_artist(l2)

    techs = (bus_sizes.index.levels[1]).intersection(
        pd.Index(opts["vre_techs"] + opts["conv_techs"] + opts["storage_techs"])
    )
    handles = []
    labels = []
    for t in techs:
        handles.append(
            plt.Line2D(
                [0], [0], color=tech_colors[t], marker="o", markersize=8, linewidth=0
            )
        )
        labels.append(opts["nice_names"].get(t, t))
    l3 = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.0),  # bbox_to_anchor=(0.72, -0.05),
        handletextpad=0.0,
        columnspacing=0.5,
        ncol=4,
        title="Technology",
    )

    return fig

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_network",
            simpl="",
            clusters="39",
            ll="v1.15",
            opts="Co2L0.1",
            attr="p_nom",
            ext="pdf",
        )
    configure_logging(snakemake)

    set_plot_style()

    config, wildcards = snakemake.config, snakemake.wildcards

    map_figsize = config["plotting"]["map"]["figsize"]
    map_boundaries = config["plotting"]["map"]["boundaries"]

    n = load_network_for_plots(
        r"C:\Users\clair\OneDrive - Nexus365\DPhil\Spatiotemporal heating results\Single temperature region\networks\elec_s_39_ec_lv1.25_Co2L0.1.nc",
        snakemake.input.tech_costs, config
    )
    
    # n= homo

    scenario_opts = wildcards.opts.split("-")

    fig, ax = plt.subplots(
        #!!! change projection here
        figsize=map_figsize, subplot_kw={"projection": ccrs.EuroPP()}
    )
    plot_map(n, config["plotting"], ax=ax, attribute=wildcards.attr)

    fig.savefig(snakemake.output.only_map, dpi=500, bbox_inches="tight")
