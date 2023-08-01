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

import matplotlib.cm
# from _helpers import (
#     aggregate_costs,
#     aggregate_p,
#     configure_logging,
#     load_network_for_plots,
# )
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
# import matplotlib as mpl
from matplotlib import colors
to_rgba = mpl.colors.colorConverter.to_rgba

logger = logging.getLogger(__name__)
plt.rcParams['svg.fonttype'] = 'none'
het_base = '/Users/mans3904/Library/CloudStorage/OneDrive-Nexus365/DPhil/Spatiotemporal heating results/20230609 39 temperature region battery storage'
homo_base = '/Users/mans3904/Library/CloudStorage/OneDrive-Nexus365/DPhil/Spatiotemporal heating results/20230607 single temperature region battery storage'

#%% check differences in heating demand-- map demand at peak time using local temp and national temp

homo_heat_air = xr.open_dataset(homo_base + '/heating demand/load_air_source_heating_elec_s_39.nc')
# homo_heat_air.sel(bus='GB1 18')['demand'].plot()
# homo_heat_air.sel(bus='GB1 12')['demand'].plot()

homo_cop_air = xr.open_dataset(homo_base + '/heating demand/cop_air_elec_s_39.nc')
# homo_cop_air.sel(bus='GB1 18')['cop'].plot()

homo_heat_ground = xr.open_dataset(homo_base + '/heating demand/load_ground_source_heating_elec_s_39.nc')
# homo_heat_ground.sel(bus='GB1 12')['demand'].plot()
homo_cop_ground = xr.open_dataset(homo_base + '/heating demand/cop_ground_elec_s_39.nc')
# homo_cop_ground.sel(bus='GB1 18')['cop'].plot()

# total homogeneous thermal demand

homo_total_therm_demand = homo_heat_air['demand'].sum() + homo_heat_ground['demand'].sum()

homo_heat_elec = homo_heat_air['demand']/homo_cop_air['cop'] + homo_heat_ground['demand']/homo_cop_ground['cop']
homo_heat_therm = homo_heat_air['demand'] + homo_heat_ground['demand']
# plot time series of total electric demand
# identify time when maximum demand occurs
homo_max_time = homo_heat_elec.sum(dim='bus').idxmax()

het_heat_air = xr.open_dataset(het_base + '/heating demand/load_air_source_heating_elec_s_39.nc')
# het_heat_air.sel(bus='GB1 18')['demand'].plot()
# het_heat_air.sel(bus='GB1 12')['demand'].plot()

het_cop_air = xr.open_dataset(het_base + '/heating demand/cop_air_elec_s_39.nc')
# het_cop_air.sel(bus='GB1 18')['cop'].plot()


het_heat_ground = xr.open_dataset(het_base + '/heating demand/load_ground_source_heating_elec_s_39.nc')
# het_heat_ground.sel(bus='GB1 12')['demand'].plot()
het_cop_ground = xr.open_dataset(het_base + '/heating demand/cop_ground_elec_s_39.nc')
# het_cop_ground.sel(bus='GB1 12')['cop'].plot()
het_total_therm_demand = het_heat_air['demand'].sum() + het_heat_ground['demand'].sum()

# make a map of demand at when system-wide national demand occurs


het_heat_elec = het_heat_air['demand']/het_cop_air['cop'] + het_heat_ground['demand']/het_cop_ground['cop']
het_heat_therm = het_heat_air['demand'] + het_heat_ground['demand']
# plot time series of total electric demand

plt.figure()
homo_heat_elec.sum(dim='bus').plot(alpha=0.5)

het_heat_elec.sum(dim='bus').plot(alpha=0.5)

# plot time series of total electric demand on peak heating demand day

plt.figure()
(homo_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3).plot()

(het_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3).plot()

# plot of bus 7 demand


# %%

# violin plot
fig, ax = plt.subplots()
ax.violinplot(homo_heat_elec.sum(dim='bus').to_pandas())
ax.violinplot(het_heat_elec.sum(dim='bus').to_pandas())
# duration curve for whole system
fig, ax = plt.subplots()
ax.plot(homo_heat_elec.sum(dim='bus').to_pandas().sort_values(ascending=False,ignore_index=True))
ax.plot(het_heat_elec.sum(dim='bus').to_pandas().sort_values(ascending=False,ignore_index=True))

#!!! can also plot load duration curves for het vs homo
plt.figure()
# identify time when maximum demand occurs
het_max_time = het_heat_elec.sum(dim='bus').idxmax()
homo_max_time_demand = homo_heat_elec.sel(time=homo_max_time).to_pandas()

# make a map of demand at when system-wide national demand occurs

het_max_time_demand = het_heat_elec.sel(time=het_max_time).to_pandas()

regions = gpd.read_file(homo_base + '/regions_onshore_elec_s_39.geojson')
regions = regions.set_index('name')

regions_gb = regions[regions.index.str.contains("GB1")]
# regions_gb = regions_gb.to_crs(ccrs.EuroPP()) # can use this line to reproject GB into EuroPP

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

local_temp_air = xr.open_dataset(het_base + '/heating demand/temp_air_elec_s_39.nc')


# try plotting mean temperature-- maybe London or another populous area is often close to breakpoint

mean_temp = local_temp_air['temperature'].mean(dim='time').to_pandas()

# min_mean_temp_bus = local_temp_air['temperature'].mean(dim='time').idxmin(dim='bus')

regions_gb = regions_gb.assign(mean_temp=mean_temp)

ax = regions_gb.plot('mean_temp',cmap='viridis', 
                     legend=True, edgecolor='k',
                     legend_kwds={"label": "°C"})
ax.set_title('Mean temperature')

ax.set_axis_off()
plt.savefig(het_base +'/plots/temperature mean.png', bbox_inches = 'tight', dpi=300)

# standard deviation in temperature

std_temp = local_temp_air['temperature'].std(dim='time').to_pandas()

regions_gb = regions_gb.assign(std_temp=std_temp)

ax = regions_gb.plot('std_temp',cmap='viridis',
                     legend=True, edgecolor='k',
                     legend_kwds={"label": "°C"})
ax.set_title('Temperature standard deviation')
ax.set_axis_off()

national_temp_air = xr.open_dataset(homo_base + '/heating demand/temp_air_elec_s_39.nc')

national_temp_air.mean(dim='time').to_pandas()

plt.savefig(het_base +'/plots/temperature standard deviation.png', bbox_inches = 'tight', dpi=300)

# plot differences in daily mean temperature at bus 7, which contains most of London

# local_temp_air['temperature'].sel(bus = 'GB1 7').resample(time='1D').mean().plot()
# national_temp_air['temperature'].sel(bus = 'GB1 7').resample(time='1D').mean().plot()


# local_temp_air['temperature'].sel(bus = 'GB1 7').resample(time='1D').mean().plot.hist()
# national_temp_air['temperature'].sel(bus = 'GB1 7').resample(time='1D').mean().plot.hist()

# national_temp_air['temperature'].sel(bus = 'GB1 18',time = '2019-03').plot.hist()
# national_temp_air['temperature'].sel(bus='GB1 5').resample(time='1D').mean().plot()

# violin plots of daily mean temperatures? compare with GB1 14 and GB1 1
# national_temp = national_temp_air['temperature'].sel(bus='GB1 1').resample(time='1D').mean().to_pandas()
# low_temp = local_temp_air['temperature'].sel(bus='GB1 1').resample(time='1D').mean().to_pandas()
# high_temp = local_temp_air['temperature'].sel(bus='GB1 14').resample(time='1D').mean().to_pandas()

# fig, ax = plt.subplots(1,3,sharey=True)

# ax[0].violinplot(national_temp)
# ax[1].violinplot(low_temp)
# ax[2].violinplot(high_temp)




# %% temporal plot of sampled heaitng demand
import rioxarray as rio

# calculate households in each region using the fact that every household has the same peak based on national temp
# total number of households
population_match = rio.open_rasterio(het_base+'/population_match.tif')
households = population_match.sum().data/2.4
# homogeneous peak per household
peak = homo_heat_elec.sum(dim='bus').data.max() #MW
peak_time = homo_heat_elec.sum(dim='bus').idxmax()
ADMD = peak/households # convert to kW
# households per region of interest
regions_gb['households'] = regions_gb['peak_demand_homo']/ADMD

# Underestimated peak, overestimated total demand
plt.figure(figsize=(9,3))
(homo_heat_elec.sel(bus = 'GB1 7',
                    time = slice('2019-01-29','2019-02-02')
                    )/regions_gb.loc['GB1 7','households']*1e3).plot(linestyle=':',lw=3, label = 'National')
(het_heat_elec.sel(bus = 'GB1 7',
                   time = slice('2019-01-29','2019-02-02')
                   )/regions_gb.loc['GB1 7','households']*1e3).plot(linestyle = '-',lw=3, label = 'Multi-regional')
plt.title('Bus 7')
plt.legend(fontsize='large')
plt.ylabel('Electric heat demand per household [kW]')
plt.grid(True)
plt.ylim(0.75,2.75)
plt.savefig(het_base +'/plots/bus 7 demand.png', bbox_inches = 'tight', dpi=300)

# Overestimated  peak, overestimated total demand
plt.figure(figsize=(9,3))
(homo_heat_elec.sel(bus = 'GB1 17',
                    time = slice('2019-01-29','2019-02-02')
                    )/regions_gb.loc['GB1 17','households']*1e3).plot(linestyle=':',lw=3,label='National')
(het_heat_elec.sel(bus = 'GB1 17',
                   time = slice('2019-01-29','2019-02-02')
                   )/regions_gb.loc['GB1 17','households']*1e3).plot(linestyle='-',lw=3, label = 'Multi-regional')
plt.title('Bus 17')
plt.legend(fontsize='large')
plt.ylabel('Electric heat demand per household [kW]')
plt.grid(True)
plt.ylim(0.75,2.75)
plt.savefig(het_base +'/plots/bus 17 demand.png', bbox_inches = 'tight', dpi=300)

# Underestimated peak, underestimated total demand:
plt.figure(figsize=(9,3))
(homo_heat_elec.sel(bus = 'GB1 19',
                    time = slice('2019-01-29','2019-02-02')
                    )/regions_gb.loc['GB1 19','households']*1e3).plot(linestyle=':',lw=3, label = 'National')
(het_heat_elec.sel(bus = 'GB1 19',
                   time = slice('2019-01-29','2019-02-02')
                   )/regions_gb.loc['GB1 19','households']*1e3).plot(linestyle='-',lw=3, label = 'Multi-regional')
plt.title('Bus 19')
plt.legend(fontsize='large')
plt.ylabel('Electric heat demand per household [kW]')
plt.grid(True)
plt.ylim(0.75,2.75)
plt.savefig(het_base +'/plots/bus 19 demand.png', bbox_inches = 'tight', dpi=300)

# Overestimated peak, underestimated total demand
plt.figure(figsize=(9,3))
(homo_heat_elec.sel(bus = 'GB1 12',
                    time = slice('2019-01-29','2019-02-02')
                    )/regions_gb.loc['GB1 12','households']*1e3).plot(linestyle=':',lw=3, label = 'National')
(het_heat_elec.sel(bus = 'GB1 12',
                   time = slice('2019-01-29','2019-02-02')
                   )/regions_gb.loc['GB1 12','households']*1e3).plot(linestyle = '-',lw=3, label = 'Multi-regional')
plt.title('Bus 12')
plt.legend(fontsize='large')
plt.ylabel('Electric heat demand per household [kW]')
plt.grid(True)
plt.ylim(0.75,2.75)
plt.savefig(het_base +'/plots/bus 12 demand.png', bbox_inches = 'tight', dpi=300)

# map with sampled regions highlighted


regions_gb['temporal plot'] = regions_gb.index.str.contains('GB1 7|GB1 17|GB1 19|GB1 12')

ax = regions_gb.plot(column = 'temporal plot', cmap = 'Blues', ec= 'k', alpha = 0.9)
ax.set_axis_off()
plt.savefig(het_base +'/plots/bus highlights.png', bbox_inches = 'tight', dpi=300)

# %% how do I find the population per bus region? use zonal stats in QGIS

# regions_pop = gpd.read_file(het_base + '/regions_population.geojson')
# regions_pop = regions_pop.set_index('name')

# regions_gb['households'] = regions_pop['_sum']/2.4

# per-capita demand maps
# !!! can redo this with households calculated in next section
# regional demand at system peak
regions_gb['peak_demand_het_per_household'] = regions_gb['peak_demand_het']/regions_gb['households']*1e3

# divnorm = colors.TwoSlopeNorm(vmin=regions_gb['peak_demand_ratio'].min(), vcenter=1., vmax=regions_gb['peak_demand_ratio'].max())
ax = regions_gb.plot('peak_demand_het_per_household',cmap='RdBu_r', legend=True, edgecolor='k')
ax.set_axis_off()
ax.set_title('Peak demand per household')
# plt.savefig(het_base +'/plots/peak demand per household.png', bbox_inches = 'tight', dpi=300)



# also show difference in total heating electricity demand at each bus
plt.figure()
het_total_demand = het_heat_elec.sum(dim='time').to_pandas()
# homo_total_demand = homo_heat_elec.sum(dim='time').to_pandas()

regions_gb['het_total_demand']=het_total_demand

regions_gb['total heating per household'] = regions_gb['het_total_demand']/regions_gb['households']*1e3
# divnorm = colors.TwoSlopeNorm(vmin=regions_gb['total_demand_ratio'].min(), vcenter=1., vmax=regions_gb['total_demand_ratio'].max())
# divnorm = colors.TwoSlopeNorm(vmin=regions_gb['total_demand_ratio'].min(), vcenter=1.)
ax = regions_gb.plot('total heating per household',cmap='RdBu_r', legend=True, edgecolor='k')

ax.set_axis_off()
ax.set_title('Total heating demand ratio')
# plt.savefig(het_base+'/plots/total heating demand ratio.png',bbox_inches = 'tight', dpi=300)


#%%% differences in network design

# central scenario

het = pypsa.Network(het_base + "/networks/elec_s_39_ec_lv1.15_Co2L0.1-EQ0.95c.nc")

homo = pypsa.Network(homo_base + "/networks/elec_s_39_ec_lv1.15_Co2L0.1-EQ0.95c.nc")

het_cost = het.objective/1e9 # billion euros p.a.
homo_cost = homo.objective/1e9 # billion euros p.a.

# other scenarios
het_localized_transition = pypsa.Network(het_base + "/networks/elec_s_39_ec_lv1.0_Co2L0.01-EQ0.95c.nc")
homo_localized_transition= pypsa.Network(homo_base + "/networks/elec_s_39_ec_lv1.0_Co2L0.01-EQ0.95c.nc")

het_localized_transition_cost = het_localized_transition.objective/1e9
homo_localized_transition_cost = homo_localized_transition.objective/1e9

het_national_green_grid = pypsa.Network(het_base + "/networks/elec_s_39_ec_lv1.3_Co2L0.01-EQ0.95c.nc")
homo_national_green_grid= pypsa.Network(homo_base + "/networks/elec_s_39_ec_lv1.3_Co2L0.01-EQ0.95c.nc")

het_national_green_grid_cost = het_national_green_grid.objective/1e9
homo_national_green_grid_cost = homo_national_green_grid.objective/1e9

het_capacity_for_carbon = pypsa.Network(het_base + "/networks/elec_s_39_ec_lv1.3_Co2L1.0-EQ0.95c.nc")
homo_capacity_for_carbon = pypsa.Network(homo_base + "/networks/elec_s_39_ec_lv1.3_Co2L1.0-EQ0.95c.nc")

het_capacity_for_carbon_cost = het_capacity_for_carbon.objective/1e9
homo_capacity_for_carbon_cost = homo_capacity_for_carbon.objective/1e9

het_carbon_stagnation = pypsa.Network(het_base + "/networks/elec_s_39_ec_lv1.0_Co2L1.0-EQ0.95c.nc")
homo_carbon_stagnation = pypsa.Network(homo_base + "/networks/elec_s_39_ec_lv1.0_Co2L1.0-EQ0.95c.nc")

het_carbon_stagnation_cost = het_carbon_stagnation.objective/1e9
homo_carbon_stagnation_cost =homo_carbon_stagnation.objective/1e9

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

# def plot_network(n, ax = None):
#     if ax is None:
#         ax = plt.gca()
    
    
#     bus_sizes = pd.concat(
#         (
#             n.generators.query('carrier != "load"')
#             .groupby(["bus", "carrier"])
#             .p_nom_opt.sum(),
#             n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum(),
#         )
#         )
#     bus_size_factor = 1.3e+5
    
#     line_colors = {
#         "cur": "purple",
#         "exp": mpl.colors.rgb2hex(to_rgba("red", 0.7), True),
#     }
    
#     line_widths_exp = n.lines.s_nom_opt
#     line_widths_cur = n.lines.s_nom_min
#     link_widths_exp = n.links.p_nom_opt
#     link_widths_cur = n.links.p_nom_min
#     line_colors_with_alpha = (line_widths_cur / n.lines.s_nom > 1e-3).map(
#         {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
#     )
#     link_colors_with_alpha = (link_widths_cur / n.links.p_nom > 1e-3).map(
#         {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
#     )
#     linewidth_factor = 2.e+3
#     # gb_boundaries = [-6., 2, 50, 60.5]
#     gb_boundaries = [-9, 12.5, 44.5, 62]
#     n.plot(
#             line_widths=line_widths_exp / linewidth_factor,
#             link_widths=link_widths_exp / linewidth_factor,
#             line_colors=line_colors["exp"],
#             link_colors=line_colors["exp"],
#             bus_sizes=bus_sizes/bus_size_factor,
#             boundaries = gb_boundaries,
#             geomap=True,
#             color_geomap = {'ocean':'whitesmoke','land':'lightslategray'},
#             ax=ax)
    
#     n.plot(
#         line_widths=line_widths_cur / linewidth_factor,
#         link_widths=link_widths_cur / linewidth_factor,
#         line_colors=line_colors_with_alpha,
#         link_colors=link_colors_with_alpha,
#             bus_sizes=bus_sizes/bus_size_factor,
#             boundaries = gb_boundaries,
#             geomap=True,
#             color_geomap = {'ocean':'whitesmoke','land':'lightslategray'},
#             ax=ax)

# fig, ax = plt.subplots(1,2,
#     figsize = [7,7],
#     subplot_kw={"projection": ccrs.EuroPP()}
# )

# plot_network(homo,ax[0])
# plot_network(het,ax[1])
#%% network plot based on differences in generation capacity and transmission capacity
# probably also need to change scaling factors

import requests

# def make_legend_circles_for(sizes, scale=1.0, **kw):
#     return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]
# def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
#     fig = ax.get_figure()

#     def axes2pt():
#         return np.diff(ax.transData.transform([(0, 0), (1, 1)])) * (
#             72.0 / fig.dpi
#         )

#     ellipses = []
#     if not dont_resize_actively:
#         def update_width_height(event):
#             dist = axes2pt()
#             for e, radius in ellipses:
#                 e.width, e.height = 2.0 * radius * dist

#         fig.canvas.mpl_connect("resize_event", update_width_height)
#         ax.callbacks.connect("xlim_changed", update_width_height)
#         ax.callbacks.connect("ylim_changed", update_width_height)

#     def legend_circle_handler(
#         legend, orig_handle, xdescent, ydescent, width, height, fontsize
#     ):
#         w,h = orig_handle.get_radius() * axes2pt()
#         e = Ellipse(
#             xy=(0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
#             width=w,
#             height=w,
#             )
#         ellipses.append((e, orig_handle.get_radius()))
#         return e

#     return {Circle: HandlerPatch(patch_func=legend_circle_handler)}
def get_projection_from_crs(crs):
    if crs == 4326:
        # if data is in latlon system, return default map with latlon system
        return ccrs.PlateCarree()
    try:
        return ccrs.epsg(crs)
    except requests.RequestException:
        logger.warning(
            "A connection to http://epsg.io/ is "
            "required for a projected coordinate reference system. "
            "Falling back to latlong."
        )
    except ValueError:
        logger.warning(
            "'{crs}' does not define a projected coordinate system. "
            "Falling back to latlong.".format(crs=crs)
        )
        return ccrs.PlateCarree()

def projected_area_factor(ax, original_crs=4326):
    """
    Helper function to get the area scale of the current projection in
    reference to the default projection.

    The default 'original crs' is assumed to be 4326, which translates
    to the cartopy default cartopy.crs.PlateCarree()
    """
    if not hasattr(ax, "projection"):
        return 1
    x1, x2, y1, y2 = ax.get_extent()
    pbounds = get_projection_from_crs(original_crs).transform_points(
        ax.projection, np.array([x1, x2]), np.array([y1, y2])
    )

    return np.sqrt(
        abs((x2 - x1) * (y2 - y1)) / abs((pbounds[0] - pbounds[1])[:2].prod())
    )
class HandlerCircle(HandlerPatch):
    """
    Legend Handler used to create circles for legend entries.

    This handler resizes the circles in order to match the same
    dimensional scaling as in the applied axis.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        fig = legend.get_figure()
        ax = legend.axes

        # take minimum to protect against too uneven x- and y-axis extents
        unit = min(np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0])
        radius = orig_handle.get_radius() * (72 / fig.dpi) * unit
        center = 5 - xdescent, 3 - ydescent
        p = plt.Circle(center, radius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# def add_legend_circles(ax, sizes, labels, srid=4326, patch_kw={}, legend_kw={}):
#     """
#     Add a legend for reference circles.

#     Parameters
#     ----------
#     ax : matplotlib ax
#     sizes : list-like, float
#         Size of the reference circle; for example [3, 2, 1]
#     labels : list-like, str
#         Label of the reference circle; for example ["30 GW", "20 GW", "10 GW"]
#     patch_kw : defaults to {}
#         Keyword arguments passed to matplotlib.patches.Circle
#     legend_kw : defaults to {}
#         Keyword arguments passed to ax.legend
#     """
#     sizes = np.atleast_1d(sizes)
#     labels = np.atleast_1d(labels)

#     assert len(sizes) == len(labels), "Sizes and labels must have the same length."

#     if hasattr(ax, "projection"):
#         area_correction = projected_area_factor(ax, srid) ** 2
#         sizes = [s * area_correction for s in sizes]

#     handles = [Circle((0, 0), radius=s**0.5, **patch_kw) for s in sizes]

#     legend = ax.legend(
#         handles, labels, handler_map={Circle: HandlerCircle()}, **legend_kw
#     )

#     ax.get_figure().add_artist(legend)



def map_differences(n_national, n_local, ax = None, title=None):
    if ax is None:
        ax = plt.gca()
            
    # generators and storage units
    bus_sizes = (
            n_national.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()+\
            n_national.storage_units.groupby(["bus"]).p_nom_opt.sum()
        )-\
        (n_local.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()+\
            n_local.storage_units.groupby(["bus"]).p_nom_opt.sum()
        )
    # Generator-only
    # bus_sizes = n_national.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()-\
    #             n_local.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()
                
    # Storage-only
    # bus_sizes = n_national.storage_units.groupby(["bus"]).p_nom_opt.sum()-\
    #     n_local.storage_units.groupby(["bus"]).p_nom_opt.sum()
    print(bus_sizes)

    # bus_size_factor = 5e+3 # central scenario
    bus_size_factor= 1e+4
    line_widths_exp = n_national.lines.s_nom_opt-n_local.lines.s_nom_opt
    # do not display differences less than 1 MVA
    line_widths_exp[abs(line_widths_exp)<1]=0.
    link_widths_exp = n_national.links.p_nom_opt-n_local.links.p_nom_opt
    link_widths_exp[abs(link_widths_exp)<1]=0.

    # linewidth_factor = 8e1 # central scenario
    linewidth_factor = 1.5e2
    gb_boundaries = [-6, 2, 50, 60.5]
    # gb_boundaries = [-9, 12.5, 46, 62]

    n_national.plot(
            line_widths= abs(line_widths_exp) / linewidth_factor,
            link_widths=abs(link_widths_exp) / linewidth_factor,
            line_colors=np.sign(line_widths_exp),
            link_colors=np.sign(link_widths_exp),
            line_cmap='RdBu_r',
            link_cmap='RdBu_r',
            bus_sizes=(abs(bus_sizes))/bus_size_factor,
            bus_colors = np.sign(bus_sizes),
            # bus_cmap='RdBu_r',
            bus_cmap = 'RdBu_r',
            boundaries = gb_boundaries,
            geomap=True,
            color_geomap = {'ocean':'None','land':'lightslategray'},
            ax=ax)
    if title != None:
        ax.set_title(f'{title}', fontsize=10)
    # return fig

def make_legend(fig):
    handles = []
    labels = []
    linewidth_factor = 8e1
    

    for s in (250, 100):
        handles.append(
            plt.Line2D(
                [0], [0], linewidth= s / linewidth_factor,
                color ='k',
            )
        )
        labels.append("{} MW".format(s))
    l1_1 = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.52),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1.5,
        title="Transmission",
    )
    l1_1._legend_box.align = "left"

    ax.add_artist(l1_1)

    sizes = [2, 1, 0.5]

    labels = ["{} GW".format(s) for s in sizes]
    
    if hasattr(ax, "projection"):
        area_correction = projected_area_factor(ax) ** 2
        sizes = [s * area_correction/10 for s in sizes]

    handles = [Circle((0, 0), radius=s**0.5, color = 'k') for s in sizes]

    l2 = ax.legend(
        handles,
        labels,
        handler_map = {Circle: HandlerCircle()},
        loc="center left",
        bbox_to_anchor=(1.05, 0.3),
        frameon=False,
        labelspacing=1.7,
        title="Generation and storage",
    )
    
    l2._legend_box.align = "left"

    ax.get_figure().add_artist(l2)

    # investment colors
    cmap = matplotlib.cm.get_cmap('RdBu_r')    
    handles = []
    labels = []
    handles = [plt.Line2D([0], [0], color=cmap(1.), linewidth=10),
             plt.Line2D([0], [0], color=cmap(0.), linewidth=10) 
        ]
    labels = ['Oversized','Undersized']
    
    l3 = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.7),
        handletextpad=1.5,
        labelspacing=0.8,
        frameon=False,
        title = 'Error direction'

    )
    l3._legend_box.align = "left"
    ax.add_artist(l3)

    return fig

def make_legend_multi(fig):
    handles = []
    labels = []
    linewidth_factor = 1.5e2

    for s in (500, 250):
        handles.append(
            plt.Line2D(
                [0], [0], linewidth= s / linewidth_factor,
                color ='k',
            )
        )
        labels.append("{} MW".format(s))
    l1_1 = ax[1,1].legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 1.2),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1.5,
        title="Transmission",
    )
    l1_1._legend_box.align = "left"

    fig.add_artist(l1_1)
    # investment colors
    cmap = matplotlib.cm.get_cmap('RdBu_r')    
    handles = []
    labels = []
    handles = [plt.Line2D([0], [0], color=cmap(1.), linewidth=10),
             plt.Line2D([0], [0], color=cmap(0.), linewidth=10) 
        ]
    labels = ['Oversized','Undersized']
    
    l3 = ax[1,1].legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 1.6),
        handletextpad=1.5,
        labelspacing=0.8,
        frameon=False,
        title = 'Error direction'

    )
    l3._legend_box.align = "left"
    fig.add_artist(l3)
    # bus_size_factor= 7e+3


    sizes = [5, 3, 1]

    labels = ["{} GW".format(s) for s in sizes]
    
    if hasattr(ax[0,0], "projection"):
        area_correction = projected_area_factor(ax[0,0]) ** 2
        sizes = [s * area_correction/10 for s in sizes]

    handles = [Circle((0, 0), radius=s**0.5, color = 'k') for s in sizes]


    # handles = make_legend_circles_for(
    #     [5,3,1],
    #     scale=bus_size_factor, 
    #     color = 'k',
    #     edgecolor='None'
    # )
    
    # labels = ["{} GW".format(s) for s in (5, 3, 1)]
    l2 = ax[1,1].legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.055,0.7),
        frameon=False,
        labelspacing=1.7,
        title="Generation and storage",
        handler_map = {Circle: HandlerCircle()},
    )
    l2._legend_box.align = "left"

    fig.add_artist(l2)

    return fig

###
# central scenario
# fig, ax = plt.subplots(
#     figsize = [7,7],
#     subplot_kw={"projection": ccrs.EuroPP()}
# )

# map_differences(homo, het, ax)
# make_legend(fig)

# plt.savefig(het_base+'/Plots/central scenario differences.png', format = 'png',
#             bbox_inches = 'tight', dpi=300)

# all other scenarios
fig, ax = plt.subplots(2,2,
    figsize = [3.5,6],
    subplot_kw={"projection": ccrs.EuroPP()}
)
map_differences(homo_localized_transition, het_localized_transition, ax[0,0],
                title = 'Localized transition')
map_differences(homo_national_green_grid, het_national_green_grid, ax[0,1],
                title = 'National green grid')
map_differences(homo_capacity_for_carbon, het_capacity_for_carbon, ax[1,1],
                title = 'Capacity for carbon')
map_differences(homo_carbon_stagnation, het_carbon_stagnation, ax[1,0],
                title = 'Carbon stagnation')
make_legend_multi(fig)
plt.savefig(het_base+'/Plots/other scenario differences.png',bbox_inches = 'tight', dpi=300)

# %% plot bus size differences vs. peak demand differences? no clear correlation
# probably want to investigate more correlations (e.g. with solar or wind potential, flexible generation capacity, etc)

regions_gb['peak_demand_ratio']

bus_sizes = homo.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()-\
            het.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()
            
bus_sizes_gb = bus_sizes[bus_sizes.index.str.contains("GB1")]

            
plt.scatter(regions_gb['peak_demand_ratio'],bus_sizes_gb)


# %% error calculations

# total absolute error in storage and generation

def generation_bus_differences(national,local, bus_filter = None):
    national_buses = national.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()+\
        national.storage_units.groupby(["bus"]).p_nom_opt.sum()
    local_buses = local.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()+\
        local.storage_units.groupby(["bus"]).p_nom_opt.sum()
    if bus_filter!=None:
        national_buses_filtered = national_buses[national_buses.index.str.contains(bus_filter)]
        local_buses_filtered = local_buses[local_buses.index.str.contains(bus_filter)]

    else:
        national_buses_filtered = national_buses
        local_buses_filtered = local_buses
    bus_differences = national_buses_filtered - local_buses_filtered
    return bus_differences

def generation_total_absolute_error(national, local, *bus_filter):
    bus_diffs = generation_bus_differences(national, local, *bus_filter)
    absolute_error = abs(bus_diffs).sum()
    return absolute_error

def generation_total_absolute_percent_error(national, local, *bus_filter):
    bus_diffs = generation_bus_differences(national, local, *bus_filter)
    absolute_error = abs(bus_diffs).sum()
    local_buses_total = (local.generators.query('carrier != "load"').groupby(["bus"]).p_nom_opt.sum()+\
        local.storage_units.groupby(["bus"]).p_nom_opt.sum()).sum()
    TAPE = absolute_error/local_buses_total
    return TAPE

central_gen_TAE = generation_total_absolute_error(homo, het, 'GB1')/1e3
central_gen_TAPE = generation_total_absolute_percent_error(homo, het, 'GB1')*100
central_gen_max_error = generation_bus_differences(homo, het, 'GB1').max()/1e3
central_gen_min_error = generation_bus_differences(homo, het, 'GB1').min()/1e3



localized_transition_gen_TAE = generation_total_absolute_error(homo_localized_transition, het_localized_transition, 'GB1')/1e3
localized_transition_gen_TAPE = generation_total_absolute_percent_error(homo_localized_transition, het_localized_transition, 'GB1')*100
localized_gen_max_error = generation_bus_differences(homo_localized_transition, het_localized_transition, 'GB1').max()/1e3
localized_gen_min_error = generation_bus_differences(homo_localized_transition, het_localized_transition, 'GB1').min()/1e3

national_green_grid_gen_TAE = generation_total_absolute_error(homo_national_green_grid, het_national_green_grid, 'GB1')/1e3
national_green_grid_gen_TAPE = generation_total_absolute_percent_error(homo_national_green_grid, het_national_green_grid, 'GB1')*100
national_green_grid_gen_max_error = generation_bus_differences(homo_national_green_grid, het_national_green_grid, 'GB1').max()/1e3
national_green_grid_gen_min_error = generation_bus_differences(homo_national_green_grid, het_national_green_grid, 'GB1').min()/1e3

capacity_for_carbon_gen_TAE = generation_total_absolute_error(homo_capacity_for_carbon, het_capacity_for_carbon, 'GB1')/1e3
capacity_for_carbon_gen_TAPE = generation_total_absolute_percent_error(homo_capacity_for_carbon, het_capacity_for_carbon, 'GB1')*100
capacity_for_carbon_gen_max_error = generation_bus_differences(homo_capacity_for_carbon, het_capacity_for_carbon, 'GB1').max()/1e3
capacity_for_carbon_gen_min_error = generation_bus_differences(homo_capacity_for_carbon, het_capacity_for_carbon, 'GB1').min()/1e3

carbon_stagnation_gen_TAE = generation_total_absolute_error(homo_carbon_stagnation, het_carbon_stagnation, 'GB1')/1e3
carbon_stagnation_gen_TAPE = generation_total_absolute_percent_error(homo_carbon_stagnation, het_carbon_stagnation, 'GB1')*100
carbon_stagnation_gen_max_error = generation_bus_differences(homo_carbon_stagnation, het_carbon_stagnation, 'GB1').max()/1e3
carbon_stagnation_gen_min_error = generation_bus_differences(homo_carbon_stagnation, het_carbon_stagnation, 'GB1').min()/1e3

# transmission-- no need to do no-expansion cases

# transmission_MAE_central = mean_absolute_error(pd.concat([het.lines.s_nom_opt,
#     het.links.p_nom_opt[~homo.links.p_nom_opt.index.str.contains('heat pump')&~homo.links.p_nom_opt.index.str.contains('battery')]]),
#     pd.concat([homo.lines.s_nom_opt,
#         homo.links.p_nom_opt[~homo.links.p_nom_opt.index.str.contains('heat pump')&~homo.links.p_nom_opt.index.str.contains('battery')]]))

def transmission_line_differences(national, local):
    national_lines = pd.concat([national.lines.s_nom_opt,
        national.links.p_nom_opt[~national.links.p_nom_opt.index.str.contains('heat pump')]])
    local_lines = pd.concat([local.lines.s_nom_opt,
        local.links.p_nom_opt[~local.links.p_nom_opt.index.str.contains('heat pump')]])
    transmission_differences = national_lines - local_lines
    return transmission_differences

def transmission_total_absolute_error(national, local):
    transmission_differences = transmission_line_differences(national, local)
    absolute_error = abs(transmission_differences).sum()
    return absolute_error

def transmission_total_absolute_percent_error(national, local):
    absolute_error = transmission_total_absolute_error(national, local)
    local_lines_total = pd.concat([local.lines.s_nom_opt,
        local.links.p_nom_opt[~local.links.p_nom_opt.index.str.contains('heat pump')]]).sum()
    TAPE = absolute_error/local_lines_total
    return TAPE

central_trans_TAE = transmission_total_absolute_error(homo, het)/1e3
central_trans_TAPE = transmission_total_absolute_percent_error(homo, het)*100
central_trans_max_error = transmission_line_differences(homo,het).max()/1e3
central_trans_min_error = transmission_line_differences(homo,het).min()/1e3
# !!! left off here
national_green_grid_trans_TAE = transmission_total_absolute_error(homo_national_green_grid, het_national_green_grid)/1e3
national_green_grid_trans_TAPE = transmission_total_absolute_percent_error(homo_national_green_grid, het_national_green_grid)*100
national_green_grid_trans_max_error = transmission_line_differences(homo_national_green_grid,het_national_green_grid).max()/1e3
national_green_grid_trans_min_error = transmission_line_differences(homo_national_green_grid,het_national_green_grid).min()/1e3

capacity_for_carbon_trans_TAE = transmission_total_absolute_error(homo_capacity_for_carbon, het_capacity_for_carbon)/1e3
capacity_for_carbon_trans_TAPE = transmission_total_absolute_percent_error(homo_capacity_for_carbon, het_capacity_for_carbon)*100
capacity_for_carbon_trans_max_error = transmission_line_differences(homo_capacity_for_carbon,het_capacity_for_carbon).max()/1e3
capacity_for_carbon_trans_min_error = transmission_line_differences(homo_capacity_for_carbon,het_capacity_for_carbon).min()/1e3

#%% some plot of storage capacity differences

# !!! ADD STORAGE UNITS like this
# bus_sizes = pd.concat(
#     (
#         n.generators.query('carrier != "load"')
#         .groupby(["bus", "carrier"])
#         .p_nom_opt.sum(),
#         n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum(),
#     )
# )

# homo_gen = homo.generators.groupby(['bus', 'carrier']).p_nom_opt.sum()/1e3

homo_gen = pd.concat(
    (
        homo.generators.query('carrier != "load"')
        .groupby(["bus", "carrier"])
        .p_nom_opt.sum()/1e3,
        homo.storage_units.groupby(["bus", "carrier"])
        .p_nom_opt.sum()/1e3,
    )
)

homo_gen_gb = homo_gen[homo_gen.index.get_level_values(0).str.contains('GB1')]

het_gen = pd.concat(
    (
        het.generators.query('carrier != "load"')
        .groupby(["bus", "carrier"])
        .p_nom_opt.sum()/1e3,
        het.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum()/1e3,
    )
)


het_gen_gb = het_gen[het_gen.index.get_level_values(0).str.contains('GB1')]

gen_diffs = (homo_gen_gb.unstack() - het_gen_gb.unstack()).fillna(0.)


def drop_zero_columns(df):
    zero_columns = df.columns[df.eq(0).all()]
    df.drop(zero_columns, axis=1, inplace=True)

drop_zero_columns(gen_diffs)

gen_diffs.set_index(pd.to_numeric(gen_diffs.index.str.removeprefix('GB1 ')),inplace=True)
gen_diffs.sort_index(inplace=True)

fig, ax = plt.subplots(
    figsize = [12,8],
)
gen_diffs.plot(kind='barh', stacked=True, ax=ax)
ax.set_xlabel('Capacity difference (GW)')
ax.set_title('Generation and storage differences by technology')

plt.savefig(het_base +'/plots/technology differences.png', bbox_inches = 'tight', dpi=300)

#%% technology differences in buses instead?

gen_diffs = (homo_gen_gb.unstack() - het_gen_gb.unstack()).fillna(0.)


def drop_zero_columns(df):
    zero_columns = df.columns[df.eq(0).all()]
    df.drop(zero_columns, axis=1, inplace=True)

drop_zero_columns(gen_diffs)

technologies = gen_diffs.columns

for tech in technologies:
    regions_gb[f'{tech}_difference'] = gen_diffs[f'{tech}']
    divnorm = colors.TwoSlopeNorm(vmin=regions_gb[f'{tech}_difference'].min(), vcenter=0., vmax=regions_gb[f'{tech}_difference'].max())
    ax = regions_gb.plot(column=f'{tech}_difference',cmap='RdBu_r',norm = divnorm, legend = True, edgecolor='k')
    ax.set_axis_off()
    ax.set_title(f'{tech}')
# %% map of different labeled regions
# fig, ax = plt.subplots(
#     # figsize = [7,7],
#     subplot_kw={"projection": ccrs.EuroPP()}
# )
# ax = regions_gb.plot('peak_demand_ratio',cmap='RdBu_r',norm = divnorm, legend=True, edgecolor='k')
# import random
# regions_gb['color'] = ''

# for idx, row in regions_gb.iterrows():
#     rand_color = list([random.random(), random.random(), random.random()])

#     regions_gb.at[idx, 'color'] = rand_color
    
    
# ax = regions_gb.plot(fc = regions_gb['color'])
ax = regions_gb.plot(cmap = 'tab20', categorical=True, edgecolor='0.3',linewidth=0.5)

# color = list(np.random.choice(range(256), size=3))
ax.set_axis_off()

regions_gb['coords'] = regions_gb['geometry'].apply(lambda x: x.representative_point().coords[:])
regions_gb['coords'] = [coords[0] for coords in regions_gb['coords']]
# regions_gb['centroid'] = regions_gb.to_crs(ccrs.EuroPP()).geometry.centroid
# plt.savefig(het_base +'/plots/bus regions.png', bbox_inches = 'tight', dpi=300)

# plt.annotate(regions_gb.index, 
#              regions_gb['centroid'], 
#              horizontalalignment='center',
#              fontsize='x-small')


# regions_gb.apply(lambda x: ax.annotate(text=x.index, xy=x.geometry.centroid.coords[0], ha='center'), axis=1);

# for idx, row in regions_gb.iterrows():
#     centroid = row.geometry.centroid
#     label = idx.replace('GB1 ','')
#     region_size = row.geometry.area

#     # Adjust label placement for smaller regions
#     if region_size < 0.5:  # Adjust the threshold as needed
#         ax.annotate(text=label, xy=(centroid.x, centroid.y), xytext=(5, 5),
#                     textcoords="offset points", ha='left', va='bottom', fontsize=8,
#                     arrowprops = {'arrowstyle':'-'})
#     else:
#         ax.text(centroid.x, centroid.y, label, fontsize=8, ha='center', va='center')



#  Works, just testing something different
for idx, row in regions_gb.iterrows():

    bus_num = idx.replace('GB1 ','')
    plt.annotate(text = bus_num,
                  xy = row['coords'],
                  horizontalalignment = 'center',
                  fontsize = 'x-small'
                  )
plt.savefig(het_base +'/plots/region map.png', bbox_inches = 'tight', dpi=300)

#%% plot time series of total electric demand

plt.figure()
homo_heat_elec.sum(dim='bus').plot(alpha=0.5)

het_heat_elec.sum(dim='bus').plot(alpha=0.5)

# plot time series of total electric demand on peak heating demand day

plt.figure()
(homo_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3).plot()

(het_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3).plot()

# add electricity demand at each bus-- maybe use pandas dataframes

homo_elec = homo.loads_t.p_set.loc[:,homo.loads_t.p_set.columns.str.contains("AC")]
homo_elec_gb = homo_elec.loc[:,homo_elec.columns.str.contains("GB1")]

het_elec = het.loads_t.p_set.loc[:,het.loads_t.p_set.columns.str.contains("AC")]
het_elec_gb = het_elec.loc[:,het_elec.columns.str.contains("GB1")]


homo_elec_gb.loc['2019-01-31'].sum(axis=1).plot()

# total electric heating demand
homo_elec_gb.sum(axis=1)+homo_heat_elec.sum(dim='bus')
het_elec_gb.sum(axis=1)+het_heat_elec.sum(dim='bus')

# area chart of stacked demand for the whole year
plt.stackplot(homo_elec_gb.index,homo_elec_gb.sum(axis=1),homo_heat_elec.sum(dim='bus'))

fig, ax = plt.subplots(1,2, sharey=True)

# area chart for stacked demand on peak demand day
ax[0].stackplot(homo_elec_gb.loc['2019-01-31'].index,
              homo_elec_gb.loc['2019-01-31'].sum(axis=1)/1e3,
              homo_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3)
ax[0].set_title('Homogeneous demand')
ax[1].stackplot(het_elec_gb.loc['2019-01-31'].index,
              het_elec_gb.loc['2019-01-31'].sum(axis=1)/1e3,
              het_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3)
ax[1].set_title('Heterogeneous demand')
ax[0].set_ylabel('Demand [GW]')

# %% heterogeneous only for Rhodes poster

# import matplotlib.dates as mdates


# fig, ax = plt.subplots()

# # area chart for stacked demand on peak demand day
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# ax.stackplot(het_elec_gb.loc['2019-01-31'].index,
#               het_elec_gb.loc['2019-01-31'].sum(axis=1)/1e3,
#               het_heat_elec.sum(dim='bus').sel(time='2019-01-31')/1e3,
#               labels = ['Historical', 'Heat pump (projected)'],
#               alpha = 0.85)
# plt.gcf().autofmt_xdate()
# ax.set_title('Additional heat pump demand on 31 Jan 2019')
# ax.set_ylabel('Electricity demand [GW]')
# ax.legend(loc='lower center')
# plt.savefig(het_base +'/plots/additional demand.png', bbox_inches = 'tight', dpi=300)

