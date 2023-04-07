# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 23:35:20 2022

@author: Claire
"""

import pypsa
import matplotlib.pyplot as plt
# plt.style.use('bmh')
import cartopy.crs as ccrs

# n = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\results\networks\elec_s_47_ec_lcopt_Co2L.nc")
n = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\elec_s.nc")
base = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\base.nc")
elec = base = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\elec.nc")
n_39 = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\elec_s_39.nc")
n_39_ec = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\elec_s_39_ec.nc")
n_48_ec_opt = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\elec_s_48_ec_lcopt_Co2L.nc")
n_solved =pypsa.Network(r'C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\results\networks\elec_s_48_ec_lcopt_Co2L.nc')

#%% checking heating
cops = n_39_ec.links_t.efficiency

loads = n_39_ec.loads_t.p_set

#%% plot regions in Britain

import geopandas as gpd
import random
regions = gpd.read_file('resources/regions_onshore_elec_s_39.geojson')
# regions = regions.set_index('name')

regions_gb = regions[regions['name'].str.contains("GB1")]

colors = [('#%06X' % random.randint(0, 0xFFFFFF)) for i in range(len(regions_gb))]

ax = regions_gb.plot(color = colors)

ax.set_axis_off()

ax = regions_gb.plot()
ax.set_axis_off()
#%% plot network in Britain

n_39.plot()



# create a list of the buses you want to plot
bus_list = ['GB1 1', 'GB1 18']

# subset the bus_geodata attribute of the network to include only the selected buses
bus_geodata_subset = n_39_ec.buses.loc[bus_list]

# plot the selected buses using the subsetted bus_geodata attribute
plt.figure(figsize=(8, 8))
bus_geodata_subset.plot()
plt.show()



#%%
# plot map of network
n.plot()
# list of generators
generators = n.generators
# list of storage units
n.storage_units.head()

links = n_solved.links

#%% identify any disconnected buses

unconnected_buses = n_39_ec.buses.index.difference(n_39_ec.links.bus0).difference(n_39_ec.links.bus1)


#%% plot clustered network

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.EuroPP()}, figsize=(9, 9))

network = n_39.plot(
       projection=ccrs.EuroPP(),
       color_geomap={'ocean':'lightskyblue','land':'floralwhite'}
       )


#%% time-varying input data
# load time series data at each bus
n.loads_t.p_set
# plot total system demand
n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))
# generator capacity factors at different times and locations
n.generators_t.p_max_pu

#%% results
# system cost
n.objective/1e9 # billion euros p.a.

# transmission line expansion
(n.lines.s_nom_opt - n.lines.s_nom)

# optimal generator and storage capacities
n.generators.groupby('carrier').p_nom_opt.sum()/1e3
n.generators.groupby('bus').p_nom_opt.sum()/1e3


n.storage_units.groupby('carrier').p_nom_opt.sum()/1e3

# energy storage
# state of charge over whole system
n.storage_units_t.state_of_charge.sum(axis=1)
# all batteries in system
(n.storage_units_t.state_of_charge.filter(like='battery', axis=1))
(n.storage_units_t.state_of_charge.filter(like='H2', axis=1))
(n.storage_units_t.state_of_charge.filter(like='hydro', axis=1))

#%%

n.generators.query('carrier != "load"').groupby(["bus", "carrier"]).p_nom_opt.sum(),
n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum(),
n.stores.groupby(["bus","carrier"]).sum().columns,
n.stores
links = n.links
#%% plotting networks


# display line loading

loading = (n.lines_t.p0.abs().mean().sort_index()/(n.lines.s_nom_opt*n.lines.s_max_pu).sort_index()).fillna(0.)
fig,ax = plt.subplots(
            figsize=(10,10),
            subplot_kw={'projection':ccrs.EuroPP()})
n.plot(ax=ax,
       bus_colors='gray',
       branch_components=['Line'],
       line_widths=n.lines.s_nom_opt/3e3,
       line_colors=loading,
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       bus_sizes=0)
ax.axis('off')
