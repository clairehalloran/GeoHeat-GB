# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:34:34 2023

@author: Claire Halloran, University of Oxford

This runs a power flow on the PyPSA network built in PyPSA-Eur for the UK. 
Then plots the average voltage angle at each bus.

"""

import pypsa
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import numpy as np

# network with 0% heat penetration
n = pypsa.Network(r"C:\Users\clair\OneDrive - Nexus365\DPhil\pypsa-eur\networks\GB No Heat\elec.nc")

# run a power flow analysis
n.lpf()

# plot mean annual voltage angle this on a map using rainbow colors

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.OSGB()}, figsize=(9, 9))
cmap = plt.cm.jet_r
norm = mpl.colors.Normalize(vmin=n.buses_t.v_ang.mean().min()* 180.0 / np.pi,
                            vmax=n.buses_t.v_ang.mean().max()* 180.0 / np.pi)

network = n.plot(
       projection=ccrs.OSGB(),
       bus_colors=n.buses_t.v_ang.mean()* 180.0 / np.pi,
       bus_cmap = cmap,
       bus_sizes = 5e-3,
       title = 'Mean voltage angle 2019',
       color_geomap={'ocean':'lightskyblue','land':'floralwhite'}
       )

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             label='Voltage angle [deg]')

plt.savefig(r'C:\Users\clair\OneDrive\Desktop\voltage_angle.png', dpi= 300.)


# maybe use agglomerative clustering
