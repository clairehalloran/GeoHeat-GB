#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:35:55 2023

@author: Claire Halloran, University of Oxford

Script for visualizating operations regret analysis
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
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
import matplotlib as mpl
from matplotlib import colors
to_rgba = mpl.colors.colorConverter.to_rgba

logger = logging.getLogger(__name__)

national_base = r'/Users/mans3904/Library/CloudStorage/OneDrive-Nexus365/DPhil/Spatiotemporal heating results/20230607 single temperature region battery storage'


# import network
national = pypsa.Network(national_base + r"/networks/elec_s_39_ec_lv1.15_Co2L0.1-EQ0.95c_op.nc")

# investigate load-shed generators in Britain
### NOTE: load-shedding in kW, not MW!!!

load_shed_gb = national.generators_t.p.filter(like='GB1').filter(like='load')

load_shed_gb.columns = load_shed_gb.columns.str.strip(' load')

# load shedding near peak day
load_shed_gb.sum(axis=1).loc['2019-01-30':'2019-01-31'].plot()
load_shed_gb.sum(axis=1).plot()



# load shedding near peak day
load_shed_gb.loc['2019-01-31 08:00'].plot()


regions = gpd.read_file(national_base + r'/regions_onshore_elec_s_39.geojson')
regions = regions.set_index('name')

regions_gb = regions[regions.index.str.contains("GB1")]


peak_load_shed = load_shed_gb.max()
# peak_load_shed.set_index(peak_load_shed.index.str.removesuffix(' load'))

regions_gb = regions_gb.assign(peak_load_shed = load_shed_gb.max()/1e6)

ax = regions_gb.plot('peak_load_shed',cmap='Reds', legend=True, edgecolor='k')
ax.set_axis_off()
ax.set_title('Peak load shed')


regions_gb = regions_gb.assign(total_load_shed = load_shed_gb.sum()/1e6)

ax = regions_gb.plot('total_load_shed',cmap='Reds', legend=True, edgecolor='k')
ax.set_axis_off()
ax.set_title('Total load shed')
