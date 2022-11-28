
import pandas as pd
import mpl_toolkits
# from mpl_toolkits.basemap import Basemap
from obspy import UTCDateTime
import numpy as np
import matplotlib.pyplot as plb
import datetime
from geopy import distance
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from matplotlib import path, cm
import time
from Compare_cluster_functions import *
from Foreshocks_Aftershocks_productivity import *

Cat_names = ['INGV2', 'KOERI2', 'ALASKA2', 'NCEDC', 'SCEDC', 'NOA', 'JMA', 'OVSICORI']; MCP = 3.0#
Cat_names2 = ['Italy', 'Turkey', 'Alaska', 'N. CA', 'S. CA', 'Greece', 'Japan', 'Costa Rica']
# Cat_names = ['NEIC', 'GCMT2']; MCP = 5.0#
# Cat_names2 = ['NEIC', 'GCMT']

# Cat_names = ['OVSICORI']; MCP = 3.0#
# Cat_names2 = ['Costa Rica']

csv_folder = '/Users/nadavwetzler/Dropbox/Public/DataSet/UCSC/Foreshocks/Catalogs_csv/'
# folder0 = '/Users/nadavwetzler/Dropbox/Public/DataSet/UCSC/Foreshocks/Figs_Compare/clusters_v6/'

DATA = CatData(Cat_names, csv_folder, 30, 2.0, 30, 0, 500, 70, -999)
DATA.trimcats()
DATA.CalcGR_Mc_cats(True, save_b=False)
DATA.AboveMcCat(MCP)

# Moment foresocks vs BG
M0_WnC = np.zeros(len(Cat_names))
M0_ZnBZ_R = np.zeros(len(Cat_names))
M0_TMC_R = np.zeros(len(Cat_names))
cat_years = np.zeros(len(Cat_names))
n_mainshocks = np.zeros((len(Cat_names), 3))

if MCP > -999:
    csv_folder = '%sMc%d/' % (csv_folder, MCP*10)
fig = plb.figure(4161)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

for ii in range(len(Cat_names)):
    cat_years[ii] = DATA.CATS[ii].years
    folder0 = '/Users/nadavwetzler/Dropbox/Public/DataSet/UCSC/Foreshocks/Figs_Compare/clusters%s/' % Cat_names[ii]

    CLUST_Z = pd.read_csv('%scat_%s.clusters.a2.0.ZnBZ.csv' % (csv_folder, Cat_names[ii]))
    deCLUST_Z = pd.read_csv('%scat_%s.declust.ZnBZ.csv' % (csv_folder, Cat_names[ii]))
    n_mainshocks[ii, 0] = sum((CLUST_Z.c_type == 1) & (CLUST_Z.m0 >= MCP + 2.0))

    CLUST_M = pd.read_csv('%s%s.clusters.a2.0.WnC.csv' % (csv_folder, Cat_names[ii]))
    deCLUST_M = pd.read_csv('%scat_%s.declust.WnC.csv' % (csv_folder, Cat_names[ii]))
    n_mainshocks[ii, 1] = sum((CLUST_M.c_type == 1) & (CLUST_M.m0 >= MCP + 2.0))

    # CLUST_G = pd.read_csv('%s%s.clusters.a2.0.GnK.csv' % (csv_folder, Cat_names[ii]))
    # deCLUST_G = pd.read_csv('%scat_%s.declust.GnK.csv' % (csv_folder, Cat_names[ii]))

    CLUST_W = pd.read_csv('%s%s.clusters.a2.0.TMC.csv' % (csv_folder, Cat_names[ii]))
    deCLUST_W = pd.read_csv('%scat_%s.declust.TMC.csv' % (csv_folder, Cat_names[ii]))
    n_mainshocks[ii, 2] = sum((CLUST_W.c_type == 1) & (CLUST_W.m0 >= MCP + 2.0))

    # plotM0timeLine(CLUST_Z, CLUST_M, CLUST_W)

    print('Cdata loaded: %s' % Cat_names[ii])
    CLUST = make_comparison_tble3(CLUST_Z, CLUST_M, CLUST_W)
    CLUST.to_csv('%s%s.clusters.Comp.csv' % (csv_folder, Cat_names[ii]))
    # M0_WnC[ii], M0_ZnBZ_R[ii], M0_TMC_R[ii] = ForeshockMoment_BG_FS(CLUST, deCLUST_Z, deCLUST_M, deCLUST_W, DATA.CATS[ii].cat)

    # plot_foreshocks_mag(CLUST_G,CLUST_M, CLUST_Z, fig10.add_subplot(2,3,ii+1), Cat_names[ii])

    plot_cluster_Maps3(Cat_names[ii], CLUST, deCLUST_Z, deCLUST_M, deCLUST_W, DATA.CATS[ii].cat, folder0, MCP)

    ForeshocksDuration(ax1, ax2, ax3, Cat_names[ii], CLUST_Z, CLUST_M, CLUST_W, deCLUST_Z, deCLUST_M, deCLUST_W)

    # plot_events_after60days(CLUST_Z, Cat_names[ii], ii)

    # plot_N_ZvM(CLUST, Cat_names[ii], ii)

    # plot_events_Z_vs_WnC(CLUST, Cat_names2[ii], ii)

    # plot_ctype_bars_2cats(CLUST_Z, CLUST_M, Cat_names[ii], ii)
    

ax1.legend()
# plot_FS_Moment_cat(M0_WnC, M0_ZnBZ_R, M0_TMC_R, Cat_names2)
# fig = plb.figure(877)
#plot_mainshocks_rate(fig.add_subplot(1, 1, 1), n_mainshocks, cat_years, Cat_names2)
print('DONE!!!')
plb.show()

