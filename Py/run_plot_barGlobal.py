import pandas as pd
import numpy as np
import matplotlib.pyplot as plb
from Compare_cluster_functions import *
from Foreshocks_Aftershocks_productivity import *


plb.rc('pdf', fonttype=42)

# Cat_names = ['NEIC', 'GCMT2']
# Cat_Label = ['NEIC', 'GCMT']

Cat_names = ['NEIC']
Cat_Label = ['NEIC']


ZMc = 0.0
Mc = 5.0
dxy = 0.5  # grid cell size in degrees to calculate area of seismicity
rMaxC = 0.1  # ratio from max seismicity cell to calc area of seismicity
fsTypes = ['nf_all', 'nf', 'nf_apr']
fsTypes1 = ['All Foreshocks', '30 days', 'Aperture 2.0']

folderF = '/Users/nadavwetzler/Library/CloudStorage/GoogleDrive-nadav.wetzler@gmail.com/My Drive/DataSet/UCSC/Foreshocks/'
csv_folder = folderF+'Catalogs_csv/'

DATA = CatData(Cat_names, folderF, 60, 2.0, 30, 0, 500, 70, -999)
DATA.trimcats()
DATA.CalcGR_Mc_cats(True, save_b=False)
DATA.AboveMcCat(Mc)
if Mc > -999:
    csv_folder = '%sMc%d/' % (csv_folder, Mc*10)

n_mainshocks = np.zeros((len(Cat_names), 3))
cat_years = np.zeros(len(Cat_names))

CLUST_Z = []
CLUST_M = []
CLUST_G = []
CLUST_W = []
deCLUST_Z = []
deCLUST_M = []
deCLUST_G = []
deCLUST_W = []
PBG_Z = []
PBG_W = []


nfh0 = np.zeros((len(fsTypes), 3, len(Cat_names)))
nfh1 = np.zeros((len(fsTypes), 3, len(Cat_names)))
nfh5 = np.zeros((len(fsTypes), 3, len(Cat_names)))
nfh0std = np.zeros((len(fsTypes), 3, len(Cat_names)))
nfh1std = np.zeros((len(fsTypes), 3, len(Cat_names)))
nfh5std = np.zeros((len(fsTypes), 3, len(Cat_names)))
iGCMT = -1
CL = pd.read_csv(folderF+'Py/GlobalCoastLine.csv')
LL = pd.read_csv(folderF+'LingLingTable.csv')
AnSh = pd.read_csv(folderF+'Allmann_and_Shearer_SD_2009.csv')

n_mainshocks = np.zeros((len(Cat_names), 3))
n_mainshocks_a = np.zeros((len(Cat_names), 3))

for ii in range(len(Cat_names)):
    print('Loading %s' % Cat_names[ii])
    cat_years[ii] = DATA.CATS[ii].years
    if Cat_names[ii] == 'GCMT2':
        iGCMT = ii

    CLUST_Z.append(pd.read_csv('%scat_%s.clusters.a2.0.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
    deCLUST_Z.append(pd.read_csv('%scat_%s.declust.ZnBZ.csv' % (csv_folder, Cat_names[ii])))

    CLUST_M.append(pd.read_csv('%s%s.clusters.a2.0.WnC.csv' % (csv_folder, Cat_names[ii])))
    deCLUST_M.append(pd.read_csv('%scat_%s.declust.WnC.csv' % (csv_folder, Cat_names[ii])))

    CLUST_W.append(pd.read_csv('%s%s.clusters.a2.0.TMC.csv' % (csv_folder, Cat_names[ii])))
    deCLUST_W.append(pd.read_csv('%scat_%s.declust.TMC.csv' % (csv_folder, Cat_names[ii])))

    CLUST_Z[ii] = add_lingling_data(LL, CLUST_Z[ii])
    CLUST_M[ii] = add_lingling_data(LL, CLUST_M[ii])
    CLUST_W[ii] = add_lingling_data(LL, CLUST_W[ii])

    CLUST_Z[ii] = add_Almann_Shearer_2009_data(AnSh, CLUST_Z[ii])
    CLUST_M[ii] = add_Almann_Shearer_2009_data(AnSh, CLUST_M[ii])
    CLUST_W[ii] = add_Almann_Shearer_2009_data(AnSh, CLUST_W[ii])

    deCLUST_W[ii] = define_Event_type(deCLUST_W[ii])
    deCLUST_Z[ii] = define_Event_type(deCLUST_Z[ii])

    I_bgW = deCLUST_W[ii]['eType'] != 1.0
    I_bgZ = deCLUST_Z[ii]['eType'] != 1.0

    n_mainshocks[ii, 0] = sum((CLUST_Z[ii].c_type == 1) & (CLUST_Z[ii].m0 >= Mc))
    n_mainshocks[ii, 1] = sum((CLUST_M[ii].c_type == 1) & (CLUST_M[ii].m0 >= Mc))
    n_mainshocks[ii, 2] = sum((CLUST_W[ii].c_type == 1) & (CLUST_W[ii].m0 >= Mc))


fig = plb.figure(4300)
plot_LL_params_fs(fig.add_subplot(2, 3, 1), CLUST_Z, LL, Cat_Label, 'ZnBZ')
plot_LL_params_fs(fig.add_subplot(2, 3, 2), CLUST_M, LL, Cat_Label, 'WnC')
plot_LL_params_fs(fig.add_subplot(2, 3, 3), CLUST_W, LL, Cat_Label, 'TMC')

plot_AnSh_params_fs(fig.add_subplot(2, 3, 4), CLUST_Z, AnSh, Cat_Label, 'ZnBZ')
plot_AnSh_params_fs(fig.add_subplot(2, 3, 5), CLUST_M, AnSh, Cat_Label, 'WnC')
plot_AnSh_params_fs(fig.add_subplot(2, 3, 6), CLUST_W, AnSh, Cat_Label, 'TMC')

fig = plb.figure(7600)
for fi in range(len(fsTypes)):
    ax1 = fig.add_subplot(3, len(fsTypes), fi+1)
    nfh0[fi, 0], nfh1[fi, 0], nfh5[fi, 0], nfh0std[fi, 0], nfh1std[fi, 0], nfh5std[fi, 0], f_nof_Z, fper_Z = plotFSbar(CLUST_Z, ax1, Cat_Label, fsTypes[fi])
    ax1.set_ylabel('ZnBZ')
    ax1.set_title(fsTypes1[fi])
    ax1.set_ylim([0, 100])

    ax2 = fig.add_subplot(3, len(fsTypes), len(fsTypes)*1+fi+1)
    nfh0[fi, 1], nfh1[fi, 1], nfh5[fi, 1], nfh0std[fi, 1], nfh1std[fi, 1], nfh5std[fi, 1], f_nof_M, fper_M = plotFSbar(CLUST_M, ax2, Cat_Label, fsTypes[fi])
    ax2.set_ylabel('WnC')
    ax2.set_ylim([0, 100])

    ax3 = fig.add_subplot(3, len(fsTypes), len(fsTypes)*2+fi+1)
    nfh0[fi, 2], nfh1[fi, 2], nfh5[fi, 2], nfh0std[fi, 2], nfh1std[fi, 2], nfh5std[fi, 2], f_nof_W, fper_W = plotFSbar(CLUST_W, ax3, Cat_Label, fsTypes[fi])
    ax3.set_ylabel('TMC')
    ax3.set_ylim([0, 100])
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

Methods = ['ZnBZ', 'WnC', 'TMC']
colors = ['dodgerblue', 'skyblue', 'whitesmoke']
width = 0.35


for ii in range(len(Cat_names)):
    fig = plb.figure(7690 + ii)
    for fi in range(len(fsTypes)):
        ax = fig.add_subplot(1, len(fsTypes), fi+1)
        ax.bar(Methods, nfh5[fi, :, ii], width, color=colors[0], edgecolor='black')#, colors=colors[0]) # shadow=True,
        ax.bar(Methods, nfh1[fi, :, ii], width, bottom=nfh5[fi, :, ii], color=colors[1], edgecolor='black')
        ax.bar(Methods, nfh0[fi, :, ii], width, bottom=nfh5[fi, :, ii]+nfh1[fi, :, ii], color=colors[2], edgecolor='black')
        for jj in range(len(Methods)):
            ax.text(x=jj, y=nfh5[fi, jj, ii]+1, s='%d' % np.round(nfh5[fi, jj, ii]), fontdict=dict(fontsize=10), horizontalalignment='center')
            ax.text(x=jj, y=nfh1[fi, jj, ii]+nfh5[fi, jj, ii]+1, s='%d' % np.round(nfh1[fi, jj, ii]+nfh5[fi, jj, ii]), fontdict=dict(fontsize=10), horizontalalignment='center')
        plb.grid(axis='y', linestyle='--')
        ax.set_ylim([0, 100])
        ax.set_title(fsTypes[fi])
    fig.suptitle(Cat_names[ii])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)


# Compare faulting styles (Global Figure 8)
for iGCMT in range(len(Cat_names)):
    fig = plb.figure(7610 + iGCMT + 1)
    ax11 = fig.add_subplot(3, 3, 1)
    plotFMS(CLUST_Z[iGCMT], ax11, 0)
    ax11.set_ylabel('ZnBZ')
    ax11.set_title('All')

    ax12 = fig.add_subplot(3, 3, 4)
    plotFMS(CLUST_M[iGCMT], ax12, 0)
    ax12.set_ylabel('WnC')

    ax13 = fig.add_subplot(3, 3, 7)
    plotFMS(CLUST_W[iGCMT], ax13, 0)
    ax13.set_ylabel('TMC')
    
    ax21 = fig.add_subplot(3, 3, 2)
    plotFMS(CLUST_Z[iGCMT], ax21, 1)
    ax21.set_ylabel('ZnBZ')
    ax21.set_title('Interplate')

    ax22 = fig.add_subplot(3, 3, 5)
    plotFMS(CLUST_M[iGCMT], ax22, 1)
    ax22.set_ylabel('WnC')

    ax23 = fig.add_subplot(3, 3, 8)
    plotFMS(CLUST_W[iGCMT], ax23, 1)
    ax23.set_ylabel('TMC')
    
    ax31 = fig.add_subplot(3, 3, 3)
    plotFMS(CLUST_Z[iGCMT], ax31, -1)
    ax31.set_ylabel('ZnBZ')
    ax31.set_title('Intraplate')

    ax32 = fig.add_subplot(3, 3, 6)
    plotFMS(CLUST_M[iGCMT], ax32, -1)
    ax32.set_ylabel('WnC')

    ax33 = fig.add_subplot(3, 3, 9)
    plotFMS(CLUST_W[iGCMT], ax33, -1)
    ax33.set_ylabel('TMC')

    fig.suptitle(Cat_names[iGCMT])

# Compare Interplate vs. Intraplate (Global - Figure 7)
for ii in range(len(Cat_names)):
    fig = plb.figure(7710 + ii + 1)
    iGCMT = ii
    for fi in range(len(fsTypes)):
        ax11 = fig.add_subplot(3, 4, fi+1)
        plotII(CLUST_Z[iGCMT], ax11, fsTypes[fi])
        ax11.set_title(fsTypes1[fi])
        ax11.set_ylabel('ZnBZ')

        ax12 = fig.add_subplot(3, 4, 4*1+fi+1) # len(fsTypes)
        plotII(CLUST_M[iGCMT], ax12, fsTypes[fi])
        ax12.set_ylabel('WnC')

        ax13 = fig.add_subplot(3, 4, 4*2+fi+1)
        plotII(CLUST_W[iGCMT], ax13, fsTypes[fi])
        ax13.set_ylabel('TMC')
    figm = plb.figure(7810 + ii + 1)
    ax21 = figm.add_subplot(3, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax22 = figm.add_subplot(3, 1, 2, projection=ccrs.PlateCarree(central_longitude=180))
    ax23 = figm.add_subplot(3, 1, 3, projection=ccrs.PlateCarree(central_longitude=180))
    plotIImap(ax21, CLUST_Z[iGCMT], 'ZnBZ', folderF)
    plotIImap(ax22, CLUST_M[iGCMT], 'WnC', folderF)
    plotIImap(ax23, CLUST_W[iGCMT], 'TMC', folderF)
        
        
    fig.suptitle(Cat_names[ii])

print('DONE!')
plb.show()
