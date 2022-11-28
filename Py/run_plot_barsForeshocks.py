import pandas as pd
import numpy as np
import matplotlib.pyplot as plb
from Compare_cluster_functions import *
from Foreshocks_Aftershocks_productivity import *


plb.rc('pdf', fonttype=42)
Cat_names = ['INGV2', 'SCEDC', 'NCEDC', 'OVSICORI', 'ALASKA2', 'JMA', 'KOERI2', 'NOA']#,'NEIC']
Cat_Label = ['Italy', 'S. CA', 'N. CA', 'Costa Rica', 'Alaska', 'Japan', 'Turkey', 'Greece']#,'NEIC']
# Cat_names = ['OVSICORI']
# Cat_Label = ['Costa-Rica']

AreaKM2_W = [102005.,  86550.,  52548.,  68003., 510027., 278197., 469843., 247286.]
AreaKM2_Z = [204011.,  92732.,  74185.,  64912., 599669., 318381., 633670.,  210193.]
# AreaKM2_W = np.zeros(len(Cat_names))
# AreaKM2_Z = np.zeros(len(Cat_names))

dM = 2.0
ZMc = 0.0
Mc = 3.0
dxy = 0.5 # grid cell size in degrees to calculate area of seismicity
rMaxC = 0.1 # ratio from max seismicity cell to calc area of seismicity
fsTypes = ['nf_all', 'nf', 'nf_apr']
fsTypes1 = ['All Foreshocks', '30 days', 'Aperture 2.0']

csv_folder = '/Users/nadavwetzler/Dropbox/Public/DataSet/UCSC/Foreshocks/Catalogs_csv/'

DATA = CatData(Cat_names, csv_folder, 60, 2.0, 30, 0, 500, 70, -999)
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


nfh0 = np.zeros(3)
nfh1 = np.zeros(3)
nfh5 = np.zeros(3)
nfh0std = np.zeros(3)
nfh1std = np.zeros(3)
nfh5std = np.zeros(3)
iGCMT = -1
CL = pd.read_csv('/Users/nadavwetzler/Dropbox/Public/DataSet/UCSC/Foreshocks/Py/GlobalCoastLine.csv')
figAreaW = plb.figure(66801)
figAreaZ = plb.figure(66802)
figAreaW.suptitle('TMC')
figAreaZ.suptitle('ZnBZ')
n_mainshocks = np.zeros((len(Cat_names), 3))
n_mainshocks_a = np.zeros((len(Cat_names), 3))

k_mainshocks_fs = np.zeros((len(Cat_names), 3))
k_mainshocks_all = np.zeros((len(Cat_names), 3))

for ii in range(len(Cat_names)):
    print('Loading %s' % Cat_names[ii])
    cat_years[ii] = DATA.CATS[ii].years
    if Cat_names[ii] == 'GCMT2':
        iGCMT = ii
    if ZMc == 0.5:
        CLUST_Z.append(pd.read_csv('%sMc05/cat_%s.clusters.a2.0.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
        deCLUST_Z.append(pd.read_csv('%sMc05/cat_%s.declust.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
    elif ZMc == 1.0:
        CLUST_Z.append(pd.read_csv('%sMc10/cat_%s.clusters.a2.0.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
        deCLUST_Z.append(pd.read_csv('%sMc10/cat_%s.declust.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
    else:
        CLUST_Z.append(pd.read_csv('%scat_%s.clusters.a2.0.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
        deCLUST_Z.append(pd.read_csv('%scat_%s.declust.ZnBZ.csv' % (csv_folder, Cat_names[ii])))


    CLUST_M.append(pd.read_csv('%s%s.clusters.a2.0.WnC.csv' % (csv_folder, Cat_names[ii])))
    deCLUST_M.append(pd.read_csv('%scat_%s.declust.WnC.csv' % (csv_folder, Cat_names[ii])))

    # CLUST_G.append(pd.read_csv('%s%s.clusters.a2.0.GnK.csv' % (csv_folder, Cat_names[ii])))

    CLUST_W.append(pd.read_csv('%s%s.clusters.a2.0.TMC.csv' % (csv_folder, Cat_names[ii])))
    deCLUST_W.append(pd.read_csv('%scat_%s.declust.TMC.csv' % (csv_folder, Cat_names[ii])))

    PBG_Z.append(pd.read_csv('%s%s.poisson.ZnBZ.csv' % (csv_folder, Cat_names[ii])))
    PBG_W.append(pd.read_csv('%s%s.poisson.TMC.csv' % (csv_folder, Cat_names[ii])))

    # AreaKM2n[ii] = polygon_area(deCLUST_Z[ii].Lat.values, deCLUST_Z[ii].Lon.values, radius = 6378137)
    deCLUST_W[ii] = define_Event_type(deCLUST_W[ii])
    deCLUST_Z[ii] = define_Event_type(deCLUST_Z[ii])

    I_bgW = deCLUST_W[ii]['eType'] != 1.0
    I_bgZ = deCLUST_Z[ii]['eType'] != 1.0

    # [AreaKM2_W[ii], N_W, X, Y,xyz,I_NW] = Calc_Area(deCLUST_W[ii].Lon[I_bgW].values,deCLUST_W[ii].Lat[I_bgW].values,dxy,rMaxC)
    # [AreaKM2_Z[ii], N_Z, X, Y,xyz,I_NZ] = Calc_Area(deCLUST_Z[ii].Lon[I_bgZ].values,deCLUST_Z[ii].Lat[I_bgZ].values,dxy,rMaxC)
    #
    #
    # axW = figAreaW.add_subplot(4,2,ii+1)
    # axZ = figAreaZ.add_subplot(4,2,ii+1)
    # plotSeisArea(axW, I_NW, X,Y, dxy, AreaKM2_W[ii], deCLUST_W[ii].Lon[I_bgW].values, deCLUST_W[ii].Lat[I_bgW].values, CL, Cat_Label[ii])
    # plotSeisArea(axZ, I_NZ, X,Y, dxy, AreaKM2_Z[ii], deCLUST_Z[ii].Lon[I_bgZ].values, deCLUST_Z[ii].Lat[I_bgZ].values, CL, Cat_Label[ii])

    n_mainshocks[ii, 0] = sum((CLUST_Z[ii].c_type == 1) & (CLUST_Z[ii].m0 >= 5.0))
    n_mainshocks[ii, 2] = sum((CLUST_M[ii].c_type == 1) & (CLUST_M[ii].m0 >= 5.0))
    n_mainshocks[ii, 1] = sum((CLUST_W[ii].c_type == 1) & (CLUST_W[ii].m0 >= 5.0))

    k_data_all, k_data_fs, k_data_nofs = calc_k_clust(CLUST_Z[ii], Mc, dM)
    k_mainshocks_fs[ii, 0] = k_data_fs.k
    k_mainshocks_all[ii, 0] = k_data_all.k
    k_data_all, k_data_fs, k_data_nofs = calc_k_clust(CLUST_M[ii], Mc, dM)
    k_mainshocks_fs[ii, 2] = k_data_fs.k
    k_mainshocks_all[ii, 2] = k_data_all.k
    k_data_all, k_data_fs, k_data_nofs = calc_k_clust(CLUST_W[ii], Mc, dM)
    k_mainshocks_fs[ii, 1] = k_data_fs.k
    k_mainshocks_all[ii, 1] = k_data_all.k

print(AreaKM2_W)
print(AreaKM2_Z)

fper_Z = []
fper_M = []
fper_W = []
fig = plb.figure(7600)
for fi in range(len(fsTypes)):
# for fi in range(1):
    ax1 = fig.add_subplot(4,len(fsTypes),fi+1)
    nfh0[0],nfh1[0],nfh5[0],nfh0std[0],nfh1std[0],nfh5std[0], f_nof_Z, fper_Z0 = plotFSbar(CLUST_Z, ax1,Cat_Label,fsTypes[fi])
    fper_Z.append(fper_Z0)
    ax1.set_ylabel('ZnBZ')
    ax1.set_title(fsTypes1[fi])
    ax1.set_ylim([0,100])

    ax2 = fig.add_subplot(4,len(fsTypes),len(fsTypes)*1+fi+1)
    #nfh0[1],nfh1[1],nfh5[1],nfh0std[1],nfh1std[1],nfh5std[1] = plotFSbar(CLUST_G, ax2,Cat_Label,fsTypes[fi])
    nfh0[1],nfh1[1],nfh5[1],nfh0std[1],nfh1std[1],nfh5std[1], f_nof_W, fper_W0 = plotFSbar(CLUST_W, ax2,Cat_Label,fsTypes[fi])
    fper_W.append(fper_W0)
    # ax2.set_ylabel('GnK')
    ax2.set_ylabel('TMC')
    ax2.set_ylim([0,100])

    ax3 = fig.add_subplot(4,len(fsTypes),len(fsTypes)*2+fi+1)
    nfh0[2],nfh1[2],nfh5[2], nfh0std[2], nfh1std[2],nfh5std[2], f_nof_M, fper_M0 = plotFSbar(CLUST_M, ax3,Cat_Label,fsTypes[fi])
    fper_M.append(fper_M0)
    ax3.set_ylabel('WnC')
    ax3.set_ylim([0,100])

    ax = fig.add_subplot(4,len(fsTypes),len(fsTypes)*3+fi+1)
    width = 0.35
    colors = ['dodgerblue','skyblue','whitesmoke']
    Methods = ['ZnBZ', 'TMC', 'WnC']
    ax.bar(Methods, nfh5, width, yerr=nfh5std, color=colors[0],edgecolor='black')#, colors=colors[0]) # shadow=True,
    ax.bar(Methods, nfh1, width,yerr=nfh1std, bottom=nfh5, color=colors[1], edgecolor='black')
    ax.bar(Methods, nfh0, width,yerr=nfh0std, bottom=nfh5+nfh1, color=colors[2], edgecolor='black')
    # for ii in range(len(Methods)):
        # ax.text(x=ii, y=nfh5[ii]+1 , s='%d' % np.round(nfh5[ii]) , fontdict=dict(fontsize=10), horizontalalignment ='left')
        # ax.text(x=ii, y=nfh1[ii]+nfh5[ii]+1, s='%d' % np.round(nfh1[ii]+nfh5[ii]), fontdict=dict(fontsize=10), horizontalalignment ='left')
    plb.grid(axis='y', linestyle='--')
    ax.set_ylim([0,100])
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)


if iGCMT > -1:
    fig = plb.figure(7610)
    ax11 = fig.add_subplot(3,2,1)
    plotFMS(CLUST_Z[iGCMT], ax11)
    ax11.set_ylabel('ZnBZ')

    ax12 = fig.add_subplot(3,2,3)
    plotFMS(CLUST_M[iGCMT], ax12)
    ax12.set_ylabel('WnC')

    ax13 = fig.add_subplot(3,2,5)
    plotFMS(CLUST_W[iGCMT], ax13)
    ax13.set_ylabel('TMC')

BG_W = np.zeros(len(Cat_names))
BG_M = np.zeros(len(Cat_names))
BG_Z = np.zeros(len(Cat_names))
for ii in range(len(Cat_names)):
    BG_W[ii] = sum(deCLUST_W[ii].clID == 0) / len(deCLUST_W[ii].clID) * 100
    BG_M[ii] = sum(deCLUST_M[ii].clID == 0) / len(deCLUST_M[ii].clID) * 100
    BG_Z[ii] = sum(deCLUST_Z[ii].clID == 0) / len(deCLUST_Z[ii].clID) * 100

LamdaZ = np.zeros(len(Cat_names))
LamdaW = np.zeros(len(Cat_names))
LamdaZstd = np.zeros(len(Cat_names))
LamdaWstd = np.zeros(len(Cat_names))
for ii in range(len(Cat_Label)):
    LamdaZ[ii] = PBG_Z[ii].Lambda.values[0]
    LamdaW[ii] = PBG_W[ii].Lambda.values[0]
    LamdaZstd[ii] = PBG_Z[ii].stdBG.values[0]
    LamdaWstd[ii] = PBG_W[ii].stdBG.values[0]

fig = plb.figure(7605)
plot_val_2cat(fig.add_subplot(3,3,1),Cat_Label, BG_Z,BG_W,'ZnBZ','TMC','Background (%)')
plot_val_2NF(fig.add_subplot(3,3,2), fper_Z[0], BG_Z,'ZnBZ', fper_W[0], BG_W, 'TMC', Cat_Label, 'Background (%)', 'Mainshocks* with \nforeshocks (%)', np.zeros(len(Cat_names)),np.zeros(len(Cat_names)))
plot_compare2(fig.add_subplot(3,3,3), BG_Z, BG_W, Cat_Label, 'Background ZnBZ (%)','Background TMC (%)')
plot_val_2cat(fig.add_subplot(3,3,4),Cat_Label, LamdaZ,LamdaW,'ZnBZ','TMC','Lambda')
plot_val_2NF(fig.add_subplot(3,3,5), fper_Z[0], LamdaZ, 'ZnBZ', fper_W[0], LamdaW, 'TMC', Cat_Label, 'Lambda', 'Mainshocks* with \n foreshocks (%)', LamdaZstd, LamdaWstd)
plot_compare2(fig.add_subplot(3,3,6), LamdaZ, LamdaW, Cat_Label, 'Lambda ZnBZ','Lambda TMC')
plot_val_2cat(fig.add_subplot(3,3,7),Cat_Label, LamdaZ/AreaKM2_Z,LamdaW/AreaKM2_Z,'ZnBZ', 'TMC', 'Lambda/km2')
plot_val_2NF(fig.add_subplot(3,3,8), fper_Z[0], LamdaZ/AreaKM2_Z,'ZnBZ', fper_W[0], LamdaW/AreaKM2_Z, 'TMC', Cat_Label, 'Lambda/km2', 'Mainshocks* with \n foreshocks (%)', LamdaZstd,LamdaWstd)
plot_compare2(fig.add_subplot(3,3,9), LamdaZ/AreaKM2_W, LamdaW/AreaKM2_W, Cat_Label, 'Lambda ZnBZ /km2','Lambda TMC /km2')

fig = plb.figure(7606)
plot_val_2cat(fig.add_subplot(3,3,1),Cat_Label, BG_Z,BG_W,'ZnBZ','TMC','Background (%)')
plot_val_2NF(fig.add_subplot(3,3,2), fper_Z[2], BG_Z,'ZnBZ', fper_W[2], BG_W, 'TMC', Cat_Label, 'Background (%)', 'Mainshocks* with \n foreshocks2.0 (%)', np.zeros(len(Cat_names)),np.zeros(len(Cat_names)))
plot_compare2(fig.add_subplot(3,3,3), BG_Z, BG_W, Cat_Label, 'Background ZnBZ (%)','Background TMC (%)')
plot_val_2cat(fig.add_subplot(3,3,4),Cat_Label, LamdaZ,LamdaW,'ZnBZ','TMC','Lambda')
plot_val_2NF(fig.add_subplot(3,3,5), fper_Z[2], LamdaZ, 'ZnBZ', fper_W[2], LamdaW, 'TMC', Cat_Label, 'Lambda', 'Mainshocks* with \n foreshocks2.0 (%)', LamdaZstd, LamdaWstd)
plot_compare2(fig.add_subplot(3,3,6), LamdaZ, LamdaW, Cat_Label, 'Lambda ZnBZ','Lambda TMC')
plot_val_2cat(fig.add_subplot(3,3,7),Cat_Label, LamdaZ/AreaKM2_Z,LamdaW/AreaKM2_Z,'ZnBZ', 'TMC', 'Lambda/km2')
plot_val_2NF(fig.add_subplot(3,3,8), fper_Z[2], LamdaZ/AreaKM2_Z,'ZnBZ', fper_W[2], LamdaW/AreaKM2_Z, 'TMC', Cat_Label, 'Lambda/km2', 'Mainshocks* with \n foreshocks2.0 (%)', LamdaZstd,LamdaWstd)
plot_compare2(fig.add_subplot(3,3,9), LamdaZ/AreaKM2_W, LamdaW/AreaKM2_W, Cat_Label, 'Lambda ZnBZ /km2','Lambda TMC /km2')



fig = plb.figure(7608)
plot_val_2NF(fig.add_subplot(1, 2, 1), k_mainshocks_all[:, 0], LamdaZ/AreaKM2_Z, 'ZnBZ', k_mainshocks_all[:, 1], LamdaW/AreaKM2_Z, 'TMC', Cat_Label, 'Lambda/km2', 'Aftershocks productivity (k) all', LamdaZstd, LamdaWstd)
plot_val_2NF(fig.add_subplot(1, 2, 2), k_mainshocks_fs[:, 0], LamdaZ/AreaKM2_Z, 'ZnBZ', k_mainshocks_fs[:, 1], LamdaW/AreaKM2_Z, 'TMC', Cat_Label, 'Lambda/km2', 'Aftershocks productivity (k) with FS', LamdaZstd, LamdaWstd)



fig11 = plb.figure(11)
ax11 = fig11.add_subplot(1,1,1)
plot_BG_NF(ax11,fper_Z[2], BG_Z,Cat_Label)
plot_BG_NF(ax11,fper_M[2], BG_M,Cat_Label)
plot_BG_NF(ax11,fper_W[2], BG_W,Cat_Label)

fig22 = plb.figure(22)
ax22_1 = fig22.add_subplot(3,2,1)
plot_BG_Lamda(ax22_1, PBG_Z, BG_Z, Cat_names,'ZnBZ')

ax22_2 = fig22.add_subplot(3,2,2)
plot_BG_Lamda(ax22_2, PBG_W, BG_Z, Cat_names, 'TMC')

ax22_3 = fig22.add_subplot(3,2,3)
plot_FS_Lamda(ax22_3, PBG_Z, fper_Z[2],Cat_names, 'ZnBZ')

ax22_4 = fig22.add_subplot(3,2,4)
plot_FS_Lamda(ax22_4, PBG_W, fper_W[2],Cat_names, 'TMC')

ax22_5 = fig22.add_subplot(3,2,5)
plot_compare2(ax22_5, LamdaZ, LamdaW, Cat_names, 'Lambda ZnBZ', 'Lambda TMC')

fig8 = plb.figure(877)
plot_mainshocks_rate(fig8.add_subplot(2, 3, 1), n_mainshocks, cat_years, Cat_Label, 'Mainshocks* per year')
plot_val_2NF(fig8.add_subplot(2, 3, 4), fper_Z[2], n_mainshocks[:, 0]/cat_years, 'ZnBZ', fper_W[2], n_mainshocks[:, 1]/cat_years, 'TMC', Cat_Label, 'Mainshock* per year', 'Mainshocks with foreshocks (%)', np.zeros(len(Cat_names)), np.zeros(len(Cat_names)))

for ii in range(3):
    n_mainshocks_a[:, ii] = n_mainshocks[:, ii] / AreaKM2_Z
plot_mainshocks_rate(fig8.add_subplot(2, 3, 2), n_mainshocks_a, cat_years, Cat_Label, 'Mainshocks* per year per km2')
plot_val_2NF(fig8.add_subplot(2, 3, 5), fper_Z[2], n_mainshocks_a[:, 0]/cat_years, 'ZnBZ', fper_W[2], n_mainshocks_a[:, 1]/cat_years, 'TMC', Cat_Label, 'Mainshock* per year per km2', 'Mainshocks with foreshocks (%)', np.zeros(len(Cat_names)), np.zeros(len(Cat_names)))

plot_val_2NF(fig8.add_subplot(2, 3, 3), n_mainshocks[:, 0]/cat_years, LamdaZ, 'ZnBZ', n_mainshocks[:, 1]/cat_years, LamdaW, 'TMC', Cat_Label, 'Lambda', 'Mainshocks* per year', np.zeros(len(Cat_names)), np.zeros(len(Cat_names)))
plot_val_2NF(fig8.add_subplot(2, 3, 6), n_mainshocks_a[:, 0]/cat_years, LamdaZ/AreaKM2_Z, 'ZnBZ', n_mainshocks_a[:, 1]/cat_years, LamdaW/AreaKM2_W, 'TMC', Cat_Label, 'Lambda#', 'Mainshocks* per year per km2', np.zeros(len(Cat_names)), np.zeros(len(Cat_names)))


print('DONE!')
plb.show()
