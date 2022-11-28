import pandas as pd
import mpl_toolkits
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from obspy import UTCDateTime
import numpy as np
from obspy.imaging.beachball import beach, aux_plane
import matplotlib.pyplot as plb
import datetime
from geopy import distance
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from matplotlib import path, cm
from Foreshocks_Aftershocks_productivity import *
import matplotlib as mpl
import matplotlib.patches as mpatches
import time
import cartopy
mpl.rcParams['pdf.fonttype'] = 42
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore")

def make_time_vector(CLUST):
    # CLUST = CLUST.reset_index()
    ll = len(CLUST)
    t0 = np.zeros(ll)
    for ii in range(ll):
        t0[ii] = UTCDateTime(CLUST['ot0'][ii])
    return t0


def make_comparison_tble4(CLUST_Z, CLUST_Zb, CLUST_M, CLUST_G):
    ll = len(CLUST_Z)

    t0_M      = make_time_vector(CLUST_M)
    t0_Z      = make_time_vector(CLUST_Z)
    t0_Zb     = make_time_vector(CLUST_Zb)
    t0_G      = make_time_vector(CLUST_G)

    M0        = []
    T0Z       = []
    T0Zb       = []
    T0M       = []
    T0G       = []
    Lat0      = []
    Lon0      = []
    Dep0      = []
    Cid_Z     = []
    Cid_Zb    = []
    Cid_M     = []
    Cid_G     = []
    N_AS_allZ = []
    N_FS_allZ = []
    N_AS_allZb = []
    N_FS_allZb = []
    N_FS_allG = []
    N_AS_allG = []
    N_AS_Z    = []
    N_FS_Z    = []
    N_AS_Zb    = []
    N_FS_Zb    = []
    N_AS_M    = []
    N_FS_M    = []
    N_AS_G    = []
    N_FS_G    = []
    c_typeM   = []
    c_typeZ   = []
    c_typeZb   = []
    c_typeG   = []
    c_typeallZ= []
    c_typeallZb= []
    c_typeallG= []
    R_predict_M = []
    R_predict_Z = []
    R_M = []
    R_Z = []
    

    for ii in range(ll):
        if CLUST_Z['m0'][ii] > min(CLUST_M['m0']):
            if np.min(np.abs(t0_Z[ii] - t0_M)) <= 2.0:
                pos = np.argmin(np.abs(t0_Z[ii] - t0_M))
                posG = np.argmin(np.abs(t0_Z[ii] - t0_G))
                posZb = np.argmin(np.abs(t0_Z[ii] - t0_Zb))
                if np.logical_and(CLUST_Z['m0'][ii] == CLUST_M['m0'][pos],CLUST_Z['m0'][ii] == CLUST_G['m0'][posG]):
                    # print('%d / %d' % (ii, ll))
                    M0.append(CLUST_Z['m0'][ii])
                    T0Zb.append(CLUST_Zb['ot0'][posZb])
                    T0Z.append(CLUST_Z['ot0'][ii])
                    T0M.append(CLUST_M['ot0'][pos])
                    T0G.append(CLUST_G['ot0'][posG])
                    Lat0.append(CLUST_Z['lat0'][ii])
                    Lon0.append(CLUST_Z['lon0'][ii])
                    Dep0.append(CLUST_Z['depth0'][ii])
                    Cid_Z.append(CLUST_Z['cid'][ii])
                    Cid_Zb.append(CLUST_Zb['cid'][posZb])
                    Cid_M.append(CLUST_M['cid'][pos])
                    Cid_G.append(CLUST_G['cid'][posG])
                    N_AS_allZ.append(np.int(CLUST_Z['na_all'][ii]))
                    N_FS_allZ.append(np.int(CLUST_Z['nf_all'][ii]))
                    N_AS_allZb.append(np.int(CLUST_Zb['na_all'][posZb]))
                    N_FS_allZb.append(np.int(CLUST_Zb['nf_all'][posZb]))
                    N_AS_allG.append(np.int(CLUST_G['na_all'][posG]))
                    N_FS_allG.append(np.int(CLUST_G['nf_all'][posG]))
                    N_AS_Z.append(np.int(CLUST_Z['na'][ii]))
                    N_FS_Z.append(np.int(CLUST_Z['nf'][ii]))
                    N_AS_Zb.append(np.int(CLUST_Zb['na'][posZb]))
                    N_FS_Zb.append(np.int(CLUST_Zb['nf'][posZb]))
                    N_AS_M.append(np.int(CLUST_M['na'][pos]))
                    N_FS_M.append(np.int(CLUST_M['nf'][pos]))
                    N_AS_G.append(np.int(CLUST_G['na'][posG]))
                    N_FS_G.append(np.int(CLUST_G['nf'][posG]))
                    c_typeZ.append(np.int(CLUST_Z['c_type60'][ii]))
                    c_typeZb.append(np.int(CLUST_Zb['c_type60'][posZb]))
                    c_typeG.append(np.int(CLUST_G['c_type60'][posG]))
                    c_typeallZ.append(np.int(CLUST_Z['c_type'][ii]))
                    c_typeallZb.append(np.int(CLUST_Zb['c_type'][posZb]))
                    c_typeallG.append(np.int(CLUST_G['c_type'][posG]))
                    c_typeM.append(np.int(CLUST_M['c_type60'][pos]))

    CLUST = {'M0':M0, 'T0Z':T0Z,'T0Zb':T0Zb, 'T0M':T0M,'T0G':T0G,'Lat0':Lat0, 'Lon0':Lon0, 'Dep0':Dep0,
             'Cid_Z':Cid_Z,'Cid_Zb':Cid_Zb,'Cid_G':Cid_G, 'Cid_M':Cid_M, 'N_AS_allZ':N_AS_allZ,'N_AS_allZb':N_AS_allZb,
             'N_FS_allZ':N_FS_allZ,'N_FS_allZb':N_FS_allZb, 'N_AS_Z':N_AS_Z,'N_AS_Zb':N_AS_Zb, 'N_FS_Z':N_FS_Z,
             'N_FS_Zb':N_FS_Zb, 'N_AS_allG':N_AS_allG,'N_FS_allG':N_FS_allG, 'N_AS_G':N_AS_G, 'N_FS_G':N_FS_G,
             'N_AS_M':N_AS_M, 'N_FS_M':N_FS_M, 'c_typeM':c_typeM,'c_typeZ':c_typeZ, 'c_typeallZ':c_typeallZ,
             'c_typeZb':c_typeZb, 'c_typeallZb':c_typeallZb,'c_typeG':c_typeG, 'c_typeallG':c_typeallG}
    CLUST = pd.DataFrame(data=CLUST)
    print('Found %d events out of %d' % (len(M0), ll))
    return CLUST

def make_comparison_tble3(CLUST_Z, CLUST_M, CLUST_G):
    
    MminM = min(CLUST_M['m0'].values)
    CLUST_Z = CLUST_Z[CLUST_Z['m0'].values >= MminM].reset_index(drop=True)
    CLUST_G = CLUST_G[CLUST_G['m0'].values >= MminM].reset_index(drop=True)
    CLUST_M = CLUST_M.reset_index(drop=True)
    
    
    ll = len(CLUST_Z)

    t0_M      = make_time_vector(CLUST_M)
    t0_Z      = make_time_vector(CLUST_Z)
    t0_G      = make_time_vector(CLUST_G)

    M0        = []
    T0Z       = []
    T0M       = []
    T0G       = []
    Lat0      = []
    Lon0      = []
    Dep0      = []
    Cid_Z     = []
    Cid_M     = []
    Cid_G     = []
    N_AS_allZ = []
    N_FS_allZ = []
    N_FS_allG = []
    N_AS_allG = []
    N_AS_Z    = []
    N_FS_Z    = []
    N_AS_M    = []
    N_FS_M    = []
    N_AS_G    = []
    N_FS_G    = []
    c_typeM   = []
    c_typeZ   = []
    c_typeG   = []
    c_typeallZ= []
    c_typeallG= []
    R_predict_M = []
    R_predict_Z = []
    R_M = []
    R_Z = []
    

    for ii in range(ll):
        # min magnitude condition
        if CLUST_Z['m0'][ii] > MminM:
            # time condition
            if np.logical_and(np.min(np.abs(t0_Z[ii] - t0_M)) <= 5.0, np.min(np.abs(t0_Z[ii] - t0_G)) <= 5.0):
                pos = np.argmin(np.abs(t0_Z[ii] - t0_M))
                posG = np.argmin(np.abs(t0_Z[ii] - t0_G))

                # magnitude condition
                if np.logical_and(CLUST_Z['m0'][ii].round(1) == CLUST_M['m0'][pos].round(1), CLUST_Z['m0'][ii].round(1) == CLUST_G['m0'][posG].round(1)):
                    # print('%d / %d' % (ii, ll))
                    M0.append(CLUST_Z['m0'][ii])
                    T0Z.append(CLUST_Z['ot0'][ii])
                    T0M.append(CLUST_M['ot0'][pos])
                    T0G.append(CLUST_G['ot0'][posG])
                    Lat0.append(CLUST_Z['lat0'][ii])
                    Lon0.append(CLUST_Z['lon0'][ii])
                    Dep0.append(CLUST_Z['depth0'][ii])
                    Cid_Z.append(CLUST_Z['cid'][ii])
                    Cid_M.append(CLUST_M['cid'][pos])
                    Cid_G.append(CLUST_G['cid'][posG])
                    N_AS_allZ.append(np.int(CLUST_Z['na_all'][ii]))
                    N_FS_allZ.append(np.int(CLUST_Z['nf_all'][ii]))
                    N_AS_allG.append(np.int(CLUST_G['na_all'][posG]))
                    N_FS_allG.append(np.int(CLUST_G['nf_all'][posG]))
                    N_AS_Z.append(np.int(CLUST_Z['na'][ii]))
                    N_FS_Z.append(np.int(CLUST_Z['nf'][ii]))
                    N_AS_M.append(np.int(CLUST_M['na'][pos]))
                    N_FS_M.append(np.int(CLUST_M['nf'][pos]))
                    N_AS_G.append(np.int(CLUST_G['na'][posG]))
                    N_FS_G.append(np.int(CLUST_G['nf'][posG]))
                    c_typeZ.append(np.int(CLUST_Z['c_type60'][ii]))
                    c_typeG.append(np.int(CLUST_G['c_type60'][posG]))
                    c_typeallZ.append(np.int(CLUST_Z['c_type'][ii]))
                    c_typeallG.append(np.int(CLUST_G['c_type'][posG]))
                    c_typeM.append(np.int(CLUST_M['c_type60'][pos]))

    CLUST = {'M0':M0, 'T0Z':T0Z,'T0M':T0M,'T0G':T0G,'Lat0':Lat0, 'Lon0':Lon0, 'Dep0':Dep0,
             'Cid_Z':Cid_Z,'Cid_G':Cid_G, 'Cid_M':Cid_M, 'N_AS_allZ':N_AS_allZ,
             'N_FS_allZ':N_FS_allZ,'N_AS_Z':N_AS_Z,'N_FS_Z':N_FS_Z,
             'N_AS_allG':N_AS_allG,'N_FS_allG':N_FS_allG, 'N_AS_G':N_AS_G, 'N_FS_G':N_FS_G,
             'N_AS_M':N_AS_M, 'N_FS_M':N_FS_M, 'c_typeM':c_typeM,'c_typeZ':c_typeZ, 'c_typeallZ':c_typeallZ,
             'c_typeG':c_typeG, 'c_typeallG':c_typeallG}
    CLUST = pd.DataFrame(data=CLUST)
    print('Found %d events out of %d' % (len(M0), ll))
    return CLUST

def year_fraction(dates):
    years = np.zeros(len(dates))
    for ii in range(len(years)):
        date = UTCDateTime(dates[ii])
        start = datetime.date(date.year, 1, 1).toordinal()
        year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
        years[ii] = date.year + float(date.toordinal() - start) / year_length
    return years


def stemW(ax,x,y,c,s,lbl):

    for ii in range(len(x)):
        ax.plot([x[ii],x[ii]],[0,y[ii]],c='grey',alpha=0.8)
    ax.scatter(x,y,edgecolors=c, facecolors='w',marker=s,label=lbl)


def plotM0timeLine(CLUST_Z, CLUST_M, CLUST_G):
    MminM = min(CLUST_M['m0'].values)
    CLUST_Z = CLUST_Z[CLUST_Z['m0'].values >= MminM].reset_index(drop=True)
    CLUST_G = CLUST_G[CLUST_G['m0'].values >= MminM].reset_index(drop=True)
    CLUST_M = CLUST_M.reset_index(drop=True)

    t0_M      = year_fraction(make_time_vector(CLUST_M))
    t0_Z      = year_fraction(make_time_vector(CLUST_Z))
    t0_G      = year_fraction(make_time_vector(CLUST_G))
    tmin = min([min(t0_M), min(t0_G),min(t0_Z)])
    tmax = max([max(t0_M), max(t0_G),max(t0_Z)])

    fig = plb.figure(7009)

    ax1 = fig.add_subplot(3,1,1)
    stemW(ax1,t0_Z,CLUST_Z.m0,'m','o','ZnBZ')
    ax1.set_xlim(tmin,tmax)
    ax1.set_ylim(MminM-1,max(CLUST_Z.m0) + 0.5)
    ax1.legend(loc='upper left')
    ax1.grid()

    ax2 = fig.add_subplot(3,1,2)
    stemW(ax2,t0_M,CLUST_M.m0,'m','o','WnC')
    ax2.set_xlim(tmin,tmax)
    ax2.set_ylim(MminM-1,max(CLUST_M.m0) + 0.5)
    ax2.legend(loc='upper left')
    ax2.grid()

    ax3 = fig.add_subplot(3,1,3)
    stemW(ax3,t0_G,CLUST_G.m0,'m','o','TMC')
    ax3.set_xlim(tmin,tmax)
    ax3.set_ylim(MminM-1,max(CLUST_G.m0) + 0.5)
    ax3.legend(loc='upper left')
    ax3.grid()


    
#----------------PLOTS--------------------
def plotworldloc5(ax, Lon0, Lat0):
    geo = ccrs.Geodetic()
    ax.coastlines(resolution='50m')
    ax.scatter(Lon0, Lat0, 70, marker='*', facecolors='m', edgecolor='k', linewidth=0.25, zorder=10, transform=geo)
    # fnames = ['ridge', 'trench', 'transform']
    # c = 'gbr'
    # for pp in range(len(fnames)):
    #     shapfile1 = '/Users/nadavwetzler/Dropbox/ARCMAP files/Global_TL_Faults/%s.shp' % fnames[pp]
    #     pltshape(ax, shapfile1, c[pp], ccrs=ccrs.Orthographic())





def plotIImap(ax, clust, type, folderF):
    x, y = shape2xy2(folderF + 'shapefiles/EW_Pacific.shp')
    nFS = 1

    m = clust['m0'].values
    na = clust['na_all_apr'].values
    nf = clust['nf_all_apr'].values
    
    Iinter = clust['InterIntra'] == 1
    Iintra = clust['InterIntra'] == -1

    x[0] = Make360(x[0])
    x[1] = Make360(x[1])

    m_lon = Make360(clust.lon0.values)
    m_lat = clust.lat0.values

    Iw = inpolygon(m_lon, m_lat, x[0], y[0])
    Ie = inpolygon(m_lon, m_lat, x[1], y[1])

    Idm = clust['c_type'].values == 1
    IyesF = np.logical_and(nf >= nFS, Idm)
    InoF = nf == 0

    Iw = np.logical_and(Iw, Idm)
    Ie = np.logical_and(Ie, Idm)

    IwF = np.logical_and(Iw, IyesF)
    IeF = np.logical_and(Ie, IyesF)

    # print('West Pacific Forshocks %d perc.' % (sum(IwF) / sum(Iw)*100))
    # print('East Pacific Forshocks %d perc.' % (sum(IeF) / sum(Ie)*100))

    # ax.set_extent([max(m_lon[Ie])+5, min(m_lon[Iw])-5, min(m_lat)-5, max(m_lat)+5], crs=ccrs.PlateCarree())
    ax.set_extent([359, 1, -80, 80], crs=ccrs.PlateCarree())
    ax.stock_img()

    # Plot All
    ax.scatter(m_lon[Idm & Iinter], m_lat[Idm & Iinter], 10, zorder=15, facecolors='m', label='Interpalte', transform=ccrs.PlateCarree())
    ax.scatter(m_lon[Idm & Iintra], m_lat[Idm & Iintra], 25, zorder=15, facecolors='none', edgecolors='k', label='Intraplate', transform=ccrs.PlateCarree())
    ax.plot(x[0], y[0], transform=ccrs.PlateCarree())
    ax.plot(x[1], y[1], transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
    set_legend_title(ax, type, 14, 'Impact', 'upper right')

    


def plotworldloc4(ax, Lon0, Lat0, dlatlon):
    minLat = max([-90, Lat0-dlatlon])
    maxLat = min([90, Lat0+dlatlon])
    Lons = [Lon0 - dlatlon, Lon0 + dlatlon]

    ax.set_xlim([min(Lons), max(Lons)])
    ax.set_ylim([minLat, maxLat])

    fnames = ['ridge', 'trench', 'transform']
    c = 'gbr'
    for pp in range(len(fnames)):
        shapfile1 = '/Users/nadavwetzler/Dropbox/ARCMAP files/Global_TL_Faults/%s.shp' % fnames[pp]
        pltshape(ax, shapfile1, c[pp])
    ax.scatter(Lon0, Lat0, 70, marker='*', facecolors='m', edgecolor='k', linewidth=0.25, zorder=10)
    ax.osm = OSM(ax)
    # ax.coastlines()

def plotworldloc3(ax, Lon0, Lat0, dlatlon):
    minLat = max([-90, Lat0-dlatlon])
    maxLat = min([90, Lat0+dlatlon])
    Lons = [Lon0 - dlatlon, Lon0 + dlatlon]
    Lons = Make360(Lons)
    if Lons[0] > Lons[1]:
        Lons = [Lon0 - dlatlon, Lon0 + dlatlon]
    ax.set_extent([min(Lons), max(Lons), minLat, maxLat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    fnames = ['ridge', 'trench', 'transform']
    c = 'gbr'
    for pp in range(len(fnames)):
        shapfile1 = '/Users/nadavwetzler/Dropbox/ARCMAP files/Global_TL_Faults/%s.shp' % fnames[pp]
        pltshape(ax, shapfile1, c[pp], ccrs=ccrs.PlateCarree())
    ax.scatter(Lon0, Lat0, 70, marker='*', facecolors='m', edgecolor='k', linewidth=0.25, transform=ccrs.PlateCarree(), zorder=10)
    # ax.coastlines()

def plotworldloc2(ax, Lon0, Lat0, Lats, Longs):
    img = plb.imread('/Users/nadavwetzler/Dropbox/ARCMAP files/Global_TOPO/eo_base_2020_clean_3600x1800.png')
    img_extent = [min(Longs), max(Longs), min(Lats), max(Lats)]
    ax.set_extent(img_extent, crs=ccrs.Geodetic())
    ax.imshow(img, origin='upper', extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.stock_img()
    ax.scatter(Lon0, Lat0, 50, marker='*', color='m', transform=ccrs.PlateCarree())
    ax.coastlines()

def plotworldloc(ax, Lon0, Lat0):
    ax.set_extent([-180, 180, -80, 80], crs=ccrs.PlateCarree())
    ax.scatter(Lon0, y0, Lat0, marker='o', color='m', transform=ccrs.PlateCarree())
    ax.coastlines()

def plotMapC(ax, clust, Lats, Lons, Lon0, Lat0, ot, m0, ctype, cat, years, intra=False):
    from Foreshocks_Aftershocks_productivity import WnCfaultL, AdjustEffectiveRadius, makeCircle, MakeCircleUTM

    ylong = years*365  # years
    t0 = UTCDateTime(ot)

    # get maisnhocks FMS
    Idate = (cat.ot > t0-3600) & (cat.ot < t0+3600)
    Im = (cat.M > (m0 - 0.5)) & (cat.M < (m0 + 0.5))
    Ifms = Idate & Im
    fstyle = -1
    pos = 0
    try:
        fms = np.zeros(3)
        fms[0] = cat.strike[Ifms][0]
        fms[1] = cat.dip[Ifms][0]
        fms[2] = cat.rake[Ifms][0]
        # fstyle = int(cat.fstyle[Ifms][0])
        fms2 = aux_plane(fms[0], fms[1], fms[2])
        fstyle1, fstyle = ftypeShearer(fms[2], fms2[2])
        if intra == True:
            pos = InterIntra(fms, Lon0, Lat0, m0, t0.date)
    except:
        pass
    
    # catalog time
    # Ilon = np.logical_and(cat.Long > min(Lons), cat.Long < max(Lons))
    Ilon = np.logical_and(Make360(cat.Long) > Make360(min(Lons)), Make360(cat.Long) < Make360(max(Lons)))
    Ilat = np.logical_and(cat.Lat > min(Lats), cat.Lat < max(Lats))
    Ireg = np.logical_and(Ilat, Ilon)
    ItlongAS = np.logical_and(cat.ot > t0, cat.ot < t0 + ylong*3600*24)
    ItlongFS = np.logical_and(cat.ot < t0, cat.ot > t0 - ylong*3600*24)

    # ax.set_extent([min(Lons), max(Lons), min(Lats), max(Lats)], ccrs.Geodetic())
    ax.set_extent([min(Lons), max(Lons), min(Lats), max(Lats)])
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
    mtype = id2ctype(ctype)
    ax.scatter(Lon0, Lat0, 200, marker='*', color='m', alpha=1.0, zorder=2, label=mtype, transform=ccrs.Geodetic()) # label='Mainshock'
    if fstyle != -1:
        dllf = np.min([np.diff([min(Lons), max(Lons)]), np.diff([min(Lats), max(Lats)])])
        Long01 = Lon0 + np.cos(np.deg2rad(25)) * (dllf/5)
        Lat01 = Lat0 + np.sin(np.deg2rad(25)) * (dllf/5)
        ax.plot([Lon0, Long01], [Lat0, Lat01], linewidth=0.2, color='m', transform=ccrs.Geodetic())
        projection = ccrs.Mercator(central_longitude=180.0)
        x0, y0 = projection.transform_point(x=Long01, y=Lat01, src_crs=ccrs.Geodetic())
        cfms = 'krgb'
        b = beach(fms, xy=(x0, y0), width=dllf/15*100000, linewidth=0.1, facecolor=cfms[fstyle], zorder=10)
        ax.add_collection(b)
        
    # plot foreshocks/aftershocks from the entire catalog
    ax.scatter(cat.Long[np.logical_and(ItlongAS, Ireg)], cat.Lat[np.logical_and(ItlongAS, Ireg)], 20, marker='o', color='grey', zorder=2, alpha=0.5, transform=ccrs.PlateCarree())#  label='AS.Cat %d yr' % years
    ax.scatter(cat.Long[np.logical_and(ItlongFS, Ireg)], cat.Lat[np.logical_and(ItlongFS, Ireg)], 20, marker='^', color='grey', zorder=2, alpha=0.5, transform=ccrs.PlateCarree())#  label='FS.Cat %d yr' % years
    
    # cluster time
    Time = []
    for ii in range(len(clust['Time'])):
        Time.append(UTCDateTime(clust['Time'].values[ii]))
    Time = np.array(Time)
    Time = (Time - t0) / (3600*24)

    # plot foreshocks/aftershocks from the cluster
    selFS = Time < 0
    selAS = Time > 0
    x = clust['Lon'].values 
    y = clust['Lat'].values
    ax.scatter(x[selAS], y[selAS], 25, marker='o', facecolors='None', edgecolor='r', alpha=0.8, zorder=3, transform=ccrs.PlateCarree())# label='AS.Cl %d' % sum(selAS)
    ax.scatter(x[selFS], y[selFS], 25, marker='^', facecolors='None', edgecolor='b', alpha=0.8, zorder=4, transform=ccrs.PlateCarree())# label='FS.Cl %d' % sum(selFS)
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
    
    # fnameshp = '/Users/nadavwetzler/Dropbox/ARCMAP files/tectonicplates-master/PB2002_boundaries.shp'
    # shape_feature = ShapelyFeature(Reader(fnameshp).geometries(), ccrs.PlateCarree(), facecolor='none')
    # ax.add_feature(shape_feature)
    
    fnames = ['ridge', 'trench', 'transform']
    c = 'gbr'
    for pp in range(len(fnames)):
        shapfile1 = '/Users/nadavwetzler/Dropbox/ARCMAP files/Global_TL_Faults/%s.shp' % fnames[pp]
        pltshape(ax, shapfile1, c[pp], ccrs=ccrs.PlateCarree())
    # ax.coastlines()

    if m0 > 0:
        [r_predicted, _] = WnCfaultL(m0, ifault=0)
        fact_r = AdjustEffectiveRadius(m0)
        r_predicted = fact_r * r_predicted
        [xr, yr, xyr] = MakeCircleUTM(r_predicted, Lon0, Lat0)
        xr = Make360(xr)
        ax.plot(xr, yr, c='m', transform=ccrs.PlateCarree())#  label='E.R.: %d km' % int(r_predicted / 1000)
    return pos

def scaleT(T, len, t0, t1):
        T = T[np.logical_and(T>t0, T<t1)] - t0
        T = T * (len / t1) + t0
        return T

def plotLatTimeClust(ax, clust, ot,Lat0, M0, cat, Lats,Lons):
    years = 1
    ylong = years*365 # years
    t0 = UTCDateTime(ot)
    T = (cat.ot - t0)/(3600*24)
    # catalog time
    Ilon = np.logical_and(cat.Long>min(Lons), cat.Long<max(Lons))
    Ilat = np.logical_and(cat.Lat>min(Lats), cat.Lat<max(Lats))
    Ireg = np.logical_and(Ilat, Ilon)
    It60 = np.logical_and(cat.ot > t0- 60*3600*24, cat.ot < t0 + 60*3600*24)
    ItlongAS = np.logical_and(cat.ot > t0 + 60*3600*24, cat.ot < t0 + (ylong)*3600*24)
    ItlongFS = np.logical_and(cat.ot < t0 - 60*3600*24, cat.ot > t0 - (ylong)*3600*24)

    # cluster time
    Time = []
    for ii in range(len(clust['Time'])):
        Time.append(UTCDateTime(clust['Time'].values[ii]))
    Time = np.array(Time)
    Time = (Time - t0) / (3600*24)

    selFS60 = np.logical_and(Time < 0, Time > -60)
    selFSlong = np.logical_and(Time <= -60, Time > -(ylong))
    selAS60 = np.logical_and(Time > 0, Time < 60)
    selASlong = np.logical_and(Time >= 60, Time < (ylong))


    ax.scatter(T[np.logical_and(It60,Ireg)], cat.Lat[np.logical_and(It60,Ireg)],15,marker='o', color='gray',alpha=0.6)
    ax.scatter(scaleT(T[np.logical_and(ItlongAS,Ireg)],30,60,ylong), cat.Lat[np.logical_and(ItlongAS,Ireg)],15,marker='o', color='gray',alpha=0.6)
    ax.scatter(-scaleT(-T[np.logical_and(ItlongFS,Ireg)],30,60,ylong), cat.Lat[np.logical_and(ItlongFS,Ireg)],15,marker='o', color='gray',alpha=0.6)

    ax.scatter(Time[selFS60], clust['Lat'][selFS60].values,s=25, facecolors='none', edgecolors='b',label='FS: %d' % sum(selFS60))
    ax.scatter(Time[selAS60], clust['Lat'][selAS60].values,s=25, facecolors='none', edgecolors='r',label='AS: %d' % sum(selAS60))
    ax.scatter(-scaleT(-Time[selFSlong],30,60,ylong), clust['Lat'][selFSlong].values,s=25, edgecolors='b', facecolors='none')
    ax.scatter(scaleT(Time[selASlong],30,60,ylong), clust['Lat'][selASlong].values,s=25, edgecolors='r', facecolors='none')
    # ax.scatter(np.log10(Time[selFS]), clust['Lat'][selFS].values,s=25, facecolors='none', edgecolors='b',label='FS: %d' % sum(np.logical_and(Time < 0, Time > -60)))
    # ax.scatter(-np.log10(Time[selAS]), clust['Lat'][selAS].values,s=25, facecolors='none', edgecolors='r',label='AS: %d' % sum(np.logical_and(Time > 0, Time < 60)))
    ax.scatter(0, Lat0, 100, marker='*', color='m',alpha=1.0)
    ax.plot([30,30],Lats,'--k')
    ax.plot([-30,-30],Lats,'--k')
    ax.plot([60,60],Lats,'-k')
    ax.plot([-60,-60],Lats,'-k')
    ax.set_xlim([-90, 90])
    ax.set_xticks([-90,-60,-30,0,30,60,90])
    ax.set_xticklabels(['-%d years' % years,'-60d','-30d','ot','30d','60d','%d years' % years])
    ax.set_ylim(Lats)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time from t0 (days)')
    ax.set_ylabel('Latitude')


    
def plotR_TimeClust(ax, clust, ot,Lat0, Lon0, M0, cat, Lats, Lons, years):
    # years = 1
    ylong = years*365 # years
    t0 = UTCDateTime(ot)
    T = (cat.ot - t0)/(3600*24)
    # catalog time
    Ilon = np.logical_and(cat.Long > min(Lons), cat.Long < max(Lons))
    # Ilon = np.logical_and(Make360(cat.Long) > Make360(min(Lons)), Make360(cat.Long) < Make360(max(Lons)))
    Ilat = np.logical_and(cat.Lat > min(Lats), cat.Lat < max(Lats))
    Ireg = np.logical_and(Ilat, Ilon)
    It60FS = np.logical_and(cat.ot < t0, cat.ot > t0 - 60*3600*24)
    It60AS = np.logical_and(cat.ot > t0, cat.ot < t0 + 60*3600*24)
    ItlongAS = np.logical_and(T > 60, T < ylong)
    ItlongFS = np.logical_and(T < -60, T > -ylong)
    
    R = DistLatLonUTM(Lat0, Lon0, cat.Lat, Make360(cat.Long))
    Rc = DistLatLonUTM(Lat0, Lon0, clust.Lat.values, clust.Lon.values)
    [r_predicted, _] = WnCfaultL(M0,ifault=0)
    fact_r = AdjustEffectiveRadius(M0)
    r_predicted = np.round(fact_r * r_predicted/1000) * 1000

    # cluster time
    Time = []
    for ii in range(len(clust['Time'])):
        Time.append(UTCDateTime(clust['Time'].values[ii]))
    Time = np.array(Time)
    Time = (Time - t0) / (3600*24)

    selFS60 = np.logical_and(Time < 0, Time > -60)
    selFSlong = np.logical_and(Time <= -60, Time > -ylong)
    selAS60 = np.logical_and(Time > 0, Time < 60)
    selASlong = np.logical_and(Time >= 60, Time < ylong)

    R = R / 1000
    Rc = Rc / 1000
    ax.scatter(T[np.logical_and(It60AS, Ireg)], R[np.logical_and(It60AS, Ireg)], 15, marker='o', color='gray', alpha=0.6)
    ax.scatter(T[np.logical_and(It60FS, Ireg)], R[np.logical_and(It60FS, Ireg)], 15, marker='^', color='gray', alpha=0.6)
    ax.scatter(scaleT(T[np.logical_and(ItlongAS, Ireg)], 30, 60, ylong), R[np.logical_and(ItlongAS, Ireg)], 15, marker='o', color='gray', alpha=0.6)
    ax.scatter(-scaleT(-T[np.logical_and(ItlongFS, Ireg)], 30, 60, ylong), R[np.logical_and(ItlongFS, Ireg)], 15, marker='^', color='gray', alpha=0.6)

    ax.scatter(Time[selFS60], Rc[selFS60], s=25, marker='^', facecolors='none', edgecolors='b', label='FS: %d' % (sum(selFS60)+sum(selFSlong)))
    ax.scatter(Time[selAS60], Rc[selAS60], s=25, facecolors='none', edgecolors='r', label='AS: %d' % (sum(selAS60)+sum(selASlong)))
    ax.scatter(-scaleT(-Time[selFSlong], 30, 60, ylong), Rc[selFSlong], marker='^', s=25, edgecolors='b', facecolors='none')
    ax.scatter(scaleT(Time[selASlong], 30, 60, ylong), Rc[selASlong], s=25, edgecolors='r', facecolors='none')
    R = R[R > 0]
    R = R[R < np.inf]
    ax.plot([0, 0], [0.1, max(R)], '--m')
    ax.plot([30, 30], [0.1, max(R)], '--k')
    ax.plot([-30, -30], [0.1, max(R)], '-k')
    ax.plot([60, 60], [0.1, max(R)], '-k')
    ax.plot([-60, -60], [0.1, max(R)], '--k')
    ax.plot([-90, 90], [r_predicted / 1000, r_predicted / 1000], '-m', linewidth=1)
    ax.set_xlim([-90, 90])
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticklabels(['-%d years' % years, '-60d', '-30d', 'ot', '30d', '60d', '%d years' % years])

    ax.legend(loc='upper left')
    ax.set_xlabel('Time from t0 (days)')
    ax.set_ylabel('Distance from mainshock* (km)')
    ax.set_yscale('log')
    ax.set_ylim([0.1, max(R)])


def plotTimeDistClust(ax, clust, ot,Lat0, Lon0):
    Time = []
    for ii in range(len(clust['Time'])):
        Time.append(UTCDateTime(clust['Time'].values[ii]))
    Time = np.array(Time)
    t0 = UTCDateTime(ot)
    Time = (Time - t0) / (3600*24)
    selFS = Time < 0
    selAS = Time > 0

    # R = DistLatLon(Lat0, Lon0, clust.Lat.values, clust.Lon.values)
    R = DistLatLonUTM(Lat0, Lon0, clust.Lat.values, clust.Lon.values)
    
    ax.scatter(Time[selFS], R[selFS],s=15, color='b',alpha=0.8,label='FS: %d' % sum(np.logical_and(Time < 0, Time > -60)))
    ax.scatter(Time[selAS], R[selAS],s=15, color='r',alpha=0.8,label='AS: %d' % sum(np.logical_and(Time > 0, Time < 60)))
    ax.set_xlim([-60, 60])
    ax.legend(loc='upper left')
    ax.set_xlabel('Time from t0')
    ax.set_ylabel('Distance')

def add_lingling_data(LL, CLUST):
    LLid = np.zeros(len(CLUST)) - 999
    tj = []
    for jj in range(len(LL)):
        t = LL.Date[jj].split('/')
        tj.append(UTCDateTime('%s-%s-%s' % (t[2], t[1], t[0])).date)
    tj = np.array(tj)
    for ii in range(len(CLUST)):
        ti = UTCDateTime(CLUST.ot0[ii]).date
        I = np.logical_and(((ti - tj) / datetime.timedelta(1)) == 0, np.abs(LL.Mw - CLUST.m0[ii]) < 0.25)
        if sum(I) == 1:
            LLid[ii] = np.argmax(I + 1)
    CLUST.LLid = LLid
    return CLUST
                    
def add_Almann_Shearer_2009_data(LL, CLUST):
    LLid = np.zeros(len(CLUST)) - 999
    tj = []
    for jj in range(len(LL)):
        tj.append(UTCDateTime('%s-%s-%s' % (LL.yr[jj], LL.mon[jj], LL.day[jj])).date)
    tj = np.array(tj)    
    for ii in range(len(CLUST)):
        ti = UTCDateTime(CLUST.ot0[ii]).date
        I = np.logical_and(((ti - tj) / datetime.timedelta(1)) == 0, np.abs(LL.Mw - CLUST.m0[ii]) < 0.25)
        if sum(I) == 1:
            LLid[ii] = np.argmax(I + 1)
    CLUST.AnShid = LLid
    return CLUST


def plotTimesMclust(ax, clust, ot, M0, cat,Lats, Lons, years):
    # years = 1
    ylong = years*365 # years
    t0 = UTCDateTime(ot)
    T = (cat.ot - t0)/(3600*24)
    # # catalog time
    Ilon = np.logical_and(cat.Long>min(Lons), cat.Long<max(Lons))
    Ilat = np.logical_and(cat.Lat>min(Lats), cat.Lat<max(Lats))
    Ireg = np.logical_and(Ilat, Ilon)
    # It60 = np.logical_and(cat.ot > t0- 60*3600*24, cat.ot < t0 + 60*3600*24)
    It60FS = np.logical_and(cat.ot < t0, cat.ot > t0 - 60*3600*24)
    It60AS = np.logical_and(cat.ot > t0, cat.ot < t0 + 60*3600*24)
    ItlongAS = np.logical_and(cat.ot > t0 + 60*3600*24, cat.ot < t0 + ylong*3600*24)
    ItlongFS = np.logical_and(cat.ot < t0 - 60*3600*24, cat.ot > t0 - ylong*3600*24)

    # cluster time
    Time = []
    for ii in range(len(clust['Time'])):
        Time.append(UTCDateTime(clust['Time'].values[ii]))
    Time = np.array(Time)
    Time = (Time - t0) / (3600*24)

    selFS60 = np.logical_and(Time < 0, Time > -60)
    selFSlong = np.logical_and(Time <= -60, Time > -ylong)
    selAS60 = np.logical_and(Time > 0, Time < 60)
    selASlong = np.logical_and(Time >= 60, Time < ylong)

    stemW(ax, Time[selFS60], clust['MAG'][selFS60].values, 'b', '^', 'FS: %d' % sum(selFS60))
    stemW(ax, Time[selAS60], clust['MAG'][selAS60].values, 'r', 'o', 'AS: %d' % sum(selAS60))
    stemW(ax,-scaleT(-Time[selFSlong], 30, 60, ylong), clust['MAG'][selFSlong].values, 'b', '^', '')
    stemW(ax, scaleT(Time[selASlong], 30, 60, ylong), clust['MAG'][selASlong].values, 'r', 'o', '')

    # ax.scatter(T[np.logical_and(It60, Ireg)], cat.M[np.logical_and(It60, Ireg)], facecolors='grey', edgecolors='none', s=25, alpha=0.5)
    ax.scatter(T[np.logical_and(It60AS, Ireg)], cat.M[np.logical_and(It60AS, Ireg)], facecolors='grey', edgecolors='none', s=25, alpha=0.5)
    ax.scatter(T[np.logical_and(It60FS, Ireg)], cat.M[np.logical_and(It60FS, Ireg)], marker='^', facecolors='grey', edgecolors='none', s=25, alpha=0.5)
    ax.scatter(scaleT(T[np.logical_and(ItlongAS, Ireg)], 30, 60, ylong), cat.M[np.logical_and(ItlongAS,Ireg)], facecolors='grey', edgecolors='none', s=25, alpha=0.5)
    ax.scatter(-scaleT(-T[np.logical_and(ItlongFS, Ireg)], 30, 60, ylong), cat.M[np.logical_and(ItlongFS,Ireg)], marker='^', facecolors='grey', edgecolors='none', s=25, alpha=0.5)

    stemW(ax, [0], [M0], 'm', '*', '')
    ax.plot([30, 30], [0, M0+1], '--k')
    ax.plot([-30, -30], [0, M0+1], '-k')
    ax.plot([60, 60], [0, M0+1], '-k')
    ax.plot([-60, -60], [0, M0+1], '--k')
    ax.set_xlim([-90, 90])
    # ax.set_xlim([-10, 10])
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticklabels(['-%d years' % years, '-60d', '-30d', 'ot', '30d', '60d', '%d years' % years])
    # ax.legend(loc='upper left')
    ax.grid(color='k', linestyle='-', linewidth=0.5, axis='y', alpha=0.5)
    ax.set_xlabel('Time from t0')
    ax.set_ylabel('Magnitude')
    ax.set_ylim([min(cat.M)-1.0, M0+1])


def plotII(clust, ax, fsTypes):
    width = 0.35
    colors = ['dodgerblue', 'skyblue', 'whitesmoke']
    labels = 'Interplate', 'Intraplate'
    tp = clust['c_type'].values
    Idm = tp == 1  # types_c = 'Mainshocks'
    clust = clust[Idm]
    Iinter = clust['InterIntra'].values == 1
    Iintra = clust['InterIntra'].values == -1

    nf = clust[fsTypes].values
    nfh0 = nf == 0
    nfh1 = np.logical_and(nf >= 1, nf < 5)
    nfh5 = nf >= 5

    n2p = np.array([sum(Iinter), sum(Iintra)])
    nfh0fms = np.array([sum(np.logical_and(nfh0, Iinter)), sum(np.logical_and(nfh0, Iintra))]) / n2p * 100
    nfh1fms = np.array([sum(np.logical_and(nfh1, Iinter)), sum(np.logical_and(nfh1, Iintra))]) / n2p * 100
    nfh5fms = np.array([sum(np.logical_and(nfh5, Iinter)), sum(np.logical_and(nfh5, Iintra))]) / n2p * 100

    ax.bar(labels, nfh5fms, width, color=colors[0], edgecolor='black')#, colors=colors[0]) # shadow=True,
    ax.bar(labels, nfh1fms, width, bottom=nfh5fms, color=colors[1], edgecolor='black')
    ax.bar(labels, nfh0fms, width, bottom=nfh5fms+nfh1fms, color=colors[2], edgecolor='black')
    plb.grid(axis='y', linestyle='--')

    for ii in range(len(labels)):
        ax.text(x=ii, y =nfh5fms[ii]+1 , s='%d' % np.round(nfh5fms[ii]), fontdict=dict(fontsize=10), horizontalalignment ='center')
        ax.text(x=ii, y =nfh1fms[ii]+nfh5fms[ii]+1, s='%d' % np.round(nfh1fms[ii]+nfh5fms[ii]), fontdict=dict(fontsize=10), horizontalalignment ='center')
    ax.set_ylim([0, 100])


def plotFMS(clust, ax, InterIntra):
    width = 0.35
    colors = ['dodgerblue', 'skyblue', 'whitesmoke']

    tp = clust['c_type'].values
    Idm = tp == 1  # types_c = 'Mainshocks'
    if InterIntra == 1:
        II = clust['InterIntra'].values == 1
        Idm = Idm & II
    elif InterIntra == -1:
        II = clust['InterIntra'].values == -1
        Idm = Idm & II
    clust = clust[Idm]
    Iodd = clust['fstyle'].values == 0
    Iss  = clust['fstyle'].values == 1
    Inor = clust['fstyle'].values == 2
    Irev = clust['fstyle'].values == 3

    nf = clust['nf'].values
    nfh0 = nf == 0
    nfh1 = np.logical_and(nf >= 1, nf < 5)
    nfh5 = nf >= 5

    n2p = np.array([sum(Iodd), sum(Iss), sum(Inor), sum(Irev)])
    nfh0fms = np.array([sum(np.logical_and(nfh0, Iodd)), sum(np.logical_and(nfh0, Iss)), sum(np.logical_and(nfh0, Inor)), sum(np.logical_and(nfh0, Irev))]) / n2p * 100
    nfh1fms = np.array([sum(np.logical_and(nfh1, Iodd)), sum(np.logical_and(nfh1, Iss)), sum(np.logical_and(nfh1, Inor)), sum(np.logical_and(nfh1, Irev))]) / n2p * 100
    nfh5fms = np.array([sum(np.logical_and(nfh5, Iodd)), sum(np.logical_and(nfh5, Iss)), sum(np.logical_and(nfh5, Inor)), sum(np.logical_and(nfh5, Irev))]) / n2p * 100
    nfh5fms = np.nan_to_num(nfh5fms)
    nfh1fms = np.nan_to_num(nfh1fms)
    nfh0fms = np.nan_to_num(nfh0fms)
    labels = 'Oblique \n (%d)' % sum(Iodd), 'Strike-slip \n (%d)' % sum(Iss), 'Normal \n (%d)' % sum(Inor), 'Reverse \n (%d)' % sum(Irev)
    ax.bar(labels, nfh5fms, width, color=colors[0], edgecolor='black')
    ax.bar(labels, nfh1fms, width, bottom=nfh5fms, color=colors[1], edgecolor='black')
    ax.bar(labels, nfh0fms, width, bottom=nfh5fms+nfh1fms, color=colors[2], edgecolor='black')
    plb.grid(axis='y',linestyle='--')
    for ii in range(len(labels)):
        ax.text(x=ii, y=nfh5fms[ii]+1, s='%d' % np.round(nfh5fms[ii]), fontdict=dict(fontsize=10), horizontalalignment ='center')
        ax.text(x=ii, y=nfh1fms[ii]+nfh5fms[ii]+1, s='%d' % np.round(nfh1fms[ii]+nfh5fms[ii]), fontdict=dict(fontsize=10), horizontalalignment ='center')
    ax.set_ylim([0, 100])



def plotSeisArea(ax2, N, X, Y, dxy, area, EQx, EQy, CL, name):
    ax2.imshow(np.flipud(N), cmap=plb.cm.jet, interpolation='nearest',extent=[min(X), max(X)+dxy,min(Y),max(Y)+dxy],alpha=0.5,zorder=4)
    ax2.scatter(EQx,EQy,0.5,'w',label='Area %d km2' % round(area),zorder=5, alpha=0.5)
    ax2.grid()
    ax2.plot(CL.Lon.values,CL.Lat.values,'-k')
    # ax2.osm = OSM(ax2)
    ax2.set_xlim([min(X), max(X)])
    ax2.set_ylim([min(Y), max(Y)])
    # ax2.set_title('%s %d km2' % (self.CATS[ii].name, round(area)))

    legend1 = ax2.legend(loc='upper left')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1._legend_title_box._children[0]._fontproperties.set_size(14)
    legend1.set_title(name)

def plot_LL_params_fs(ax, CLUST, LL, Cat_Label, ctype):
    cc = 'rkg'
    for ii in range(len(CLUST)):
        pos_val = CLUST[ii].LLid[CLUST[ii].LLid > -999]
        ax.scatter(LL["SD E"][pos_val], CLUST[ii].nf_all[CLUST[ii].LLid > -999], c=cc[ii])
        ax.scatter(LL["SD E2.5"][pos_val], CLUST[ii].nf_all[CLUST[ii].LLid > -999], c=cc[ii], label=Cat_Label[ii])
    ax.set_xlabel('Stress Drop (MPa)')
    ax.set_ylabel('Foreshocks')
    ax.set_xscale('log')
    ax.grid()
    legend1 = ax.legend(loc='upper right')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1._legend_title_box._children[0]._fontproperties.set_size(14)
    legend1.set_title(ctype)

def plot_AnSh_params_fs(ax, CLUST, LL, Cat_Label, ctype):
    cc = 'rkg'
    for ii in range(len(CLUST)):
        pos_val = CLUST[ii].AnShid[CLUST[ii].AnShid > -999]
        ax.scatter(LL["stress_drop"][pos_val], CLUST[ii].nf_all[CLUST[ii].AnShid > -999], c=cc[ii], label=Cat_Label[ii])
    ax.set_xlabel('Stress Drop (MPa)')
    ax.set_ylabel('Foreshocks')
    ax.set_xscale('log')
    ax.grid()
    legend1 = ax.legend(loc='upper right')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1._legend_title_box._children[0]._fontproperties.set_size(14)
    legend1.set_title(ctype)
        

def plotFSbar(CLUST, ax, Cat_Label, fsTypes):
    width = 0.35
    # plb.rcParams.update({'font.family':'fantasy'})
    # plb.rcParams.update({'font.family':'Impact'})
    colors = ['dodgerblue', 'skyblue', 'whitesmoke']
    nfh0 = np.zeros(len(CLUST))
    nfh1 = np.zeros(len(CLUST))
    nfh5 = np.zeros(len(CLUST))
    f_nof = np.zeros(len(CLUST))
    f_per = np.zeros(len(CLUST))
    for ii in range(len(CLUST)):
        clust = CLUST[ii]
        tp = clust['c_type'].values
        nf = clust[fsTypes].values
        Idm = tp == 1  # types_c = 'Mainshocks'
        nf = nf[Idm]
        nfh0[ii] = sum(nf == 0) / len(nf) * 100

        nfh1[ii] = sum(np.logical_and(nf >= 1, nf < 5)) / len(nf) * 100

        nfh5[ii] = sum(nf >= 5) / len(nf) * 100

        f_nof[ii] = sum(nf >= 1) / sum(nf == 0)

        f_per[ii] = sum(nf >= 1) / len(nf) * 100

    nfh0m = np.mean(nfh0)
    nfh0std = np.std(nfh0)
    nfh1m = np.mean(nfh1)
    nfh1std = np.std(nfh1)
    nfh5m = np.mean(nfh5)
    nfh5std = np.std(nfh5)

    ax.bar(Cat_Label, nfh5, width, color=colors[0], edgecolor='black')#, colors=colors[0]) # shadow=True,
    ax.bar(Cat_Label, nfh1, width, bottom=nfh5, color=colors[1],edgecolor='black')
    ax.bar(Cat_Label, nfh0, width, bottom=nfh5+nfh1, color=colors[2], edgecolor='black')
    for ii in range(len(CLUST)):
        ax.text(x=ii, y =nfh5[ii]+1 , s='%d' % np.round(nfh5[ii]), fontdict=dict(fontsize=10), horizontalalignment ='center')
        ax.text(x=ii, y =nfh1[ii]+nfh5[ii]+1, s='%d' % np.round(nfh1[ii]+nfh5[ii]), fontdict=dict(fontsize=10), horizontalalignment ='center')
    ax.tick_params(axis='x', labelrotation=45)
    plb.grid(axis='y', linestyle='--')
    return nfh0m, nfh1m, nfh5m, nfh0std, nfh1std, nfh5std, f_nof, f_per

def plot_mainshocks_rate(ax, n_mainshocks, cat_years, Cat_Label, Ylabel):
    methods = ['ZnBZ',  'TMC', 'WnC']
    
    for ii in range(3):
        ax.scatter(np.arange(0, len(Cat_Label)), n_mainshocks[:, ii] / cat_years, label=methods[ii])
    ax.set_xticks(np.arange(0,len(Cat_Label)))
    ax.set_ylabel(Ylabel)
    ax.set_xticklabels(Cat_Label, rotation=45, fontname="Ariel", weight="bold")
    ax.grid()
    legend1 = ax.legend(loc='upper left')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1._legend_title_box._children[0]._fontproperties.set_size(12)

def add_rupture_circle(ax, M, lat0, lon0):
    [r_predicted,_] = WnCfaultL(M,ifault=0)
    # adjust effective radius
    fact_r = AdjustEffectiveRadius(M)
    r_predicted = fact_r * r_predicted
    Ro = kilometers2degrees(r_predicted / 1000)
    [xr, yr, xyr] = makeCircle(Ro, lon0, lat0)
    ax.plot(xr, yr, c='k')

def id2ctype(id):
    if id == -1:
        ctype = 'Low Seis.'
    elif id == 0:
        ctype = 'Undefined'
    elif id == 1:
        ctype = 'Mainshock*'
    elif id == 2:
        ctype = 'Doublet'
    elif id == 3:
        ctype = 'Triplet'
    elif id == 4:
        ctype = 'Swarms I'
    elif id == 5:
        ctype = 'Swarms II'
    return ctype


def plot_cluster_Maps3(Cat_names, CLUST, deCLUST_Z, deCLUST_M, deCLUST_G, cat, folder0, Mc):
    
    yearsLong = 2
    # remove previous files...
    files = glob.glob('%s*' % folder0)
    for f in files:
        os.remove(f)
    for ii in range(len(CLUST['M0'])):

        fig = plb.figure(10000 + ii, figsize=(15, 10), dpi=150)
        clustZ = deCLUST_Z[deCLUST_Z['clID'] == CLUST['Cid_Z'][ii]]
        clustM = deCLUST_M[deCLUST_M['clID'] == CLUST['Cid_M'][ii]]
        clustG = deCLUST_G[deCLUST_G['clID'] == CLUST['Cid_G'][ii]]
        try:
            [r_predicted, _] = WnCfaultL(CLUST['M0'][ii], ifault=0)
            fact_r = AdjustEffectiveRadius(CLUST['M0'][ii])
            r_predicted = fact_r * r_predicted
            Ro = kilometers2degrees(r_predicted / 1000)
            difx = Ro*2.1

            Lats0 = CLUST['Lat0'][ii]
            Lons0 = CLUST['Lon0'][ii]

            Lats = [Lats0-difx, Lats0+difx]
            Lons = [Lons0-difx, Lons0+difx]

            ax1a = fig.add_subplot(3, 3, 1, projection=ccrs.Mercator(central_longitude=180))
            ax1b = fig.add_subplot(3, 3, 4)
            ax1c = fig.add_subplot(3, 3, 7)
                
            # ax1a.set_title('ZnBZ')
            pos = plotMapC(ax1a, clustZ, Lats, Lons, CLUST['Lon0'][ii], CLUST['Lat0'][ii], CLUST['T0Z'][ii], CLUST['M0'][ii], CLUST['c_typeZ'][ii], cat, yearsLong, intra=True)
            plotTimesMclust(ax1b, clustZ, CLUST['T0Z'][ii], CLUST['M0'][ii], cat, Lats, Lons, yearsLong)
            plotR_TimeClust(ax1c, clustZ, CLUST['T0Z'][ii], CLUST['Lat0'][ii], CLUST['Lon0'][ii], CLUST['M0'][ii], cat, Lats, Lons, yearsLong)
            set_box_aspect(ax1b, 0.5)
            legend1 = ax1a.legend(loc='upper left')
            legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
            legend1._legend_title_box._children[0]._fontproperties.set_size(14)
            legend1.set_title('ZnBZ')
            # set_box_aspect(ax1c,0.5)
            # time.sleep(5)
            # plotTimeDistClust(ax3, clustZ, CLUST['T0Z'][ii],CLUST['Lat0'][ii], CLUST['Lon0'][ii])

            ax2a = fig.add_subplot(3, 3, 2, projection=ccrs.Mercator(central_longitude=180))
            ax2b = fig.add_subplot(3, 3, 5)
            ax2c = fig.add_subplot(3, 3, 8)
            # ax2a.set_title('WnC')
            plotMapC(ax2a, clustM, Lats, Lons, CLUST['Lon0'][ii], CLUST['Lat0'][ii], CLUST['T0M'][ii], CLUST['M0'][ii], CLUST['c_typeM'][ii], cat, yearsLong)
            plotTimesMclust(ax2b, clustM, CLUST['T0M'][ii], CLUST['M0'][ii],cat,Lats, Lons, yearsLong)
            plotR_TimeClust(ax2c, clustM, CLUST['T0M'][ii], CLUST['Lat0'][ii], CLUST['Lon0'][ii], CLUST['M0'][ii], cat, Lats, Lons, yearsLong)
            set_box_aspect(ax2b, 0.5)
            legend1 = ax2a.legend(loc='upper left')
            legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
            legend1._legend_title_box._children[0]._fontproperties.set_size(14)
            legend1.set_title('WnC')
            # set_box_aspect(ax2c,0.5)
            # time.sleep(5)
            # plotTimeDistClust(ax4, clustM, CLUST['T0M'][ii],CLUST['Lat0'][ii], CLUST['Lon0'][ii])

            ax3a = fig.add_subplot(3, 3, 3, projection=ccrs.Mercator(central_longitude=180))
            ax3b = fig.add_subplot(3, 3, 6)
            ax3c = fig.add_subplot(3, 3, 9)
            # ax3a.set_title('TMC')
            plotMapC(ax3a, clustG, Lats, Lons, CLUST['Lon0'][ii], CLUST['Lat0'][ii], CLUST['T0G'][ii], CLUST['M0'][ii], CLUST['c_typeG'][ii], cat, yearsLong)
            plotTimesMclust(ax3b, clustG, CLUST['T0G'][ii], CLUST['M0'][ii],cat,Lats, Lons, yearsLong)
            plotR_TimeClust(ax3c, clustG, CLUST['T0G'][ii], CLUST['Lat0'][ii],CLUST['Lon0'][ii], CLUST['M0'][ii],cat,Lats, Lons, yearsLong)
            set_box_aspect(ax3b, 0.5)
            legend1 = ax3a.legend(loc='upper left')
            legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
            legend1._legend_title_box._children[0]._fontproperties.set_size(14)
            legend1.set_title('TMC')
            # set_box_aspect(ax3c,0.5)
            # time.sleep(5)
            # plotTimeDistClust(ax6, clustG, CLUST['T0G'][ii],CLUST['Lat0'][ii], CLUST['Lon0'][ii])
            if pos == -1:
                posT = 'Intraplate'
            elif pos == 1:
                posT = 'Interplate'
            else:
                posT = ''
            ax2b.set_xlabel('%s Mw%s  %s' % (UTCDateTime(CLUST['T0Z'][ii]).date, CLUST['M0'][ii], posT), fontsize=20)
            
            fname = '%sOT_%s_Mw%s.png' % (folder0, UTCDateTime(CLUST['T0Z'][ii]).date, CLUST['M0'][ii])
            
            if Mc > 3:
                ax10 = fig.add_subplot(4, 7, 5, projection=ccrs.Orthographic(central_longitude=CLUST['Lon0'][ii], central_latitude=CLUST['Lat0'][ii]))
                ax10.coastlines(resolution='50m', linewidths=0.5)
                ax10.stock_img()
                ax10.plot(CLUST['Lon0'][ii], CLUST['Lat0'][ii], 'm*', transform=ccrs.Geodetic())
                fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2) # pad=0.4, w_pad=0.5, h_pad=0.9
            else:
                Lons0 = [min(cat.Long),max(cat.Long)]
                Lats0 = [min(cat.Lat),max(cat.Lat)]
                # ax10 = fig.add_subplot(4, 8, 6, projection=ccrs.Mercator())
                ax10 = fig.add_subplot(4, 8, 6, projection=ccrs.PlateCarree())
                plotworldloc2(ax10, CLUST['Lon0'][ii], CLUST['Lat0'][ii], Lats0, Lons0)
                
            fig.savefig(fname)
            plb.close(fig)
            print('%d Mw%s %s' % (ii, CLUST['M0'][ii], UTCDateTime(CLUST['T0Z'][ii]).date))
        except:
            print('no data for %d' % ii)
            plb.close(fig)


def ForeshockMoment_BG_FS(CLUST, deCLUST_Z,deCLUST_M,deCLUST_G, cat):
    M0_WnC = 0
    M0_TMC_R = 0
    M0_ZnBZ_R = 0
    n_FS_WnC = 0
    n_FS_TMC = 0
    n_FS_ZnBZ = 0

    for ii in range(len(CLUST['M0'])):
        if CLUST['c_typeM'][ii] == 1:

            clustZ = deCLUST_Z[deCLUST_Z['clID'] == CLUST['Cid_Z'][ii]]
            clustM = deCLUST_M[deCLUST_M['clID'] == CLUST['Cid_M'][ii]]
            clustG = deCLUST_G[deCLUST_G['clID'] == CLUST['Cid_G'][ii]]
            
            [r_predicted,_] = WnCfaultL(CLUST['M0'][ii], ifault=0)
            fact_r = AdjustEffectiveRadius(CLUST['M0'][ii])
            r_predicted = fact_r * r_predicted
            TimeZ = []
            TimeM = []
            TimeG = []
            for jj in range(len(clustZ['Time'])):
                TimeZ.append(UTCDateTime(clustZ['Time'].values[jj]))
            for jj in range(len(clustM['Time'])):
                TimeM.append(UTCDateTime(clustM['Time'].values[jj]))
            for jj in range(len(clustG['Time'])):
                TimeG.append(UTCDateTime(clustG['Time'].values[jj]))
            TimeZ = (np.array(TimeZ) - UTCDateTime(CLUST['T0Z'][ii])) / (3600*24)
            TimeM = (np.array(TimeM) - UTCDateTime(CLUST['T0M'][ii])) / (3600*24)
            TimeG = (np.array(TimeG) - UTCDateTime(CLUST['T0G'][ii])) / (3600*24)

            Ifs_Z  = np.logical_and(TimeZ < 0, TimeZ > -60)
            Ifs_M  = np.logical_and(TimeM < 0, TimeM > -60)
            Ifs_G  = np.logical_and(TimeG < 0, TimeG > -60)

            RZ = DistLatLon(CLUST['Lat0'][ii], CLUST['Lon0'][ii], clustZ['Lat'].values, clustZ['Lon'].values)
            RG = DistLatLon(CLUST['Lat0'][ii], CLUST['Lon0'][ii], clustG['Lat'].values, clustG['Lon'].values)
            IR_Z = RZ < r_predicted / 1000
            IR_G = RG < r_predicted / 1000
            
            I_FS_Z = np.logical_and(IR_Z, Ifs_Z)
            I_FS_G = np.logical_and(IR_G, Ifs_G)
            I_FS_M = Ifs_M
            
            M0_WnC = M0_WnC + sum(Mw2M0(clustM['MAG'][I_FS_M]))
            M0_ZnBZ_R = M0_ZnBZ_R + sum(Mw2M0(clustZ['MAG'][I_FS_Z]))
            M0_TMC_R = M0_TMC_R + sum(Mw2M0(clustG['MAG'][I_FS_G]))
            

            if sum(Mw2M0(clustM['MAG'][I_FS_M])) < sum(Mw2M0(clustG['MAG'][I_FS_G])):
                print('problem check foreshocks %s!' % (UTCDateTime(CLUST['T0Z'][ii])))


    return M0_WnC, M0_ZnBZ_R, M0_TMC_R
            
    
def plot_FS_Moment_cat(M0_WnC, M0_ZnBZ_R, M0_TMC_R, Cat_names):
    fig = plb.figure(6400)
    ax1 = fig.add_subplot(1,1,1)
    #ax2 = fig.add_subplot(1,2,2)
    ax1.scatter(np.arange(0,len(Cat_names)), (M0_WnC - M0_ZnBZ_R) / M0_WnC, label='M0* = ZnBZ')
    ax1.scatter(np.arange(0,len(Cat_names)), (M0_WnC - M0_TMC_R) / M0_WnC, label='M0* = TMC')
    ax1.set_ylabel('(M0_WnC - M0*) / M0_WnC')
    ax1.set_xticks(np.arange(0,len(Cat_names)))
    ax1.set_xticklabels(Cat_names, rotation = 45)

    # ax2.set_ylabel('(M0_WnC - M0_TMC_R) / M0_WnC')
    # ax2.set_xticks(np.arange(0,len(Cat_names)))
    # ax2.set_xticklabels(Cat_names, rotation = 45)
    ax1.set_ylim(0, 0.75)
    ax1.legend()
    ax1.grid()
    # ax2.grid()


def plot_events_after60days(CLUST_Z, Cat_names, ii):
    fig = plb.figure(700)
    ax = fig.add_subplot(2,3,ii+1)
    ax.scatter(CLUST_Z['m0'], CLUST_Z['nf_all'].values - CLUST_Z['nf'].values,20,'b',alpha=0.7)
    ax.scatter(CLUST_Z['m0'], CLUST_Z['na_all'].values - CLUST_Z['na'].values,20,'r',alpha=0.7)
    ax.set_xlabel('Mainshock Mw')
    ax.set_ylabel(r'$N_{all} - N_{60}$')
    ax.set_yscale('log')
    ax.set_ylim([1,200])
    ax.set_title('%s' % Cat_names)
    
def plot_N_ZvM(CLUST, Cat_names, ii):
    fig = plb.figure(720)
    ax = fig.add_subplot(2,3,ii+1)
    ax.scatter(CLUST['N_FS_M'].values, CLUST['N_FS_Z'].values,20,'b',alpha=0.7)
    ax.scatter(CLUST['N_AS_M'].values, CLUST['N_AS_Z'].values,20,'r',alpha=0.7)
    ax.set_xlabel('Number of event Mag. Dep.')
    ax.set_ylabel('Number of event Zaliapin 60d')
    ax.plot([1,10000],[1,10000],'-k')
    ax.set_xlim([1, 10000])
    ax.set_ylim([1, 10000])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_aspect('equal')
    ax.set_title('%s' % Cat_names)

    fig1 = plb.figure(721)
    ax1 = fig1.add_subplot(2,3,ii+1)
    ax1.scatter(CLUST['N_FS_M'].values, CLUST['N_FS_allZ'].values,20,'b',alpha=0.7)
    ax1.scatter(CLUST['N_AS_M'].values, CLUST['N_AS_allZ'].values,20,'r',alpha=0.7)
    ax1.set_xlabel('Number of event Mag. Dep.')
    ax1.set_ylabel('Number of event Zaliapin')
    ax1.plot([0,100],[0,100],'-k')
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    ax1.set_aspect('equal')
    ax1.set_title('%s' % Cat_names)

def plot_foreshocks_mag(CLUST_G,CLUST_M, CLUST_Z, ax, name0):

        mG = CLUST_G['m0'].values
        mM = CLUST_M['m0'].values
        mZ = CLUST_Z['m0'].values
        
        nfG   = CLUST_G['nf'].values
        nfM   = CLUST_M['nf'].values
        nfZ   = CLUST_Z['nf'].values
        
        ImG   = np.logical_and(CLUST_G['c_type60'].values == 1, nfG > 0)
        ImM   = np.logical_and(CLUST_M['c_type60'].values == 1, nfM > 0)
        ImZ   = np.logical_and(CLUST_Z['c_type60'].values == 1, nfZ > 0)
        
        nMrangeG, McenterG = binMag(mG[ImG],min(mG),nfG[ImG])
        nMrangeM, McenterM = binMag(mM[ImM],min(mM),nfM[ImM])
        nMrangeZ, McenterZ = binMag(mZ[ImZ],min(mZ),nfZ[ImZ])
        
        minmaxG = np.array([min(mG), max(mG)])
        minmaxM = np.array([min(mM), max(mM)])
        minmaxZ = np.array([min(mZ), max(mZ)])
        
        
        
        ax.scatter(mG[ImG], nfG[ImG], 30, alpha=0.7, facecolor='b',marker='o', edgecolors='k', label='GnK')
        ax.scatter(mM[ImM], nfM[ImM], 30, alpha=0.7, facecolor='r',marker='s', edgecolors='k', label='WnC')
        ax.scatter(mZ[ImZ], nfZ[ImZ], 30, alpha=0.7, facecolor='g',marker='^', edgecolors='k', label='ZnBZ')
        
        
        ax.plot(McenterG, nMrangeG, 's-b')
        ax.plot(McenterM, nMrangeM, 's-r')
        ax.plot(McenterZ, nMrangeZ, 's-g')

        ax.set_xlabel('Mainshock magnitud Mw')
        
        ax.set_ylabel('Number of foreshocks')
        
        ax.set_yscale('log')
        ax.set_ylim([1, 1000])
        ax.set_xlim(minmaxM)
        # ax.legend(title=self.CATS[ic].name0,loc='upper left')
        legend1 = ax.legend(loc='upper left')
        legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
        legend1._legend_title_box._children[0]._fontproperties.set_size(14)
        legend1.set_title(name0)
    
def ForeshocksDuration(ax1, ax2, ax3, Cat_names, CLUST_Z, CLUST_M, CLUST_W, deCLUST_Z, deCLUST_M, deCLUST_W):
    bins = np.arange(-100, 10, 10)
    binsc = (bins[1:] + bins[:-1]) / 2
    foreshocks_time_Z = calc_cforeshocks_duration(CLUST_Z, deCLUST_Z, 5.0)
    foreshocks_time_M = calc_cforeshocks_duration(CLUST_M, deCLUST_M, 5.0)
    foreshocks_time_W = calc_cforeshocks_duration(CLUST_W, deCLUST_W, 5.0)
    nZ, hist_Z = np.histogram(foreshocks_time_Z, bins)
    nM, hist_M = np.histogram(foreshocks_time_M, bins)
    nW, hist_W = np.histogram(foreshocks_time_W, bins)

    ax1.plot(binsc, nZ, label=Cat_names)
    ax1.set_xlabel('Days')
    ax1.grid()
    ax1.set_title('ZnBZ')
    ax1.set_ylabel('Mainshocks* with foreshocks')

    ax2.plot(binsc, nM, label=Cat_names)
    ax2.set_xlabel('Days')
    ax2.grid()
    ax2.set_title('WnC')
    
    ax3.plot(binsc, nW, label=Cat_names)
    ax3.set_xlabel('Days')
    ax3.grid()
    ax3.set_title('TMC')
    
    # legend1 = ax.legend(loc='upper left')
    # legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    # legend1._legend_title_box._children[0]._fontproperties.set_size(14)
    # legend1.set_title(self.CATS[ic].name0)
    # set_box_aspect(ax, 1)
    

def plot_events_Z_vs_WnC(CLUST, Cat_names, ii):

    # fig = plb.figure(740)
    # ax = fig.add_subplot(9,2,2*ii+1)
    # dFS_Z_M = CLUST['N_FS_Z'].values - CLUST['N_FS_M'].values
    # dFS_Z_G = CLUST['N_FS_Z'].values - CLUST['N_FS_W'].values
    # bins = [0,1,5,20,500]
    # n, bins,patches = ax.hist(dFS_Z_M, bins,density=False,facecolor='g',alpha=0.75,width=2)
    # ax.set_xlabel('Foreshocks ZnBZ - WnG %s' % Cat_names)
    #
    # ax = fig.add_subplot(9,2,2*ii+2)
    # n, bins,patches = ax.hist(dFS_Z_M, bins,density=False,facecolor='g',alpha=0.75, width=2)
    # ax.set_xlabel('Foreshocks ZnBZ - GnK %s' % Cat_names)

    fig = plb.figure(730)
    def plotRatioCats(ax, M, N_Z, N_M,c):
        Iok = np.logical_and(N_Z > 0, N_M > 0)
        ax.scatter(M[Iok], N_Z[Iok] / N_M[Iok],5,c,alpha=0.7)
        meanN = np.mean(N_Z[Iok] / N_M[Iok])
        # meanN = sum(N_Z[Iok]) / sum(N_M[Iok])
        # meanN = 10**(np.mean(np.log10(N_Z[Iok]) / np.log10(N_M[Iok])))
        xlims = ax.get_xlim()
        ax.plot(xlims, [meanN, meanN],c=c)
        return xlims


    ax = fig.add_subplot(4,2,ii+1)
    xlims = plotRatioCats(ax, CLUST['M0'], CLUST['N_AS_Z'], CLUST['N_AS_M'],'r')
    xlims = plotRatioCats(ax, CLUST['M0'], CLUST['N_FS_Z'], CLUST['N_FS_M'],'b')

    ax.set_xlabel('Mainshock Mw')
    if ii == 0:
        ax.set_ylabel(r'$ZnBZ_{60d} / WnC$')
    ax.set_yscale('log')
    ax.set_ylim([0.01, 100])
    legend1 = ax.legend(loc='upper left')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1.set_title('%s' % Cat_names)
    ax.set_xlim(xlims)
    ax.grid()

    fig1 = plb.figure(731)
    ax1 = fig1.add_subplot(4, 2, ii+1)
    xlims = plotRatioCats(ax1, CLUST['M0'], CLUST['N_AS_allZ'], CLUST['N_AS_M'], 'r')
    xlims = plotRatioCats(ax1, CLUST['M0'], CLUST['N_FS_allZ'], CLUST['N_FS_M'], 'b')
    ax1.set_xlabel('Mainshock Mw')
    if ii == 0:
        ax1.set_ylabel(r'$ZnBZ / WnC$')
    ax1.set_yscale('log')
    ax1.set_ylim([0.01,100])
    legend1 = ax1.legend(loc='upper left')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1.set_title('%s' % Cat_names)
    ax1.set_xlim(xlims)
    ax1.grid()

def plot_BG_cat(ax, Cat_names, BG_CL):
    ax.scatter(np.arange(0,len(Cat_names)), BG_CL)
    ax.set_ylabel('BG (%)')
    ax.set_xticks(np.arange(0,len(Cat_names)))
    ax.set_xticklabels(Cat_names, rotation = 90)
    ax.set_ylim(20, 100)
    ax.grid()
    set_box_aspect(ax,0.5)

def plot_val_2cat(ax, Cat_names, BG_CL1, BG_CL2,label1,label2, Ylabel):
    ax.scatter(np.arange(0,len(Cat_names)), BG_CL1,label=label1)
    ax.scatter(np.arange(0,len(Cat_names)), BG_CL2,label=label2)
    ax.set_ylabel(Ylabel)
    ax.set_xticks(np.arange(0,len(Cat_names)))
    ax.set_xticklabels(Cat_names, rotation = 90)
    #ax.set_ylim(20, 100)
    minY = min([min(BG_CL1),min(BG_CL2)])
    maxY = max([max(BG_CL1),max(BG_CL2)])
    dY = maxY - minY
    ax.set_ylim([minY-0.1*dY, maxY+0.1*dY])
    ax.grid()
    set_box_aspect(ax,0.5)
    ax.legend()



def plot_val_2NF(ax, f_nof1, BG_CL1,label1, f_nof2, BG_CL2,label2, Cat_names, Xlabel, Ylabel,std1,std2):
    ax.scatter(BG_CL1, f_nof1, label=label1)
    # ax.errorbar(BG_CL1, f_nof1, yerr=std1, fmt="o")
    ax.scatter(BG_CL2, f_nof2, label=label2)
    # ax.errorbar(BG_CL2, f_nof2, yerr=std2, fmt="o")
    for ii in range(len(f_nof1)):
        ax.text(BG_CL1[ii], f_nof1[ii],Cat_names[ii])
        ax.plot([BG_CL1[ii],BG_CL2[ii]],[f_nof1[ii],f_nof2[ii]],'-k')
    ax.set_ylabel(Ylabel)
    ax.set_xlabel(Xlabel)
    minX = min([min(BG_CL1),min(BG_CL2)])
    maxX = max([max(BG_CL1),max(BG_CL2)])
    dX = maxX - minX
    ax.set_xlim([minX-0.1*dX, maxX+0.1*dX])
    # ax.set_ylim(20, 100)
    ax.grid()
    set_box_aspect(ax,1.0)
    ax.legend()

def plot_BG_NF(ax, f_nof, BG_CL,Cat_names):
    ax.scatter(f_nof, BG_CL)
    for ii in range(len(f_nof)):
        ax.text(f_nof[ii], BG_CL[ii],Cat_names[ii])
    # ax.set_xlabel('with fs / no fs')
    ax.set_xlabel('Mainshocks with foreshocks (%)')
    ax.set_ylabel('BG (%)')
    # ax.set_xlim([-10+min(f_nof), 10+max(f_nof)])
    ax.set_xlim(10, 70)
    ax.set_ylim(20, 100)
    ax.grid()
    set_box_aspect(ax,1.0)

def plot_BG_Lamda(ax, PBG_Z, BG_CL,Cat_names, label):
    Lamda = np.zeros(len(Cat_names))
    for ii in range(len(Cat_names)):
        Lamda[ii] = PBG_Z[ii].Lambda.values[0]

    ax.scatter(Lamda, BG_CL,label=label)
    for ii in range(len(Lamda)):
        ax.text(Lamda[ii], BG_CL[ii],Cat_names[ii])
    # ax.set_xlabel('with fs / no fs')
    ax.set_ylabel('Background (%)')
    ax.set_xlabel('Lambda')
    # ax.set_xlim([-10+min(f_nof), 10+max(f_nof)])
    # ax.set_xlim(10, 70)
    # ax.set_ylim(20, 100)
    ax.grid()
    set_box_aspect(ax,1.0)
    ax.legend()

def plot_compare2(ax, Xdata, Ydata, Cat_names, Xlabel, Ylabel):

    ax.scatter(Xdata, Ydata)
    ax.plot([min(Xdata),max(Xdata)],[min(Xdata),max(Xdata)],':k')
    for ii in range(len(Xdata)):
        ax.text(Xdata[ii], Ydata[ii],Cat_names[ii])
    ax.set_xlabel(Xlabel)
    ax.set_ylabel(Ylabel)
    minX = min(Xdata)
    maxX = max(Xdata)
    dX = maxX - minX
    ax.set_xlim([minX-0.1*dX, maxX+0.1*dX])
    ax.set_ylim([minX-0.1*dX, maxX+0.1*dX])
    ax.grid()
    set_box_aspect(ax,1.0)


def plot_FS_Lamda(ax, PBG_Z, FS,Cat_names, label):
    Lamda = np.zeros(len(Cat_names))
    for ii in range(len(Cat_names)):
        Lamda[ii] = PBG_Z[ii].Lambda.values[0]

    ax.scatter(Lamda, FS, label=label)
    for ii in range(len(Lamda)):
        ax.text(Lamda[ii], FS[ii],Cat_names[ii])
    # ax.set_xlabel('with fs / no fs')
    ax.set_ylabel('Mainshocks with foreshocks (%)')
    ax.set_xlabel('Lambda')
    # ax.set_xlim([-10+min(f_nof), 10+max(f_nof)])
    # ax.set_xlim(10, 70)
    # ax.set_ylim(20, 100)
    ax.grid()
    set_box_aspect(ax,1.0)
    ax.legend()
    
def plot_ctype_bars(CLUST, Cat_names, ii):
    fig = plb.figure(760)
    ax = fig.add_subplot(2,3,ii+1)
    c_type_names = ['Mainshocks','Doublets','Triplets','Swarms','Un-defined']


    tpM   = CLUST['c_typeM'].values # WnC 60 days
    tpZ   = CLUST['c_typeallZ'].values
    tpZ60 = CLUST['c_typeZ'].values
    tpM[tpM == 0] = 5
    tpZ[tpZ == 0] = 5
    not_defined_M = sum(tpM == 5)
    not_defined_Z = sum(tpZ == 5)
    Mainshocks_M  = sum(tpM == 1)
    Mainshocks_Z  = sum(tpZ == 1)
    Doublets_M    = sum(tpM == 2)
    Doublets_Z    = sum(tpZ == 2)
    Triplets_M    = sum(tpM == 3)
    Triplets_Z    = sum(tpZ == 3)
    Swarms_M      = sum(tpM == 4)
    Swarms_Z      = sum(tpZ == 4)
    ax.hist([tpM,tpZ],[0.5,1.5,2.5,3.5,4.5,5.5],edgeColor = 'black',label=['WnC', 'ZnBZ']) #
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    for ii in range(1,len(c_type_names)):
        ax.text(ii, 1+sum(tp == ii), '%d' % sum(tp == ii))
    ax.set_ylabel('Number of sequences')
    ax.set_xticks(1.0+np.arange(len(c_type_names)))
    ax.set_xticklabels(c_type_names)
    plb.xticks(rotation=-45)
    ax.legend(title=Cat_names,loc='upper right')
    ax.set_xlim([0.5,5.5])
    ax.set_ylim([0, 1.2*max([not_defined_M,not_defined_Z,Mainshocks_M, Doublets_M, Triplets_M, Swarms_M,Mainshocks_Z, Doublets_Z, Triplets_Z, Swarms_Z])])
    fig.subplots_adjust(hspace=0.25)
    
    
def plot_ctype_bars_2cats(CLUST_Z, CLUST_M, Cat_names, ii):
    fig = plb.figure(760)
    ax = fig.add_subplot(2,3,ii+1)
    c_type_names = ['Mainshocks','Doublets','Triplets','Swarms I','Swarms II','Un-defined','Low Seis.']

    MminM = min(CLUST_M['m0'].values)
    I = CLUST_Z['m0'].values >= MminM
    CLUST_Z = CLUST_Z[I]
    CLUST_Z = CLUST_Z.reset_index(drop=True)
    CLUST_M = CLUST_M.reset_index(drop=True)


    tpM   = CLUST_M['c_type'].values
    tpZ   = CLUST_Z['c_type'].values
    tpM[tpM == 0] = 6
    tpZ[tpZ == 0] = 6
    tpM[tpM == -1] = 7
    tpZ[tpZ == -1] = 7
    
    Mainshocks_M  = sum(tpM == 1)
    Mainshocks_Z  = sum(tpZ == 1)
    Doublets_M    = sum(tpM == 2)
    Doublets_Z    = sum(tpZ == 2)
    Triplets_M    = sum(tpM == 3)
    Triplets_Z    = sum(tpZ == 3)
    Swarms1_M     = sum(tpM == 4)
    Swarms1_Z     = sum(tpZ == 4)
    Swarms2_M     = sum(tpM == 5)
    Swarms2_Z     = sum(tpZ == 5)
    not_defined_M = sum(tpM == 6)
    not_defined_Z = sum(tpZ == 6)
    LowSeis_M     = sum(tpM == 7)
    LowSeis_Z     = sum(tpZ == 7)

    ax.hist([tpM,tpZ],[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5],edgeColor = 'black',label=['WnC', 'Zaliapin']) #
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    # for ii in range(1,len(c_type_names)):
    #     ax.text(ii, 1+sum(tp == ii), '%d' % sum(tp == ii))
    ax.set_ylabel('Number of sequences')
    ax.set_xticks(1.0+np.arange(len(c_type_names)))
    ax.set_xticklabels(c_type_names)
    plb.xticks(rotation=-45)
    ax.legend(title=Cat_names,loc='upper left')
    ax.set_xlim([0.5,7.5])
    ax.set_ylim([0, 1.2*max([not_defined_M,not_defined_Z,Mainshocks_M, Doublets_M, Triplets_M, Swarms1_M,Swarms2_M,Mainshocks_Z, Doublets_Z, Triplets_Z, Swarms1_Z,Swarms2_Z,LowSeis_M,LowSeis_Z])])
    fig.subplots_adjust(hspace=0.25)


    

