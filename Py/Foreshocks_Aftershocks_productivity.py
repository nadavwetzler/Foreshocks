import concurrent.futures.process
import os
import mpl_toolkits
import shapefile
import multiprocess as mp
import mpl_toolkits
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from osm import OSM as OSM
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, read_inventory, read_events
import numpy as np
import math
from pyproj import Proj
import matplotlib.pyplot as plb
from obspy.core.event import Catalog
import pandas as pd
import datetime
from datetime import datetime as dtt
import time
from geopy import distance
from obspy.geodetics.base import gps2dist_azimuth
from geopy.distance import geodesic
import haversine as hs
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from matplotlib import path, cm
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from pyrocko import moment_tensor as pmt
from numpy import linalg as LA
from numpy import inf
from sklearn.metrics import r2_score # "scikit-learn"
from calc_gr_ks import *
from Zclustering import *
from Sequence_type import *
from mechanismStyle import *
from Compare_cluster_functions import *
from inpolygon import *
import omori as omori
import seis_utils as seis_utils
import warnings
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore")

import scipy


class CCats():
    pass


class Catalog0():
    pass


def loadMatBin(filename):
    '''
    this function should be called instead of direct scipy.io.loadmat
    as it helps with additional non-variable tags in python dictionaries from .mat files
    --> can handle 'nested' variables in matlab where variable contain several structures
    '''
    data = scipy.io.loadmat(filename,struct_as_record=False, squeeze_me=True)
    return data
        
def histw(x, bins):
    y = np.zeros(len(bins)-1)
    for ii in range(len(bins)-1):
        y[ii] = sum((x > bins[ii]) & (x <= (bins[ii+1])))
    return y

def CAT2ETAScat(cat, cat_years,Mc):
    m = cat.M
    t = cat.datenum
    catalog = np.zeros((len(t),2))
    catalog[:,0] = t
    catalog[:,1] = m
    catalog = catalog[catalog[:,0]>max(catalog[:,0] - cat_years*365),:]
    catalog = np.column_stack((catalog, (catalog[:,0] - min(catalog[:,0])) / 365))
    catalog = catalog[catalog[:,1] >= Mc,:]
    print('Catalog includes: %d events!' % (len(catalog)))
    return catalog
    
def degpalesph(u, v):
    theta = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)) / np.pi * 180
    theta = min([abs(theta), abs(180-theta)])
    return theta


def make360(x):
    x = np.array([x])
    n1 = len(x)
    x1 = np.zeros(n1)
    for ii in range(n1):
        check = 0
        if x[ii] < 0:
            x1[ii] = 360 + x[ii]
            check = 1
        if x[ii] > 360:
            x1[ii] = x[ii] - 360
            check = 1
        if x1[ii] < 0:
            x1[ii] = 360 + x1[ii]
            check = 1
        if x1[ii] > 360:
            x1[ii] = x1[ii] - 360
            check = 1
        if check == 0:
            x1[ii] = x[ii]
    return x1
        
        
def fms2tpb(fms):
    st = fms[0] 
    dp = fms[1]
    rk = fms[2]
    rk = make360(rk)
    rk = np.radians(rk)
    dp = np.radians(dp)
    st = np.radians(st)
    m = np.zeros((3, 3))
    m[0, 0] = -(np.sin(dp)*np.cos(rk)*np.sin(2*st) + np.sin(2*dp)*np.sin(rk)*(np.sin(st))**2)
    m[0, 1] = np.sin(dp)*np.cos(rk)*np.cos(2*st) + 0.5*np.sin(2*dp)*np.sin(rk)*np.sin(2*st)
    m[0, 2] = -(np.cos(dp)*np.cos(rk)*np.cos(st)+np.cos(2*dp)*np.sin(rk)*np.sin(st))
    m[1, 2] = -(np.cos(dp)*np.cos(rk)*np.cos(st) + np.cos(2*dp)*np.sin(rk)*np.sin(st))
    m[1, 1] = np.sin(dp)*np.cos(rk)*np.sin(2*st) - np.sin(2*dp)*np.sin(rk)*(np.cos(st))**2
    m[1, 2] = -(np.cos(dp)*np.cos(rk)*np.sin(st) - np.cos(2*dp)*np.sin(rk)*np.cos(st))
    m[2, 2] = np.sin(2*dp)*np.sin(rk)
    m[2, 1] = m[1, 2]
    m[1, 0] = m[0, 1]
    m[2, 0] = m[0, 2]
    val, vec = LA.eig(m)
    # vec[:, 2] = vec[:, 2] * -1
    posT = np.argmax(val)
    posP = np.argmin(val)
    posB = 5 - (posP+posT+2)

    T = vec[:, posT]
    P = vec[:, posP]
    B = vec[:, posB]

    return P, T, B



def compare_fms(fms0, fms1, deg):
    
    P0, T0, B0 = fms2tpb(fms0)
    P1, T1, B1 = fms2tpb(fms1)

    ThetaP = degpalesph(P1, P0)
    ThetaT = degpalesph(T1, T0)
    ThetaB = degpalesph(B1, B0)

    # theta = max([ThetaP, ThetaT, ThetaB])
    if (ThetaP < deg) & (ThetaT < deg) & (ThetaB < deg):
        theta = 1
    else:
        theta = 90
    
    return theta
    







def DistFromEQ2(cat0, ii):
    R = 6373.0
    lat1 = np.radians(cat0.Lat[ii])
    lon1 = np.radians(cat0.Lon[ii])
    lat2 = np.radians(cat0.Lat)
    lon2 = np.radians(cat0.Lon)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_R = R * c
    # ll = len(cat0.Long)
    # distance_R = np.zeros(ll)
    # for jj in range(ll):
    #     coordinates_from = [cat0.Lat[ii], cat0.Long[ii]]
    #     coordinates_to   = [cat0.Lat[jj], cat0.Long[jj]]
    #     distance_R[jj]   = distance.distance(coordinates_from, coordinates_to).km
    #
    return distance_R


def ftypeShearer(r1, r2):
    def ftype2fid(fptype):
        if (fptype >= -1) & (fptype < -0.5):
            fid = 2  # normal
        elif (fptype >= -0.5) & (fptype < -0.25):
            fid = 0  # oblique
        elif (fptype >= -0.25) & (fptype < 0.25):
            fid = 1  # strike-slip
        elif (fptype >= 0.25) & (fptype < 0.5):
            fid = 0  # oblique
        else:
            fid = 3  # reverse
        return fid

    r1 = Make180(r1)
    r2 = Make180(r2)
    if abs(r1) > 90:
        r1 = (180-abs(r1))*(r1/abs(r1))
    if abs(r2) > 90:
        r2 = (180-abs(r2))*(r2/abs(r2))

    fid1 = ftype2fid(r1 / 90)
    fid2 = ftype2fid(r2 / 90)
    if abs(fid1) > abs(fid2):
        fptype = r1 / 90
        fid = fid1
    else:
        fptype = r2 / 90
        fid = fid2

    return fptype, fid

def ftypeShearer2(r1, r2):
    def ftype2fid(fptype):
        if (fptype >= -1) & (fptype < -0.5):
            fid = 2  # normal
        elif (fptype >= -0.5) & (fptype < -0.25):
            fid = 0  # oblique
        elif (fptype >= -0.25) & (fptype < 0.25):
            fid = 1  # strike-slip
        elif (fptype >= 0.25) & (fptype < 0.5):
            fid = 0  # oblique
        else:
            fid = 3  # reverse
        return fid
        
    def dist2oblique(r):
        r = abs(r)
        if r > 0.5:
            l = r - 0.5
        elif r < 0.25:
            l = 0.25 - r
        else:
            l = 0
        return l
    
    r1 = Make180(r1)
    r2 = Make180(r2)
    
    
    if abs(r1) > 90:
        r1 = (180-abs(r1))*(r1/abs(r1))
    if abs(r2) > 90:
        r2 = (180-abs(r2))*(r2/abs(r2))
    
    r1 = r1 / 90
    r2 = r2 / 90
    r = [r1, r2]
    fid[0] = ftype2fid(r1)
    fid[1] = ftype2fid(r2)
    
    l[0] = dist2oblique(r1)
    l[1] = dist2oblique(r2)
    
    pos = np.argmax[l]
    
    return r[pos], fid[pos]



        
    
def DistLatLon3(lat0, lon0, lats, lons):
    distance_R = np.zeros(len(lats))
    for ii in range(len(lats)):
        distance_R[ii] = hs.haversine((lat0, lon0), (lats[ii], lons[ii]))# in km
    return distance_R

def DistLatLon2(lat0, lon0, lats, lons):
    distance_R = np.zeros(len(lats))
    for ii in range(len(lats)):
        [distance_R[ii], _, _] = gps2dist_azimuth(lat0, lon0, lats[ii], lons[ii], a=6378137.0, f=0.0033528106647474805)
    distance_R = distance_R / 1000
    return distance_R

def InterIntra(fms, lon0, lat0, m0, t0):
    # return if faulting is:
    # interplate: pos = 1
    # intraplate: pos = -1
    if t0 == UTCDateTime('2003-08-04'):  # Scotia plate boundary
        pos = 1
    elif t0 == UTCDateTime('1977-06-22'): # Pacific plat
        pos = -1
    elif t0 == UTCDateTime('1981-05-25'): # Macquarie ridge strike-slip
        pos = 1
    elif t0 == UTCDateTime('1984-02-07'): # oblique Solomon Islands
        pos = -1
    elif t0 == UTCDateTime('1990-07-16'): # Philippine fault within Luzon
        pos = -1
    elif t0 == UTCDateTime('1993-08-08'): # Guam event; intraslab
        pos = -1
    elif t0 == UTCDateTime('1995-04-07'): # slab tearing in N. Tonga
        pos = -1
    elif t0 == UTCDateTime('1994-07-13'): # Vanuatu intraslab
        pos = -1
    elif t0 == UTCDateTime('2004-09-05'):  # outer rise thrust fault
        pos = -1
    elif t0 == UTCDateTime('2005-08-16'):  # Miyagi-Oki earthquake
        pos = 1
    elif t0 == UTCDateTime('2006-05-03'):  # Tonga is intraslab earthquake
        pos = -1
    elif t0 == UTCDateTime('2009-01-15'):  # slab compressional
        pos = -1
    elif t0 == UTCDateTime('2009-09-29'):  # outer rise extension
        pos = -1
    elif t0 == UTCDateTime('2012-04-11'):  # strike-slip in Indian plate
        pos = -1
    elif t0 == UTCDateTime('2012-08-31'):  # compressional event in Pacific plate
        pos = -1
    elif t0 == UTCDateTime('2017-07-17'):  # Komandorsky strike-slip on sliver
        pos = 1
    elif t0 == UTCDateTime('2018-02-25'):  # Papua intraplate compressional event
        pos = -1
    else:
        [r_predicted, _] = WnCfaultL(m0, ifault=0)
        fact_r = AdjustEffectiveRadius(m0)
        r_predicted = fact_r * r_predicted / 1000
        fms2 = aux_plane(fms[0], fms[1], fms[2])
        fstyle1, fstyle = ftypeShearer(fms[2], fms2[2])
        fnames = ['transform', 'ridge', 'trench']
        c = 'rgb'
        r = np.zeros(3)
        for pp in range(len(fnames)):
            shapfile1 = self.work_folder + 'shapefiles/Global_TL_Faults/%s.shp' % fnames[pp]
            X, Y = shape2xy2(shapfile1)
            X = np.concatenate(X)
            Y = np.concatenate(Y)
            r[pp] = min(DistLatLonUTM(lat0, lon0, Y, X)) / 1000

        if min(r) > r_predicted:
            pos = -1
        else:
            if fstyle > 0:
                if r[fstyle-1] < r_predicted:
                    pos = 1
                else:
                    pos = -1
            else:
                pos = -1
    return pos
            
        
        
        
        
def MakeCircleUTM(rm, lon0, lat0):
    e2u_zone = int(divmod(lon0, 6)[0])+31
    e2u_conv = Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
    #Apply the converter
    x0, y0=e2u_conv(lon0, lat0)
    a = np.arange(0, 360, 1)
    a = np.deg2rad(a)
    Yo1 = rm*np.cos(a) + y0
    Xo1 = rm*np.sin(a) + x0
    lon, lat = e2u_conv(Xo1, Yo1, inverse=True)
    xy = np.zeros((len(Yo1), 2))
    xy[:, 0] = lon
    xy[:, 1] = lat
    return lon, lat, xy

def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dtt(year=year, month=1, day=1)
    startOfNextYear = dtt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def DistLatLonUTM(lat0, lon0, lats, lons):
    e2u_zone=int(divmod(lon0, 6)[0])+31
    e2u_conv=Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
    #Apply the converter
    utmx, utmy=e2u_conv(lons, lats)
    utmx0, utmy0=e2u_conv(lon0, lat0)
    R = np.sqrt((utmx0 - utmx)**2 + (utmy0 - utmy)**2)
    return R
    

def DistLatLon(lat0, lon0, lats, lons):
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0) 
    lats = np.radians(lats) 
    lons = np.radians(lons)
    R = 6373.0
    dlon = lons - lon0
    dlat = lats - lat0

    a = np.sin(dlat / 2)**2 + np.cos(lat0) * np.cos(lats) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_R = R * c
    return distance_R
    

def DistFromEQ(cat0, ii):
    # R = 6373.0
    # lat1 = np.radians(cat0.Lat[ii])
    # lon1 = np.radians(cat0.Long[ii])
    # lat2 = np.radians(cat0.Lat)
    # lon2 = np.radians(cat0.Long)
    distance_R = DistLatLon(cat0.Lat[ii], cat0.Long[ii], cat0.Lat, cat0.Long)
    return distance_R.astype(int)


def get_cat_rate(times, dt):
    dtv = np.arange(min(times), max(times), dt)
    ll = len(dtv) - 1
    nEv = np.zeros((ll, 2))
    for ii in range(ll):
        nEv[ii,0] = sum(np.logical_and(times >=dtv[ii], times < dtv[ii+1]))
        nEv[ii,1] = dtv[ii+1] - dt/2
    return nEv

def set_box_aspect(ax, ratio):
    #ratio = 0.3
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


def WnCfaultL(m,ifault=0):
# Following Wells and Coppersmith 1994

    if ifault == 1:  # strike-slip
        al = 4.33; bl = 1.49; aw = 3.80; bw = 2.59
    elif ifault == 3:  # reverse
        al = 4.49; bl = 1.49; aw = 4.37; bw = 1.95
    elif ifault == 2:  # normal
        al = 4.34; bl = 1.54; aw = 4.04; bw = 2.11
    else:
        al = 4.38; bl = 1.49; aw = 4.06; bw = 2.25

    l = 10**((m - al)/bl) * 1000
    w = 10**((m - aw)/bw) * 1000
    return l, w

def AdjustEffectiveRadius(M):
    if np.size(M) >1:
        M = np.array(M)
        fact_r = np.ones(len(M)) * 2
        fact_r[M >=9.0] = 1
        fact_r[M >=8.0] = 1.5
    else:
        if M >= 9.0:
            fact_r = 1
        elif M > 8.0:
            fact_r = 1.5
        else:
            fact_r = 2.0
    return fact_r


def moveAVG(x,y,dx,lims,ol):
    v = np.arange(lims[0],lims[1],dx*ol)
    vN = np.zeros(len(v)-1)
    vX = np.zeros(len(v)-1)
    for ii in range(len(v)-1):
        I = np.logical_and(x>=v[ii],x<v[ii+1])
        vN[ii] = np.mean(y[I])
        vX[ii] = (v[ii]+v[ii+1])/2
    return vN,vX

# def binNforeNmain(m, n, bin):
#     minmaxM = np.array([minM, max(m)])
#     Mlow = np.floor(min(minmaxM))
#     Mhigh = np.ceil(max(minmaxM))
#     magrange = np.arange(Mlow, Mhigh+bin, bin)
#     nMrange = np.zeros(len(magrange)-1)
#     for ii in range(len(magrange)-1):
#         I_n = np.logical_and(m >= magrange[ii], m < magrange[ii+1])
#         nMrange[ii] = sum(n[I_n]) / sum(I_n)
#         

def binMag(m, minM, na):
    try:
        minmaxM = np.array([minM, max(m)])
        Mlow = np.floor(min(minmaxM))
        Mhigh = np.ceil(max(minmaxM))
        magrange = np.arange(Mlow, Mhigh+0.5,0.5)
        nMrange = np.zeros(len(magrange)-1)
        for ii in range(len(magrange)-1):
            I_n = np.logical_and(m>=magrange[ii], m<magrange[ii+1])
            nMrange[ii] = sum(na[I_n]) / sum(I_n)
        Mcenter = (magrange[1:] + magrange[0:-1]) / 2
    except:
        print('un-able to bin!')
        nMrange = 0
        Mcenter = 0
    I = nMrange != 0
    
    return nMrange[I], Mcenter[I]


def Swarmer(m0, m, t, minN):
    l = len(m)
    if l > 0:
        maxm = np.max(m)
        dm = m0 - maxm
        dt = np.abs(max(t) - min(t))
        rate = l / dt
        if l > minN:
            b = b_maximum_liklihood(m, 0.1)
        else:
            b = -999
    else:
        dm = -999
        rate = -999
        b = -999
    
    return dm, b, rate    

def minmax(x):
    return min(x), max(x)

def WnCsynth(cat0, Mc, dM, DT, tstart):
    DT = datetime.timedelta(DT)
    M = cat0[:,1]
    t = cat0[:,0]
    t = t * 365*24*60*60
    t0 = []
    for ii in range(len(t)):
        t0.append(tstart + t[ii])
    t0 = np.array(t0)
    # sort cat by magnitude
    posM = np.argsort(M)
    posM = posM[::-1]
    Im = np.sort(M)[::-1] > Mc + dM
    posM = posM[Im]
    ll=len(posM)
    c = np.zeros(len(M))
    n = np.zeros(len(M))
    cid = 1
    lonlat = np.zeros((len(M),2))
    # lonlat[:,0] = cat0.Long
    # lonlat[:,1] = cat0.Lat
    for ii in range(ll):
        # print('%d / %d' % (ii, ll))
        # SD = 3 #[MPa]
        # r_predicted = ((7*10**(1.5*cat0.M[posM[ii]]+10.73))/(16*SD*10**6))**(1/3)/2
        # r_predicted = 2 * 10**(0.59*cat0.M[posM[ii]])
        # [r_predicted,_] = WnCfaultL(M[posM[ii]],ifault=0)
        # adjust effective radius
        # fact_r = AdjustEffectiveRadius(M[posM[ii]])
        # r_predicted = fact_r * r_predicted
        # Ro = kilometers2degrees(r_predicted / 1000)
        # [xr, yr, xyr] = makeCircle(Ro,cat0.Long[posM[ii]],cat0.Lat[posM[ii]])
        # p = path.Path(xyr)
        # Ir = p.contains_points(lonlat)
        Ic = np.logical_and(t0>t0[posM[ii]] - DT, t0<t0[posM[ii]] + DT)
        # Irt = np.logical_and(Ir, It)
        Ino_c = c == 0
        # Ic = np.logical_and(Ino_c, Irt)
        c[Ic] = cid
        n[posM] = sum(Ic)
        cid = cid + 1
        clist = np.arange(1,cid,1)
    print('done making clusters!')
    return n,c,posM,clist

def get_nFSASsynth(CAT, clist, c, posM):
    n_AS = []
    n_FS = []
    for ii in range(len(clist)):
        clust = CAT[c == clist[ii]]
        n_AS.append(sum(clust[:,0] > CAT[posM[ii],0]))
        n_FS.append(sum(clust[:,0] < CAT[posM[ii],0]))
    n_AS = np.array(n_AS)
    n_FS = np.array(n_FS)
    print('Total number of MS with AS: %d' % sum(n_AS > 0))
    print('Total number of MS with FS: %d' % sum(n_FS > 0))
    
    return n_FS, n_AS


def plot_k_hist(k_yes,k_no, axb):
    # set histogram range
    nbins = 30
    minmaxK = minmax(np.concatenate([k_yes,k_no]))
    hist_edges = np.arange(minmaxK[0],minmaxK[1],abs(np.diff(minmaxK)/nbins))

    # Make histograms
    axb.hist(k_yes, hist_edges, orientation="horizontal",color=cmap(1), alpha=0.5,edgeColor = 'black')
    axb.hist(k_no,  hist_edges, orientation="horizontal",color=cmap(3), alpha=0.5,edgeColor = 'black')

    k_yes95 = np.quantile(k_yes, 0.95)
    k_yes05 = np.quantile(k_yes, 0.05)
    k_no95 = np.quantile(k_no, 0.95)
    k_no05 = np.quantile(k_no, 0.05)

    xlims = axb.get_xlim()
    axb.plot(xlims, [k0_yes,k0_yes],c='k')
    axb.plot(xlims, [k0_no,k0_no],c='k')
    axb.plot(xlims, [k_yes95,k_yes95],c=cmap(1))
    axb.plot(xlims, [k_no95,k_no95],c=cmap(3))
    axb.plot(xlims, [k_yes05,k_yes05],c=cmap(1))
    axb.plot(xlims, [k_no05,k_no05],c=cmap(3))
    plb.yticks(rotation=45)
    


def plot_productivity_synth(m, na, nf, Mc, dM, DT,nbs, fig):
    cmap = cm.get_cmap('cool',5)
    grid = plb.GridSpec(2, 9)

    ax = fig.add_subplot(grid[9: 11])
    axb = fig.add_subplot(grid[11])

    minmaxM = np.array([Mc+dM, max(m)])
    
    IyesF = nf > 0
    InoF = nf == 0
    
    # fit k for const a - all
    k1,a1,rms1,res1,r2 = fitK(m, na, Mc, 1)
    
    lb1 = 'k:%2.3f r2:%2.2f' % (k1, r2)
    ax.plot(minmaxM, k1*10**(a1*minmaxM - Mc),c='k', alpha=0.5,label=lb1)
    
    # fit k for const a - yes foreshocks
    k0_yes,a1,rms_k_yes,_,r2_yes = fitK(m[IyesF], na[IyesF], Mc, 1)
    
    # fit k for const a - no foreshocks
    k0_no,a1,rms_k_no,_ ,r2_no= fitK(m[InoF], na[InoF], Mc, 1)
    
    # bootstrap data
    k_yes, a_yes, _ = bootstrap(m[IyesF],na[IyesF],Mc,1, nbs)
    k_no,  a_no,  _ = bootstrap(m[InoF], na[InoF], Mc,1, nbs)

    # set histogram range
    nbins = 30
    minmaxK = minmax(np.concatenate([k_yes,k_no]))
    hist_edges = np.arange(minmaxK[0],minmaxK[1],abs(np.diff(minmaxK)/nbins))

    # Make histograms
    axb.hist(k_yes, hist_edges, orientation="horizontal",color=cmap(1), alpha=0.5,edgeColor = 'black')
    axb.hist(k_no,  hist_edges, orientation="horizontal",color=cmap(3), alpha=0.5,edgeColor = 'black')

    k_yes95 = np.quantile(k_yes, 0.95)
    k_yes05 = np.quantile(k_yes, 0.05)
    k_no95 = np.quantile(k_no, 0.95)
    k_no05 = np.quantile(k_no, 0.05)

    xlims = axb.get_xlim()
    axb.plot(xlims, [k0_yes,k0_yes],c='k')
    axb.plot(xlims, [k0_no,k0_no],c='k')
    axb.plot(xlims, [k_yes95,k_yes95],c=cmap(1))
    axb.plot(xlims, [k_no95,k_no95],c=cmap(3))
    axb.plot(xlims, [k_yes05,k_yes05],c=cmap(1))
    axb.plot(xlims, [k_no05,k_no05],c=cmap(3))
    plb.yticks(rotation=45)


    lb_yes = 'With FS - k:%2.3f r2:%2.2f' % (k0_yes, r2_yes)
    ax.plot(minmaxM, k0_yes*10**(a1*minmaxM - Mc),c=cmap(1), alpha=0.5,label=lb_yes)
    ax.scatter(m[IyesF], na[IyesF], c=cmap(1), alpha=0.5, edgecolors='k')

    lb_no = 'No FS - k:%2.3f r2:%2.2f' % (k0_no, r2_no)
    ax.plot(minmaxM, k1*10**(a1*minmaxM - Mc),c=cmap(3), alpha=1.0,label=lb_no)
    ax.scatter(m[InoF], na[InoF], facecolors='none', edgecolors=cmap(3), alpha=1.0)

    ax.set_xlabel('Mainshock magnitud Mw')
    ax.set_ylabel('Number of aftershocks %d days' % DT)
    ax.set_yscale('log')
    ax.set_ylim([1, 2*max(na)])
    ax.set_xlim(minmaxM)

    # fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99)

def degrees2kilometers(degrees, radius=6371):
    """
    Convenience function to convert (great circle) degrees to kilometers
    assuming a perfectly spherical Earth.

    :type degrees: float
    :param degrees: Distance in (great circle) degrees
    :type radius: int, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in kilometers as a floating point number.

    .. rubric:: Example

    >>> from obspy.geodetics import degrees2kilometers
    >>> degrees2kilometers(1)
    111.19492664455873
    """
    return degrees * (2.0 * radius * math.pi / 360.0)

def Calc_Area(x,y,dxy,n):
    """

    :param x: long - degrees
    :param y: lat - degrees
    :param dxy: grid cell dimentions in degrees
    :param n: ratio from maximum seismicity in a grid cell
    :return: area
    """
    x = np.array(x)
    y = np.array(y)
    X = np.arange(min(x),max(x)+dxy,dxy)
    Y = np.arange(min(y),max(y)+dxy,dxy)
    N = np.zeros((len(Y),len(X)))
    xyn = np.zeros(((len(Y)-1)*(len(X)-1),3))
    j = 0
    for xx in range(len(X)-1):
        Ix = np.logical_and(x>=X[xx],x<X[xx+1])
        for yy in range(len(Y)-1):
            Iy = np.logical_and(y>=Y[yy],y<Y[yy+1])
            N[yy,xx] = sum(np.logical_and(Iy,Ix))
            xyn[j,0] = (X[xx] + X[xx+1]) / 2
            xyn[j,1] = (Y[yy] + Y[yy+1]) / 2
            xyn[j,2] = N[yy,xx]
            j = j + 1
    IN = N>n*N.max()
    dxyKM = degrees2kilometers(dxy)
    area = sum(sum(IN))*dxyKM*dxyKM
    return area, N, X, Y, xyn, IN


def set_legend_title(ax, txt, sz, fam, pos='upper left'):
    ax.legend(title=txt, loc=pos, title_fontsize=sz)
    leg = ax.get_legend()
    plb.setp(leg.get_title(), family=fam)

def polygon_area(lats, lons, radius = 6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """
    from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)

    # Line integral based on Green's Theorem, assumes spherical Earth

    #close polygon
    if lats[0]!=lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    #colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
    colat = 2*arctan2( sqrt(a), sqrt(1-a) )

    #azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas=diff(colat)/2
    colat=colat[0:-1]+deltas

    # Perform integral
    integrands = (1-cos(colat)) * daz

    # Integrate 
    area = abs(sum(integrands))/(4*pi)

    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return area * 4*pi*radius**2
    else: #return in ratio of sphere total area
        return area

def isPrime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False

    return True

def numSubplot(n):
    while np.logical_and(isPrime(n), n >4):
        n= n+1


def bootstrap(m,n,mc,a,nbs):
    l      = len(m)
    a_bs   = np.zeros(nbs)
    k_bs   = np.zeros(nbs)
    rms_bs = np.zeros(nbs)

    for ii in range(nbs):
        randpos = np.random.randint(l, size=l)
        k_bs[ii], a_bs[ii], rms_bs[ii], _,_ = fitK(m[randpos],n[randpos],mc,a)
    return k_bs, a_bs, rms_bs

def shape2xy2(shapfile1):
    shape = shapefile.Reader(shapfile1)
    sp = shape.shapes()
    X = []
    Y = []
    for jj in range(len(sp)):
        x = []
        y = []
        for i in range(len(sp[jj].points)):
            x.append(sp[jj].points[i][0])
            y.append(sp[jj].points[i][1])
        x = np.array(x)
        y = np.array(y)
        X.append(x)
        Y.append(y)
    return X, Y

def pltshape(ax, shapfile1, c, ccrs=False, Lons=False, Lats=False):
    shape = shapefile.Reader(shapfile1)
    sp = shape.shapes()
    for jj in range(len(sp)):
        x = []
        y = []
        for i in range(len(sp[jj].points)):
            x.append(sp[jj].points[i][0])
            y.append(sp[jj].points[i][1])
        x = np.array(x)
        y = np.array(y)
        if Lons != False:
            Ilon = np.logical_and(x>min(Lons), x<max(Lons))
        else:
            Ilon = np.full(len(x), True)

        if Lats != False:
            Ilat = np.logical_and(y>min(Lats), y<max(Lats))
        else:
            Ilat = np.full(len(x), True)
        x=np.array(x)
        y=np.array(y)
        x = x[Ilat & Ilon]
        y = y[Ilat & Ilon]


        if ccrs == False:
            ax.plot(x, y, c=c)
        else:
            ax.plot(x, y, linewidth=2, c=c, transform=ccrs)


def get_ll_domain(t0, work_folder):
    t1 = t0.timestamp / (24*3600) + 719529
    ll = pd.read_csv(work_folder + 'LingLingTable2.csv')
    dt = np.abs(ll.datenum - t1).values
    min_dt = np.min(dt)
    pos_dt = np.argmin(dt)
    domain = -1
    if min_dt < 0.005:
        domain_s = ll.Domain[pos_dt][1]
        if domain_s == 'A':
            domain = 1
        elif domain_s == 'B':
            domain = 2
        elif domain_s == 'C':
            domain = 3
    return domain
        
        
def shape2xy(shapfile1):
    shape = shapefile.Reader(shapfile1)
    sp = shape.shapes()
    x = []
    y = []
    for i in range(len(sp[0].points)):
        x.append(sp[0].points[i][0])
        y.append(sp[0].points[i][1])
    x = np.array(x)
    y = np.array(y)
    return x, y

def define_Event_type(deCLUST):
    eType = np.zeros(len(deCLUST.MAG))
    I = deCLUST.clID > 0
    eType[I] = 1
    clist = np.unique(deCLUST.clID)
    clist = clist[clist > 0]
    pos = np.arange(len(deCLUST.MAG))
    for jj in range(len(clist)):
        I = deCLUST.clID == clist[jj]
        ms = deCLUST.MAG[I]
        ts = deCLUST.Time[I].values
        t0 = UTCDateTime(ts[np.argmax(ms)])
        eType[np.argmax(ms)] = 2
        posI = pos[I]

        otc=[]
        for ii in range(len(ts)):
            otc.append(UTCDateTime(ts[ii]))
        otc = np.array(otc)
        selFS = otc < t0
        eType[posI[selFS]] = -1
    deCLUST['eType'] = eType
    return deCLUST

def fitK(m,n,mc,a):
    if len(n) >1:
        n = np.asarray(n)
        m = np.asarray(m)

        # n=n.astype(np.float32)
        # np.place(n, n == 0.0, 1.0E-5)
        I = n > 0
        n = n[I]
        m = m[I]
        mn = m - mc
        logn = np.log10(n)
        if a != -999:
            k = np.mean(logn - a*mn)
        else:
            fit = np.polyfit(mn,logn,1)
            a = fit[0]
            k = fit[1]
        k = 10**k
        Npred = k*10**(a*mn)
        # k of each sequence
        ki = 10**(logn - a*mn)
        # calc RMS
        ks = logn / (a*mn)
        if a != 0:
            res = ks - k
        else:
            res = ki - k
        rms = np.sqrt(np.mean(res**2))
        logn_pred = np.log10(Npred)
        RSS = np.sum((logn-logn_pred)**2) # sum of squares residuals
        TSS = np.sum((logn - np.mean(logn))**2) # total sum of squares
        r2 = 1 - RSS/ TSS
    else:
        k = 0
        rms = 0
        res = 0
        r2 = 0
    return k, a, rms, res, r2

def Make180(x):
    x = np.array(x)
    I = x > 180.0
    x[I] = 180 - x[I]
    return x

def Make360(x):
    x = np.array(x)
    I = x < 0
    x[I] = x[I] + 360
    return x

def datenum2time(datenum):
    # t = datenum.apply(lambda x: (x-719529.0)*24*3600*1e9)
    # A = t.apply(lambda x: datetime.datetime.utcfromtimestamp(x*1e-9))
    t = datenum.apply(lambda x: (x-719529.0)*24*3600)
    A = t.apply(lambda x: UTCDateTime(x))
    A = np.asarray(A)
    return A

def Unix2datetime(t):
    DateNum = t.apply(lambda x: x / (24*3600) + 719529)
    tot = t.apply(lambda x: UTCDateTime(x))
    # tot = t.apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%S'))
    # tot = tot.apply(lambda x: UTCDateTime(x))
    return np.array(tot), np.array(DateNum)

    
def makeCircle(Ro,x0,y0):
    a = np.arange(0, 360, 1)
    a = np.deg2rad(a)
    Yo1 = Ro*np.cos(a) + y0
    Xo1 = Ro*np.sin(a) + x0
    xy = np.zeros((len(Yo1),2))
    xy[:,0] = Xo1
    xy[:,1] = Yo1
    return Xo1, Yo1, xy


def Mw2M0(M):
    M0 = 10**(M*1.5 + 10.7)
    return M0

def trim_lat_lon(latlims,lonlims,Mmin, Lats,Lons,M):
    Ilat1 = Lats>=latlims[0]
    Ilat2 = Lats<=latlims[1]
    Ilon1 = Lons>=lonlims[0]
    Ilon2 = Lons<=lonlims[1]
    Ilat = np.logical_and(Ilat1,Ilat2)
    Ilon = np.logical_and(Ilon1,Ilon2)
    I = np.logical_and(Ilat,Ilon)
    Im = M>= Mmin
    return np.logical_and(I,Im)

def read_SCEDC(file):
    CAT = CCats()
    # Datetime,ET,GT,MAG,M,LAT,LON,DEPTH,Q,EVID,NPH,NGRM,datenum
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.DEPTH)
    CAT.Long = np.asarray(A.LON)
    CAT.Lat = np.asarray(A.LAT)
    CAT.M = np.asarray(A.MAG)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT

def read_NCEDC(file):
    CAT = CCats()
    # DateTime,Latitude,Longitude,Depth,Magnitude,MagType,NbStations,Gap,Distance,RMS,Source,EventID,datenum
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.Depth)
    CAT.Long = np.asarray(A.Longitude)
    CAT.Lat = np.asarray(A.Latitude)
    CAT.M = np.asarray(A.Magnitude)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT

def read_ALASKA(file):
    CAT = CCats()
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.Depth)
    CAT.Long = np.asarray(A.Longitude)
    CAT.Lat = np.asarray(A.Latitude)
    CAT.M = np.asarray(A.Magnitude)
    CAT.datenum = np.asarray(A.datenum)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    return CAT

def read_smap(file):
    CAT = CCats()
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.depth)
    CAT.Long = np.asarray(A.Lon)
    CAT.Lat = np.asarray(A.Lat)
    CAT.M = np.asarray(A.M)
    CAT.datenum = A.datenum1
    CAT.ot = datenum2time(A.datenum1)
    CAT.N = np.arange(0, len(CAT.M),1)
    return CAT


def read_INGV(file):
    CAT = CCats()
    # DateTime,Latitude,Longitude,Depth,Magnitude,MagType,NbStations,Gap,Distance,RMS,Source,EventID,datenum
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.Depth)
    CAT.Long = np.asarray(A.Longitude)
    CAT.Lat = np.asarray(A.Latitude)
    CAT.M = np.asarray(A.Magnitude)
    CAT.ot, CAT.datenum = Unix2datetime(A.OriginTime)
    CAT.N = np.arange(0,len(CAT.M),1)
    return CAT

def read_INGV2(file):
    CAT = CCats()
    # DateTime,Latitude,Longitude,Depth,Magnitude,MagType,NbStations,Gap,Distance,RMS,Source,EventID,datenum
    
    A = pd.read_csv(file)
    A.Se[A.Se>59] = 59
    A.Mi[A.Mi>59] = 59
    
    x, y = shape2xy(self.work_folder + 'shapefiles/Italy_region.shp')

    I = inpolygon(A.Lon.values, A.Lat.values, x, y)
    CAT.Depth = np.asarray(A.Depth[I])
    CAT.Long = np.asarray(A.Lon[I])
    CAT.Lat = np.asarray(A.Lat[I])
    CAT.M = np.asarray(A.Mw[I])
    CAT.datenum = np.asarray(A.datenum[I])
    ot = datenum2time(A.datenum[I])
    CAT.ot = ot
    CAT.N = np.arange(0, len(CAT.M), 1)
    return CAT

def read_KOERI2(file):
    CAT = CCats()
    # No,EventID,Date,OriginTime,Latitude,Longitude,Depthkm,xM,MD,ML,Mw,Ms,Mb,Type,Location,datenum
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.inDEPTH)
    CAT.Long = np.asarray(A.LON)
    CAT.Lat = np.asarray(A.LAT)
    CAT.M = np.asarray(A.xTypMw)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT

def read_JMA(file):
    CAT = CCats()
    x, y = shape2xy(self.work_folder + 'shapefiles/Japan_inland.shp')
    # lat,lon,dep,datetime,m
    A = pd.read_csv(file)
    I = inpolygon(A.lon.values, A.lat.values, x, y)
    CAT.Depth = np.asarray(A.dep[I])
    CAT.Long = np.asarray(A.lon[I])
    CAT.Lat = np.asarray(A.lat[I])
    CAT.M = np.asarray(A.m[I])
    CAT.ot = datenum2time(A.datetime[I])
    CAT.N = np.arange(0, len(CAT.M), 1)
    CAT.datenum = np.asarray(A.datetime[I])
    return CAT

def read_NOA(file):
    CAT = CCats()
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.EPTH)
    CAT.Long = np.asarray(A.LONGD)
    CAT.Lat = np.asarray(A.LAT)
    CAT.M = np.asarray(A.Ml)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0, len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT


def read_AUOT(file):
    CAT = CCats()
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.Depth)
    CAT.Long = np.asarray(A.Lon)
    CAT.Lat = np.asarray(A.Lat)
    CAT.M = np.asarray(A.Mag)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT

def read_KOERI(file):
    CAT = CCats()
    # No,EventID,Date,OriginTime,Latitude,Longitude,Depthkm,xM,MD,ML,Mw,Ms,Mb,Type,Location,datenum
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.Depthkm)
    CAT.Long = np.asarray(A.Longitude)
    CAT.Lat = np.asarray(A.Latitude)
    CAT.M = np.asarray(A.xM)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT

def read_OVSICORI(file):
    CAT = CCats()
    A = pd.read_csv(file)
    x, y = shape2xy(self.work_folder + 'shapefiles/CR_in3.shp')
    I = inpolygon(A.lon.values, A.lat.values, x, y)
    CAT.Depth = np.asarray(A.depth[I])
    CAT.Long = np.asarray(A.lon[I])
    CAT.Lat = np.asarray(A.lat[I])
    CAT.M = np.asarray(A.m[I])
    CAT.ot = datenum2time(A.datenum[I])
    CAT.N = np.arange(0, len(CAT.M), 1)
    CAT.datenum = np.asarray(A.datenum[I])
    return CAT

def read_CR(file):
    CAT = CCats()
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.depth)
    CAT.Long = np.asarray(A.lon)
    CAT.Lat = np.asarray(A.lat)
    CAT.M = np.asarray(A.m)
    ot = []
    for ii in range(len(A.lat)):
        ot.append(UTCDateTime(A.year[ii], A.month[ii], A.day[ii], A.hh[ii], A.mm[ii], A.sec[ii]))
    CAT.ot = np.array(ot)
    CAT.N = np.arange(0, len(CAT.M), 1)
    CAT.datenum = np.asarray(A.datenum)
    return CAT
    

def read_NEIC(file):
    CAT = CCats()
    # EventID,OriginID,MagID,Date,OriginTime,FirstPick,Longitude,Latitude,Depth,MinHorUncM,MaxHorUncM,MaxHorAzi,OriUncDesc,Magnitude,datenum
    A = pd.read_csv(file)
    CAT.Depth = np.asarray(A.Depth)
    CAT.Long = np.asarray(A.Longitude)
    CAT.Lat = np.asarray(A.Latitude)
    CAT.M = np.asarray(A.Magnitude)
    CAT.ot = datenum2time(A.datenum)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    CAT.strike = np.asarray(A.strike)
    CAT.dip = np.asarray(A.dip)
    CAT.rake = np.asarray(A.rake)
    return CAT


def read_GCMT(file):
    CAT = CCats()
    # catalog,year,month,day,hour,minute,seconds,latitude,longitude,depth,mb,ms,location,name,data1type,data1nstn,data1ncmp,data1sper,data2type,data2nstn,data2ncmp,data2sper,data3type,data3nstn,data3ncmp,data3sper,srcinvtype,srcfunctype,srcfuncdur,centroidstring,centroidtime,centroidtimeerr,centroidlat,centroidlaterr,centroidlon,centroidlonerr,centroiddep,centroiddeperr,depthtype,timestamp,exponent,mrr,mrrerr,mtt,mtterr,mpp,mpperr,mrt,mrterr,mrp,mrperr,mtp,mtperr,version,eigval1,plunge1,azimuth1,eigval2,plunge2,azimuth2,eigval3,plunge3,azimuth3,scalarmoment,strike1,dip1,rake1,strike2,dip2,rake2,datenum,FMS_Type,Mw,IdE
    A = pd.read_csv(file,delimiter=',')
    CAT.Depth = np.asarray(A.depth)
    CAT.Long = np.asarray(A.longitude)
    CAT.Lat = np.asarray(A.latitude)
    CAT.M = np.asarray(A.Mw2)
    CAT.ot = datenum2time(A.datenum)
    CAT.strike = np.asarray(A.strike1)
    CAT.dip = np.asarray(A.dip1)
    CAT.rake = np.asarray(A.rake1)
    CAT.N = np.arange(0,len(CAT.M),1)
    CAT.datenum = np.asarray(A.datenum)
    fstyle,dP,dT,dB = mechaStyle(CAT.strike, CAT.dip, CAT.rake)
    CAT.fstyle = fstyle
    return CAT

def m2m0(m):
    m0 = 10**(1.5*(m + 10.73))
    return m0

def calc_foreshocks_time_scated(CLUST, deCLUST, minM):
    class f_data():
        pass
    foreshocks_times = []
    foreshocks_times_01 = []
    magnitudes = []
    for ii in range(len(CLUST)):
        if np.logical_and(CLUST.c_type[ii] == 1, CLUST.nf[ii] > 0):
            if CLUST.m0[ii] >= minM:
                I = deCLUST['clID'] == CLUST.cid[ii]
                clust = deCLUST[I]
                t0 = UTCDateTime(CLUST.ot0[ii])
                Time = []
                M = []
                for jj in range(len(clust['Time'])):
                    Time.append(UTCDateTime(clust['Time'].values[jj]))
                    M.append(clust['MAG'].values[jj])
                Time = np.array(Time)
                M = np.array(M)
                Times = (Time - t0) / (3600 * 24)
                for jj in range(len(Times)):
                    Times[jj] = np.round(Times[jj], 1)

                Ifs = Times < 0

                try:
                    Times01 = Times[Ifs] / min(Times[Ifs])
                    foreshocks_times.append(np.array(Times)[Ifs])
                    magnitudes.append(M[Ifs])
                    foreshocks_times_01.append(Times01)
                except:
                    pass


    # foreshocks_time = np.array(foreshocks_time)
    # foreshocks_times_01 = np.array(foreshocks_times_01)
    return foreshocks_times, foreshocks_times_01, magnitudes
    

def calc_k_clust(clust, Mc, dM):
    class k_data():
        pass
    m = clust['m0'].values
    na = clust['na_all'].values
    nf = clust['nf_all'].values
    tp = clust['c_type'].values
    Idm = tp == 1
    Iall = np.logical_and(Idm, m >= Mc + dM)
    Ifs = np.logical_and(Iall, nf >= 1)
    Inofs = np.logical_and(Iall, nf == 0)
    # fit k for const a
    k_data_all = k_data()
    k_data_all.k, k_data_all.a, k_data_all.rms, k_data_all.res,_ = fitK(m[Iall], na[Iall], Mc, 1)
    k_data_fs = k_data()
    k_data_fs.k, k_data_fs.a, k_data_fs.rms, k_data_fs.res,_ = fitK(m[Ifs], na[Ifs], Mc, 1)
    k_data_nofs = k_data()
    k_data_nofs.k, k_data_nofs.a, k_data_nofs.rms, k_data_nofs.res,_ = fitK(m[Inofs], na[Inofs], Mc, 1)
    return k_data_all, k_data_fs, k_data_nofs


def calc_cforeshocks_duration(CLUST, deCLUST, minM):
    foreshocks_time = []
    for ii in range(len(CLUST)):
        if np.logical_and(CLUST.c_type[ii] == 1, CLUST.nf[ii] > 0):
            if CLUST.m0[ii] >= minM:
                I = deCLUST['clID'] == CLUST.cid[ii]
                clust = deCLUST[I]
                t0 = UTCDateTime(CLUST.ot0[ii])
                Time = []
                for jj in range(len(clust['Time'])):
                    Time.append(UTCDateTime(clust['Time'].values[jj]))
                Time = np.array(Time)
                Time = (Time - t0) / (3600 * 24)
                foreshocks_time.append(min(Time))
    foreshocks_time = np.array(foreshocks_time)
    return foreshocks_time


class CatData:
    def __init__(self, cat_names, work_folder, ddays, dM, minN4b,dmSwarm, nbs,max_depth, MCP):
        filefolder = work_folder+'Catalogs_csv/'
        if MCP > -999:
            fileRes = filefolder + 'Mc%d/' % round(MCP * 10)
        else:
            fileRes = filefolder
        self.work_folder = work_folder
        self.main_folder = filefolder
        self.DT = datetime.timedelta(days=ddays)
        self.dM = dM
        self.ddays = ddays
        self.fileRes = fileRes
        self.minN4b = minN4b
        self.dmSwarm = dmSwarm
        self.nbs = nbs
        self.max_depth = max_depth


        strat_time = time.time()
        CATS = []
        for ii in range(len(cat_names)):
            cat = self.GetCat(cat_names[ii])
            CATS.append(cat)
        end_time = time.time()

        print('Catalog loading time: %5.2f sec' % (end_time - strat_time))

        self.CATS = CATS

    def calc_rates_bg_fs_as(self, ic):
        print('Calc rates for %s' % self.CATS[ic].name)
        deCLUST = self.CATS[ic].deCLUST
        CLUST = self.CATS[ic].CLUST
        rates_bg = np.zeros(len(CLUST))
        rates_fs = np.zeros(len(CLUST))
        rates_as = np.zeros(len(CLUST))
        time_as = np.zeros(len(CLUST))
        time_fs = np.zeros(len(CLUST))
        for ii in range(len(CLUST)):
            I = deCLUST['clID'] == CLUST.cid[ii]
            clust = deCLUST[I]
            # calc cluster size - Rm0
            if sum([CLUST.na_all[ii] > 0, CLUST.nf_all[ii] > 0]) > 0:
                Rm0 = max(DistLatLonUTM(CLUST.lat0[ii], CLUST.lon0[ii], clust.Lat, clust.Lon))

                # calc cluster foreshocks duration
                t0 = UTCDateTime(CLUST.ot0[ii]).timestamp
                time_clust = np.array([UTCDateTime(jj).timestamp for jj in clust['Time'].values])
                foreshocks_duration_sec = min(time_clust - t0) / (3600*24)
                aftershocks_duration_sec = max(time_clust - t0) / (3600*24)
                # calc background rate
                T = np.array([UTCDateTime(jj).timestamp for jj in deCLUST['Time'].values])
                R = DistLatLonUTM(CLUST.lat0[ii], CLUST.lon0[ii], deCLUST.Lat, deCLUST.Lon)
                Ir = R < Rm0
                It = t0 - (T + foreshocks_duration_sec) < 0
                Irt = np.logical_and(It, Ir)
                if sum(Irt) > 0:
                    Tbg = np.array(T)[Irt]
                    rates_bg[ii] = len(Tbg)/((max(Tbg) - min(Tbg)) / (3600*24))
                rates_fs[ii] = CLUST.nf_all[ii] / foreshocks_duration_sec
                rates_as[ii] = CLUST.na_all[ii] / aftershocks_duration_sec
                time_as[ii] = aftershocks_duration_sec
                time_fs[ii] = foreshocks_duration_sec
        CLUST['rates_bg'] = np.abs(rates_bg)
        CLUST['rates_fs'] = np.abs(rates_fs)
        CLUST['rates_as'] = np.abs(rates_as)
        CLUST['time_fs'] = np.abs(time_fs)
        CLUST['time_as'] = np.abs(time_as)
        return CLUST

    def InterIntra(self, loader):
        if loader == False:
            nCats = len(self.CATS)
            for ic in range(nCats):
                try:
                    clust = self.CATS[ic].CLUST
                    cat = self.CATS[ic].cat
                    m = clust['m0'].values
                    pos = np.zeros(len(m))
                    fstyle = np.zeros(len(m))
                    for ii in range(len(m)):
                        t0 = UTCDateTime(clust['ot0'][ii])
                        m0 = clust['m0'][ii]
                        # get maisnhocks FMS
                        Idate = (cat.ot > t0-3600) & (cat.ot < t0+3600)
                        Im = (cat.M > (m0 - 0.5)) & (cat.M < (m0 + 0.5))
                        Ifms = Idate & Im
                        
                        fms = np.zeros(3)
                        fms[0] = cat.strike[Ifms][0]
                        fms[1] = cat.dip[Ifms][0]
                        fms[2] = cat.rake[Ifms][0]
                        pos[ii] = InterIntra(fms, clust['lon0'][ii], clust['lat0'][ii], m0, t0.date)
                        fms2 = aux_plane(fms[0], fms[1], fms[2])
                        fstyle1, fstyle[ii] = ftypeShearer(fms[2], fms2[2])
                        
                    clust['InterIntra'] = pos
                    clust['fstyle'] = fstyle
                    if self.cmethod == 'ZnBZ':
                        nameC = '%scat_%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ic].name, self.dM, self.cmethod)
                    else:
                        nameC = '%s%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ic].name, self.dM, self.cmethod)
                    print('Iter / Intra done for: %s' % nameC)
                    clust.to_csv(nameC)
                except:
                    print('unable to add Inta/Inter data to %' % self.CATS[ic].name)
            
        
    def calc_fs_moment_ratios(self):
        nCats = len(self.CATS)
        print('Calc foreshocks moment ratios...')
        for ic in range(nCats):
            pos = np.argsort(self.CATS[ic].CLUST.m0)
            m01_2 = np.zeros(len(pos))
            for ii in range(len(pos)):
                if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                    I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                    clust = self.CATS[ic].deCLUST[I]

                    t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])

                    otc = []
                    for jj in range(len(clust['Lat'].values)):
                        otc.append(UTCDateTime(clust['Time'].values[jj]))
                    otc = np.array(otc)
                    selFS = otc < t0

                    if sum(selFS) >= 2:
                        try:
                            Mf = clust[selFS].MAG.values
                            tf = otc[selFS]
                            posF = np.argmax(Mf)
                            If1 = tf < tf[posF]
                            If2 = tf > tf[posF]
                            m01_2[pos[ii]] = sum(m2m0(Mf[If1])) - sum(m2m0(Mf[If2]))
                            # print('%s %f' % (t0, m01_2[pos[ii]]))
                        except:
                            m01_2[pos[ii]] = 0

            self.CATS[ic].CLUST['m01_2'] = m01_2



    def calc_mainshocks_rates(self, loader):
        nCats = len(self.CATS)
        print('Calc rates...')
        if loader == False:
            strat_time = time.time()
            with mp.Pool() as pool:
                results = pool.map(self.calc_rates_bg_fs_as, np.arange(nCats))
            for ic in range(nCats):
                self.CATS[ic].CLUST = results[ic]
                
                # write Cluster data
                nameC = '%s%s.clusters.all.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ic].name, self.dM, self.cmethod)
                print('Cluster name: %s' % nameC)
                results[ic].to_csv(nameC)
            else:
                # Load Cluster data
                nameC = '%s%s.clusters.all.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ic].name, self.dM, self.cmethod)
                print('Loading Cluster name: %s' % nameC)
                try:
                    CLUST = pd.read_csv(nameC)
                except:
                    print('all clust %s not found' % nameC)
                    
                self.CATS[ic].CLUST = CLUST

        if loader == False:
            end_time = time.time()
            print('time to run background: %5.2f sec' % (end_time - strat_time))

    

    def calc_k_clusters(self, plotter):
        nCats = len(self.CATS)
        if plotter == True:
            fig = plb.figure(6320)
        for ic in range(nCats):
            clust = self.CATS[ic].CLUST
            Mc = self.CATS[ic].b_data.Mc
            k_data_all, k_data_fs, k_data_nofs = calc_k_clust(clust, Mc, self.dM)
            
            self.CATS[ic].k_data_all = k_data_all
            self.CATS[ic].k_data_fs = k_data_fs
            self.CATS[ic].k_data_nofs = k_data_nofs
            if plotter == True:
                m = clust['m0'].values
                na = clust['na_all'].values
                nf = clust['nf_all'].values
                tp = clust['c_type'].values
                Idm = tp == 1
                Iall = np.logical_and(Idm, m >= Mc + self.dM)
                Ifs = np.logical_and(Iall, nf >= 1)
                minmaxM = np.array([Mc+self.dM, max(m)])
                lbl_all = 'All k:%2.3f rms:%2.2f' % (k_data_all.k, k_data_all.rms)
                lbl_fs = 'With FS k:%2.3f rms:%2.2f' % (k_data_fs.k, k_data_fs.rms)
                lbl_nofs = 'Without FS k:%2.3f rms:%2.2f' % (k_data_nofs.k, k_data_nofs.rms)
                ax = fig.add_subplot(4, 2, ic+1)
                ax.scatter(m, na, facecolors='none', edgecolors='k')
                ax.scatter(m[Ifs], na[Ifs], facecolors='b', edgecolors='none')
                ax.plot(minmaxM, k_data_all.k*10**(k_data_all.a*(minmaxM - Mc)), c='k', alpha=0.5, label=lbl_all)
                ax.plot(minmaxM, k_data_fs.k*10**(k_data_fs.a*(minmaxM - Mc)), c='b', alpha=0.5, label=lbl_fs)
                ax.plot(minmaxM, k_data_nofs.k*10**(k_data_nofs.a*(minmaxM - Mc)), c='k', ls='--', alpha=0.5, label=lbl_nofs)
                ax.set_xlabel('Mainshock Mw')
                ax.set_ylabel('Number of aftershocks')
                ax.set_yscale('log')
                ax.set_ylim([1, 1000])
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
        # if plotter == True:
            # fig.suptitle(self.cmethod)
            
        
    def MakaClusterData(self, ic):
        
        cat0     = self.CATS[ic].cat
        posM     = self.CATS[ic].posM
        clist    = self.CATS[ic].clist
        na       = []
        nf       = []
        na_apr   = []
        nf_apr   = []
        na_all   = []
        nf_all   = []
        na_all_apr=[]
        nf_all_apr=[]
        m0       = []
        lat0     = []
        lon0     = []
        depth0   = []
        ot0      = []
        cid      = []
        dm_a     = []
        dm_a_all = []
        b_a      = []
        rate_a   = []
        dm_f     = []
        dm_f_all = []
        b_f      = []
        rate_f   = []
        c_type   = []
        c_type60 = []
        fstyle_m = []
        for ii in range(len(posM)):
            if cat0.M[posM[ii]] >= self.CATS[ic].b_data.Mc + self.dM:
                cid.append(clist[ii])
                m0.append(cat0.M[posM[ii]])
                lat0.append(cat0.Lat[posM[ii]])
                lon0.append(cat0.Long[posM[ii]])
                depth0.append(cat0.Depth[posM[ii]])
                ot0.append(cat0.ot[posM[ii]])
                if self.CATS[ic].name == 'GCMT' or self.CATS[ic].name == 'GCMT2':
                    fstyle_m.append(cat0.fstyle[posM[ii]])
                Ic = cat0.c == clist[ii]
                Iaperture = cat0.M >= (cat0.M[posM[ii]] - self.dmSwarm)
                If_all = cat0.ot < ot0[-1]
                Ia_all = cat0.ot > ot0[-1]
                If = np.logical_and(cat0.ot < ot0[-1], cat0.ot > ot0[-1] - self.DT)
                Ia = np.logical_and(cat0.ot > ot0[-1], cat0.ot < ot0[-1] + self.DT)
                If = np.logical_and(Ic, If)
                Ia = np.logical_and(Ic, Ia)
                If_all = np.logical_and(Ic, If_all)
                Ia_all = np.logical_and(Ic, Ia_all)
                It_60 = np.logical_and(cat0.ot > ot0[-1] - self.DT, cat0.ot < ot0[-1] + self.DT)
                Ic_60 = np.logical_and(It_60, Ic)
                nf.append(sum(If))
                na.append(sum(Ia))
                nf_all.append(sum(If_all))
                na_all.append(sum(Ia_all))
                nf_apr.append(sum(np.logical_and(If, Iaperture)))
                na_apr.append(sum(np.logical_and(Ia, Iaperture)))
                nf_all_apr.append(sum(np.logical_and(If_all, Iaperture)))
                na_all_apr.append(sum(np.logical_and(Ia_all, Iaperture)))
        
                # check swarms
                dm, b, rate = Swarmer(cat0.M[posM[ii]], cat0.M[Ia], cat0.datenum[Ia], self.minN4b)
                dm_a.append(dm)
                b_a.append(b)
                rate_a.append(rate)
                dm, b, rate = Swarmer(cat0.M[posM[ii]], cat0.M[If], cat0.datenum[If], self.minN4b)
                dm_f.append(dm)
                b_f.append(b)
                rate_f.append(rate)
                c_type.append(sequence_type(cat0.M[Ic]))
                c_type60.append(sequence_type(cat0.M[Ic_60]))
                # c_type.append(sequence_type(cat0.M[np.logical_and(Ic, Ia_60)]))
                # c_type60.append(sequence_type(cat0.M[Ic_60]))
                try:
                    dm_f_all.append(max(cat0.M[posM[ii]] - cat0.M[If_all]))
                except:
                    dm_f_all.append(-999)
                try:
                    dm_a_all.append(max(cat0.M[posM[ii]] - cat0.M[Ia_all]))
                except:
                    dm_a_all.append(-999)

        if len(fstyle_m) == len(m0):
            clusters = {'cid':cid,'m0':m0, 'lat0':lat0, 'lon0':lon0, 'depth0':depth0,
                        'ot0':ot0, 'nf':nf, 'na':na,'nf_all':nf_all, 'na_all':na_all, 'dm_a':dm_a, 'b_a':b_a, 'rate_a':rate_a,
                        'dm_f':dm_f, 'b_f':b_f, 'rate_f':rate_f, 'c_type':c_type, 'c_type60':c_type60, 'fstyle_m':fstyle_m,
                        'na_apr':na_apr, 'nf_apr':nf_apr,'na_all_apr':na_all_apr, 'nf_all_apr':nf_all_apr}
            clustersFinal = {'m0':m0, 'lat0':lat0, 'lon0':lon0, 'depth0':depth0,
                        'ot0':ot0, 'nf':nf, 'na':na,'nf_all':nf_all, 'na_all':na_all,
                        'c_type':c_type, 'c_type60':c_type60, 'fstyle_m':fstyle_m,
                        'na_apr':na_apr, 'nf_apr':nf_apr,'na_all_apr':na_all_apr, 'nf_all_apr':nf_all_apr}
        else:
            clusters = {'cid':cid,'m0':m0, 'lat0':lat0, 'lon0':lon0, 'depth0':depth0,
                        'ot0':ot0, 'nf':nf, 'na':na,'nf_all':nf_all, 'na_all':na_all, 'dm_a':dm_a, 'b_a':b_a, 'rate_a':rate_a,
                        'dm_f':dm_f, 'b_f':b_f, 'rate_f':rate_f, 'c_type':c_type, 'c_type60':c_type60,
                        'na_apr':na_apr, 'nf_apr':nf_apr,'na_all_apr':na_all_apr, 'nf_all_apr':nf_all_apr}
            
            clustersFinal = {'m0':m0, 'lat0':lat0, 'lon0':lon0, 'depth0':depth0,
                        'ot0':ot0, 'nf':nf, 'na':na,'nf_all':nf_all, 'na_all':na_all,
                        'c_type':c_type, 'c_type60':c_type60,
                        'na_apr':na_apr, 'nf_apr':nf_apr,'na_all_apr':na_all_apr, 'nf_all_apr':nf_all_apr}

        CLUST = pd.DataFrame(data=clusters)
        CLUSTfinal = pd.DataFrame(data=clustersFinal)

        # Im = CLUST['m0'].values > self.CATS[ic].b_data.Mc + self.dM
        # CLUST = CLUST[Im]

        # write Cluster data
        nameC = '%s%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ic].name, self.dM, self.cmethod)
        print('Cluster name: %s' % nameC)
        nameCfinal = '%s%s.clusters.final.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ic].name, self.dM, self.cmethod)
        CLUST.to_csv(nameC)
        CLUSTfinal.to_csv(nameCfinal)
        print('Cdata saved: %s' % self.CATS[ic].name)

        self.CATS[ic].CLUST = CLUST


    def Plotter(self, Ptype, fig=None):
        nCats = len(self.CATS)
        if Ptype == 'AftershockProductivity':
            if fig==None:
                fig = plb.figure(1000)
                
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                na = clust['na'].values
                nf = clust['nf'].values
                minmaxM = np.array([Mc+self.dM, max(m)])
                dm_a = clust['dm_a'].values
                dm_f = clust['dm_f'].values
                Idm = np.logical_and(dm_a > self.dmSwarm, dm_f > self.dmSwarm)

                # fit k for const a
                k1,a1,rms1,res1,_ = fitK(m, na, Mc, 1)
                lb1 = 'k:%2.3f a:%2.2f rms:%2.2f' % (k1, a1, rms1)
                ax.plot(minmaxM, k1*10**(a1*(minmaxM - Mc)),c='k', alpha=0.5,label=lb1)
                # fit k and a
                k0,a0,rms0,res0 = fitK(m, na, Mc, -999)
                lb0 = 'k:%2.3f a:%2.2f rms:%2.2f' % (k0, a0, rms0)
                ax.plot(minmaxM, k0*10**(a0*(minmaxM - Mc)), c='k',ls='--', alpha=0.5, label=lb0)
                ax.scatter(m, na, c='k', alpha=0.5, label='All')
                ax.scatter(m[Idm], na[Idm], c='r', alpha=0.5,label='A.S. or F.S. > Mw0-%2.2f' % self.dmSwarm)
                ax.set_xlabel('Mainshock magnitud Mw')
                ax.set_ylabel('Number of aftershocks %d days' % self.DT.days)
                ax.set_yscale('log')
                ax.set_ylim([1, 1000])
                ax.set_xlim(minmaxM)
                ax.legend(title=self.CATS[ic].name0,loc='upper left')

        elif Ptype == 'b-values':
            fig50 = plb.figure(50)
            fig51 = plb.figure(51)
            for ic in range(nCats):
                try:
                    # M1 = self.CATS[ic].b_data.Mc - 2*self.CATS[ic].b_data.dm
                    M1 = min(self.CATS[ic].b_data.Mv)
                    M2 = max(self.CATS[ic].b_data.Mv) - 1.0
                    ax1 = fig50.add_subplot(2, 3, ic+1)
                    ax2 = fig51.add_subplot(2, 3, ic+1)
                    print_b_val(self.CATS[ic].b_data, M1, M2, ax1, ax2, self.CATS[ic].name)
                    ax1.set_title('%s - %s' % (str(min(self.CATS[ic].cat.ot).date),str(max(self.CATS[ic].cat.ot).date)))
                    
                except:
                    print('Can not plot b-value! try run with b-value calculation True')

        elif Ptype == 'AftershockWithForeshocksProductivity_c_type':
            cmap = cm.get_cmap('cool',5)
            fig = plb.figure(2020)
            grid = plb.GridSpec(3, 6)
            for ic in range(nCats):

                ax = fig.add_subplot(grid[3*ic: 3*ic+2])
                axb = fig.add_subplot(grid[3*ic+2])

                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                # if self.DT.days == 60:
                    na   = clust['na'].values
                    nf   = clust['nf'].values
                    tp   = clust['c_type60'].values
                    days_cluster = self.DT.days
                else:
                    na   = clust['na_all'].values
                    nf   = clust['nf_all'].values
                    tp   = clust['c_type'].values
                    days_cluster = np.inf

                Idm = tp == 1 ; types_c = 'Mainshocks'
                # Idm = np.logical_and(tp >= 1, tp <= 2); types_c = 'Mainshocks + Doublets'
                # Idm = np.logical_and(tp >= 1, tp <= 3); types_c = 'Mainshocks + Doublets + Triplets'
                
                IyesF = np.logical_and(nf > 0, Idm)
                InoF = nf == 0

                minmaxM = np.array([Mc+self.dM, max(m)])
                # fit k for const a - all
                k1,a1,rms1,res1,r2 = fitK(m, na, Mc, 1)
                lb1 = r'$k all:%2.3f\: r^{2}:%2.2f$' % (k1, r2)
                ax.plot(minmaxM, k1*10**(a1*(minmaxM - Mc)), c='k', alpha=0.5, label=lb1)
                
                # fit k for const a - yes foreshocks
                k0_yes,a1,rms_k_yes,_,r2_yes = fitK(m[IyesF], na[IyesF], Mc, 1)
                
                # fit k for const a - no foreshocks
                k0_no,a1,rms_k_no,_,r2_no = fitK(m[InoF], na[InoF], Mc, 1)
                
                # bootstrap data
                k_yes, a_yes, _ = bootstrap(m[IyesF],na[IyesF],Mc,1,self.nbs)
                k_no,  a_no,  _ = bootstrap(m[InoF], na[InoF], Mc,1,self.nbs)

                # set histogram range
                nbins = 30
                minmaxK = minmax(np.concatenate([k_yes,k_no]))
                hist_edges = np.arange(minmaxK[0],minmaxK[1],abs(np.diff(minmaxK)/nbins))

                # Make histograms
                axb.hist(k_yes, hist_edges, orientation="horizontal",color=cmap(1), alpha=0.5,edgeColor = 'black')
                axb.hist(k_no,  hist_edges, orientation="horizontal",color=cmap(3), alpha=0.5,edgeColor = 'black')

                k_yes95 = np.quantile(k_yes, 0.95)
                k_yes05 = np.quantile(k_yes, 0.05)
                k_no95 = np.quantile(k_no, 0.95)
                k_no05 = np.quantile(k_no, 0.05)

                xlims = axb.get_xlim()
                axb.plot(xlims, [k0_yes,k0_yes],c=cmap(1),linewidth=2)
                axb.plot(xlims, [k0_no,k0_no],c=cmap(3),linewidth=2)
                axb.plot(xlims, [k_yes95,k_yes95],c=cmap(1),linestyle=':')
                axb.plot(xlims, [k_no95,k_no95],c=cmap(3),linestyle=':')
                axb.plot(xlims, [k_yes05,k_yes05],c=cmap(1),linestyle=':')
                axb.plot(xlims, [k_no05,k_no05],c=cmap(3),linestyle=':')
                plb.yticks(rotation=45)


                lb_yes = r'$k with FS:%2.3f\: r^{2}:%2.2f$' % (k0_yes, r2_yes)
                ax.plot(minmaxM, k0_yes*10**(a1*(minmaxM - Mc)),c=cmap(1), alpha=0.5,label=lb_yes)
                ax.scatter(m[IyesF], na[IyesF], c=cmap(1), alpha=0.5, edgecolors='k')

                lb_no = r'$k No FS:%2.3f\: r^{2}:%2.2f$' % (k0_no, r2_no)
                ax.plot(minmaxM, k0_no*10**(a1*(minmaxM - Mc)),c=cmap(3), alpha=1.0,label=lb_no)
                ax.scatter(m[InoF], na[InoF], facecolors='none', edgecolors=cmap(3), alpha=1.0)

                ax.set_xlabel('Mainshock magnitud Mw')
                if ic ==0:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                elif ic == 2:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                elif ic == 4:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                ax.set_yscale('log')
                ax.set_ylim([1, 1000])
                ax.set_xlim(minmaxM)
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                # ax.legend(title=self.CATS[ic].name,loc='lower right')
            #fig.tight_layout(pad=3.0)
            # fig.suptitle('%s: Sequences - %s' % (self.cmethod ,types_c))
            fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99,bottom=0.05, top=0.95)

        elif Ptype == 'AftershockProductivity_types_c_aperture':
            fig = plb.figure(2028)
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                # if self.DT.days == 60:
                    na = clust['na_apr'].values
                    nf = clust['nf_apr'].values
                    tp   = clust['c_type60'].values
                    days_cluster = self.DT.days
                else:
                    tp   = clust['c_type'].values
                    na = clust['na_all_apr'].values
                    nf = clust['nf_all_apr'].values
                    days_cluster = np.inf
                    
                Idm = tp == 1 ; types_c = 'Mainshocks'
                # Idm = np.logical_and(tp >= 1, tp <= 2); types_c = 'Mainshocks + Doublets'
                # Idm = np.logical_and(tp >= 1, tp <= 3); types_c = 'Mainshocks + Doublets + Triplets'

                
                # plot binned
                nMrange, Mcenter = binMag(m,Mc+self.dM,na)
                minmaxM = np.array([Mc+self.dM, max(m)])
                # Mlow = np.floor(min(minmaxM))
                # Mhigh = np.ceil(max(minmaxM))
                # magrange = np.arange(Mlow, Mhigh+0.5,0.5)
                # nMrange = np.zeros(len(magrange)-1)
                # for ii in range(len(magrange)-1):
                #     I_n = np.logical_and(m>=magrange[ii], m<magrange[ii+1])
                #     nMrange[ii] = sum(na[I_n]) / sum(I_n)
                # Mcenter = (magrange[1:] + magrange[0:-1]) / 2

                # fit k for const a - all
                # k1, a1, rms1, res1, r2 = fitK(m[Idm], na[Idm], Mc, -999)
                k1, a1, rms1, res1, r2 = fitK(Mcenter, nMrange, Mc, -999)
                lb1 = 'a:%2.1f  r2:%2.2f' % (a1, r2)
                ax.plot(minmaxM, k1*10**(a1*(minmaxM - Mc)),c='k', alpha=0.5,label=lb1)
                ax.scatter(m[Idm], na[Idm], alpha=0.5, c=[0.5, 0.5 ,0.5], edgecolors='k')
                # ax.scatter(Mcenter, nMrange, marker='s', c='m')
                ax.plot(Mcenter, nMrange, '-sm')
                ax.set_xlabel('Mainshock magnitud Mw')
                if ic ==0:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                elif ic == 3:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                ax.set_yscale('log')
                ax.set_ylim([1, 100])
                ax.set_xlim(minmaxM)
                ax.legend(title=self.CATS[ic].name0,loc='lower right')
            fig.suptitle('%s: Sequences - %s' % (self.cmethod, types_c))
            fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99)
                
        elif Ptype == 'BG_timeline_rates':
            fig = plb.figure(6040)
            fig2 = plb.figure(6042)
            fig3 = plb.figure(6043)
            fig4 = plb.figure(6044)
            fig5 = plb.figure(6045)
            fig6 = plb.figure(6046)
            ax3 = fig3.add_subplot(1, 1, 1)
            for ic in range(nCats):
                clust = self.CATS[ic].CLUST
                ax = fig.add_subplot(4, 2, ic+1)
                ax2 = fig2.add_subplot(4, 2, ic+1)
                ax4 = fig4.add_subplot(4, 2, ic+1)
                ax5 = fig5.add_subplot(4, 2, ic+1)
                ax6 = fig6.add_subplot(4, 2, ic+1)
                for ii in range(len(clust)):
                    t0 = toYearFraction(UTCDateTime(clust['ot0'][ii]))
                    fs_bs = clust.rates_fs / clust.rates_bg
                    as_bs = clust.rates_as / clust.rates_bg
                    if ii == 0:
                        ax.scatter(t0, fs_bs[ii], 30, 'b', alpha=0.5, label='low FS / BG: %d%%' % (sum(np.logical_and(fs_bs < 1, fs_bs > 0)) / sum(fs_bs > 0) * 100), zorder=5)
                        ax.scatter(t0, as_bs[ii], 10, 'r', alpha=0.5, label='low AS / BG: %d%%' % (sum(np.logical_and(as_bs < 1, as_bs > 0)) / sum(as_bs > 0) * 100), zorder=10)
                    else:
                        ax.scatter(t0, fs_bs[ii], 30, 'b', alpha=0.5, zorder=5)
                        ax.scatter(t0, as_bs[ii], 10, 'r', alpha=0.5, zorder=10)
                    ax.plot([t0, t0], [0, max([clust.rates_fs[ii] / clust.rates_bg[ii], clust.rates_as[ii] / clust.rates_bg[ii]])], '-k', zorder=2)
                ax.set_yscale('log')
                ax.grid()
                xlims = ax.get_xlim()
                ax.plot(xlims, [1, 1], zorder=2)
                ax.set_xlim(xlims)
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')

                # ax2.scatter(fs_bs, as_bs)
                ax2.scatter(clust.rates_as, clust.rates_fs, alpha=0.5)
                ax2.set_ylabel('Foreshock rates')
                ax2.set_xlabel('Aftershock rates')
                ax2.grid()
                ax2.set_yscale('log')
                ax2.set_xscale('log')
                ax2_xlims = ax2.get_xlim()
                ax2_ylims = ax2.get_ylim()
                ax2.plot([min([min(ax2_xlims), min(ax2_ylims)]), max([max(ax2_xlims), max(ax2_ylims)])], [min([min(ax2_xlims), min(ax2_ylims)]), max([max(ax2_xlims), max(ax2_ylims)])], '-k')
                ax2.set_aspect(1)
                set_legend_title(ax2, self.CATS[ic].name0, 14, 'Impact')

                # ax3.scatter(clust.fs_bs, clust.as_bs, label=self.CATS[ic].name0)
                ax3.scatter(clust.rates_as, clust.rates_fs, alpha=0.5, label=self.CATS[ic].name0)
                
                ax4.scatter(clust['m0'][clust.rates_as > 0].values, clust.rates_as[clust.rates_as > 0].values, 10, 'r', alpha=0.5, zorder=10)
                ax4.scatter(clust['m0'][clust.rates_fs > 0].values, clust.rates_fs[clust.rates_fs > 0].values, 30, 'b', alpha=0.5, zorder=5)
                ax4.set_xlabel('Mainshock* Mw')
                ax4.set_ylabel('Rates #events/day')
                ax4.grid()
                ax4.set_yscale('log')
                set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')
                
                ax5.scatter(clust.time_as, clust.rates_as, 10, 'r', label='Aftershock', alpha=0.5, zorder=10)
                ax5.scatter(clust.time_fs, clust.rates_fs, 30, 'b', label='Foreshock', alpha=0.5, zorder=5)
                min_duration = min([min(clust.time_as[clust.time_as > 0]), min(clust.time_fs[clust.time_fs > 0])])
                max_duration = max([max(clust.time_as), max(clust.time_fs)])
                ax5.plot([min_duration, max_duration], [1 / min_duration, 1 / max_duration], '-k')
                ax5.set_xlabel('Duration (days)')
                ax5.set_ylabel('Rates #events/day')
                ax5.grid()
                ax5.set_xscale('log')
                ax5.set_yscale('log')
                set_legend_title(ax5, self.CATS[ic].name0, 14, 'Impact')

                # Moving avarage in log
                log10_n_fs = np.log10(clust.nf_all[clust.nf_all > 0])
                log10_n_as = np.log10(clust.na_all[clust.na_all > 0])
                log10_t_fs = np.log10(clust.time_fs[clust.nf_all > 0])
                log10_t_as = np.log10(clust.time_as[clust.na_all > 0])
                vbin = np.arange(np.floor(min([min(log10_t_fs), min(log10_t_as)])), 1+np.ceil(max([max(log10_t_fs), max(log10_t_as)])), 1.0)
                nbin_as = np.zeros(len(vbin)-1)
                nbin_fs = np.zeros(len(vbin)-1)
                vbinc = np.zeros(len(vbin)-1)
                no_sies_val = 0.1
                for ii in range(len(vbin)-1):
                    try:
                        nbin_as[ii] = sum(10**log10_n_as[(log10_t_as >= vbin[ii]) & (log10_t_as < vbin[ii+1])]) / sum((clust.time_as >= 10**vbin[ii]) & (clust.time_as < 10**vbin[ii+1]))
                    except:
                        nbin_as[ii] = no_sies_val
                    try:
                        nbin_fs[ii] = sum(10**log10_n_fs[(log10_t_fs >= vbin[ii]) & (log10_t_fs < vbin[ii+1])]) / sum((clust.time_fs >= 10**vbin[ii]) & (clust.time_fs < 10**vbin[ii+1]))
                    except:
                        nbin_fs[ii] = no_sies_val
                    vbinc[ii] = (vbin[ii] + vbin[ii+1]) / 2
                vbinc = 10**vbinc
                ax6.scatter(clust.time_as, clust.na_all, 10, 'r', label='Aftershock', alpha=0.5, zorder=10)
                ax6.scatter(clust.time_fs, clust.nf_all, 30, 'b', label='Foreshock', alpha=0.5, zorder=5)
                ax6.plot(vbinc, nbin_as, '-r', zorder=12)
                ax6.scatter(vbinc, nbin_as, s=40, c='r', marker='s')
                ax6.plot(vbinc, nbin_fs, '-b', zorder=12)
                ax6.scatter(vbinc, nbin_fs, s=40, c='b', marker='s')
                ax6.set_xlabel('Duration (days)')
                ax6.set_ylabel('#events')
                ax6.set_xscale('log')
                ax6.set_yscale('log')
                ax6.set_ylim([no_sies_val/10, 2*max(clust.na_all)])
                ax6.grid()
                set_legend_title(ax6, self.CATS[ic].name0, 14, 'Impact')
                
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3_xlims = ax3.get_xlim()
            ax3_ylims = ax3.get_ylim()
            ax3.plot([min([min(ax3_xlims), min(ax3_ylims)]), max([max(ax3_xlims), max(ax3_ylims)])], [min([min(ax3_xlims), min(ax3_ylims)]), max([max(ax3_xlims), max(ax3_ylims)])], '-k')

            ax3.set_ylabel('Foreshock rates #events/day')
            ax3.set_xlabel('Aftershock rates #events/day')
            ax3.grid()
            ax3.set_aspect(1)

            ax3.legend()

            fig.subplots_adjust(top=0.98, bottom=0.05, hspace=0.25)
            fig2.subplots_adjust(top=0.98, bottom=0.05, hspace=0.25)
            fig4.subplots_adjust(top=0.98, bottom=0.05, hspace=0.25)
            fig5.subplots_adjust(top=0.98, bottom=0.05, hspace=0.25)
            fig6.subplots_adjust(top=0.98, bottom=0.05, hspace=0.25)


        elif Ptype == 'Productivity_Rmax_Rpredicted':
            fig = plb.figure(2057)
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                RmaxFS = clust['RmaxFS'].values
                RmaxAS  = clust['RmaxAS'].values
                Rpred   = clust['Rpred'].values
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                # if self.DT.days == 60:
                    tp   = clust['c_type60'].values
                else:
                    tp   = clust['c_type'].values
                Idm = tp == 1 ; types_c = 'Mainshocks'

                ax.scatter(m[Idm], RmaxFS[Idm] / Rpred[Idm], c='b', alpha=0.5, label='foreshocks')
                ax.scatter(m[Idm], RmaxAS[Idm] / Rpred[Idm], c='r', alpha=0.5, label='aftershocks')
                ax.set_xlabel('Mainshock* magnitud Mw')
                ax.set_ylabel('R_cluster / R_WNC')
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                if self.cmethod == 'WnC':
                    ax.set_ylim([0,1])
                else:
                    ax.set_yscale('log')
                    ax.set_ylim([0.1, max(RmaxFS[Idm] / Rpred[Idm])])
            
        elif Ptype == 'PlotProductivity_timing':
            fig = plb.figure(2058)
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                # if self.DT.days == 60:
                    na = clust['na_apr'].values
                    nf = clust['nf_apr'].values
                    tp   = clust['c_type60'].values
                    days_cluster = self.DT.days
                else:
                    tp   = clust['c_type'].values
                    na = clust['na_all_apr'].values
                    nf = clust['nf_all_apr'].values
                    days_cluster = np.inf
                dtmaxFS = clust['dtmaxFS'].values / (24*3600)
                MmaxFS  = clust['MmaxFS'].values
                
                Idm = tp == 1 ; types_c = 'Mainshocks'
                IyesF = np.logical_and(nf > 0, Idm)
                minmaxM = np.array([Mc+self.dM, max(m)])
                # fit k for const a - all
                k1,a1,rms1,res1,r2 = fitK(m[IyesF], na[IyesF], Mc, 1)
                lb1 = 'k all:%2.3f rms:%2.2f' % (k1, rms1)

                # residual productivity
                nRes = na[IyesF] / (k1*10**(a1*(m[IyesF] - Mc)))

                vN,vX = moveAVG(dtmaxFS[IyesF],nRes,10,[0,61],1.0)
                Iok = vN >0
                ax.scatter(dtmaxFS[IyesF], nRes)

                ax.plot(vX[Iok],vN[Iok],':k')
                ax.plot(vX,vN,'-k')
                ax.scatter(vX,vN,s=30,c='k', marker='s')
                ax.set_xlabel('days before maishock*')
                ax.set_ylabel('Relative productivity')
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                if self.cmethod == 'WnC':
                    ax.set_ylim([0, 1])
                    # ax.set_xlim([0, 60])
                # else:
                    # ax.set_xscale('log')
                    # ax.set_xlim([0.1, max(dtmaxFS[IyesF])])
                ax.set_xlim([0, self.ddays])
        
        elif Ptype == 'AftershockWithForeshocksProductivity_c_type_aperture':
            cmap = cm.get_cmap('cool', 5)
            fig = plb.figure(2027)
            grid = plb.GridSpec(2, 9)
            for ic in range(nCats):

                ax = fig.add_subplot(grid[3*ic: 3*ic+2])
                axb = fig.add_subplot(grid[3*ic+2])

                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                    na = clust['na_apr'].values
                    nf = clust['nf_apr'].values
                    tp = clust['c_type60'].values
                    days_cluster = self.DT.days
                else:
                    tp = clust['c_type'].values
                    na = clust['na_all_apr'].values
                    nf = clust['nf_all_apr'].values
                    days_cluster = np.inf

                Idm = tp == 1 ; types_c = 'Mainshocks'
                # Idm = np.logical_and(tp >= 1, tp <= 2); types_c = 'Mainshocks + Doublets'
                # Idm = np.logical_and(tp >= 1, tp <= 3); types_c = 'Mainshocks + Doublets + Triplets'

                IyesF = np.logical_and(nf > 0, Idm)
                InoF = nf == 0

                minmaxM = np.array([Mc+self.dM, max(m)])
                # fit k for const a - all
                k1,a1,rms1,res1,r2 = fitK(m, na, Mc, 0)
                lb1 = 'k all:%2.3f rms:%2.2f' % (k1, rms1)
                ax.plot(minmaxM, k1*10**(a1*(minmaxM - Mc)),c='k', alpha=0.5,label=lb1)

                # fit k for const a - yes foreshocks
                k0_yes,a1_yes,rms_k_yes,_,r2_yes = fitK(m[IyesF], na[IyesF], Mc, 0)

                # fit k for const a - no foreshocks
                k0_no,a1_no,rms_k_no,_,r2_no = fitK(m[InoF], na[InoF], Mc, 0)

                # bootstrap data
                k_yes, a_yes, _ = bootstrap(m[IyesF],na[IyesF],Mc,a1_yes,self.nbs)
                k_no,  a_no,  _ = bootstrap(m[InoF], na[InoF], Mc,a1_no,self.nbs)

                # set histogram range
                nbins = 30
                minmaxK = minmax(np.concatenate([k_yes,k_no]))
                hist_edges = np.arange(minmaxK[0],minmaxK[1],abs(np.diff(minmaxK)/nbins))

                # Make histograms
                axb.hist(k_yes, hist_edges, orientation="horizontal",color=cmap(1), alpha=0.5,edgeColor = 'black')
                axb.hist(k_no,  hist_edges, orientation="horizontal",color=cmap(3), alpha=0.5,edgeColor = 'black')

                k_yes95 = np.quantile(k_yes, 0.95)
                k_yes05 = np.quantile(k_yes, 0.05)
                k_no95 = np.quantile(k_no, 0.95)
                k_no05 = np.quantile(k_no, 0.05)

                xlims = axb.get_xlim()
                axb.plot(xlims, [k0_yes,k0_yes],c=cmap(1), linewidth=2)
                axb.plot(xlims, [k0_no,k0_no],c=cmap(3), linewidth=2)
                axb.plot(xlims, [k_yes95,k_yes95],c=cmap(1),linestyle=':')
                axb.plot(xlims, [k_no95,k_no95],c=cmap(3),linestyle=':')
                axb.plot(xlims, [k_yes05,k_yes05],c=cmap(1),linestyle=':')
                axb.plot(xlims, [k_no05,k_no05],c=cmap(3),linestyle=':')
                plb.yticks(rotation=45)


                lb_yes = 'k With FS: %2.3f rms:%2.2f' % (k0_yes, rms_k_yes)
                ax.plot(minmaxM, k0_yes*10**(a1_yes*(minmaxM - Mc)),c=cmap(1), alpha=0.5,label=lb_yes)
                ax.scatter(m[IyesF], na[IyesF], c=cmap(1), alpha=0.5, edgecolors='k')

                lb_no = 'k No FS: %2.3f rms:%2.2f' % (k0_no, rms_k_no)
                ax.plot(minmaxM, k0_no*10**(a1_no*(minmaxM - Mc)),c=cmap(3), alpha=1.0,label=lb_no)
                ax.scatter(m[InoF], na[InoF], facecolors='none', edgecolors=cmap(3), alpha=1.0)

                ax.set_xlabel('Mainshock magnitud Mw')
                if ic ==0:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                elif ic == 3:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                ax.set_yscale('log')
                ax.set_ylim([1, 100])
                ax.set_xlim(minmaxM)
                ax.legend(title=self.CATS[ic].name0,loc='upper left')
            fig.suptitle('%s: Sequences - %s' % (self.cmethod ,types_c))
            fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99, bottom=0.05, top=0.95)

        elif Ptype == 'ForeshockProductivity_c_type':
            fig = plb.figure(2090)
            for ic in range(nCats):
                ax = fig.add_subplot(4, 2, ic+1)
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':  # or if self.DT.days == 60:
                    # nf   = clust['nf'].values
                    nf   = clust['nf_apr'].values
                    tp   = clust['c_type60'].values
                else:
                    # nf   = clust['nf_all'].values
                    nf   = clust['nf_all_apr'].values
                    tp   = clust['c_type'].values

                Idm = tp == 1 ; types_c = 'Mainshocks'
                IyesF1 = np.logical_and(nf > 0, Idm)
                IyesF2 = np.logical_and(nf > 0, ~Idm)
                InoFS = np.logical_and(nf == 0, Idm)

                # nMrange1, Mcenter1 = binMag(m[IyesF1], Mc+self.dM, nf[IyesF1])
                nMrange1, Mcenter1 = binMag(m, Mc+self.dM, nf)
                # nMrange2, Mcenter2 = binMag(m[IyesF2], Mc+self.dM, nf[IyesF2])
                
                minmaxM = np.array([Mc+self.dM, max(m)+0.5])
                medianM = np.median(m[Idm])
                p_fs_low_m = sum(np.logical_and(InoFS, m < medianM)) / sum(Idm) * 100
                p_fs_large_m = sum(np.logical_and(InoFS, m > medianM)) / sum(Idm) * 100
                # fit k for const a - all
                # k1,a1,rms1,res1,r21 = fitK(m[IyesF1], nf[IyesF1], Mc, -999)
                k1, a1, rms1, res1, r21 = fitK(Mcenter1, nMrange1, Mc, -999)
                # k2,a2,rms2,res2,r22 = fitK(m[IyesF2], nf[IyesF2], Mc, -999)
                lb1 = r'$a:%2.1f\: r^{2}:%2.1f$' % (a1, r21)
                # lb2 = r'$a:%2.2f\: r^{2}:%2.2f$' % (a2, r22)
                ax.plot(np.arange(minmaxM[0], minmaxM[1], 0.1), k1*10**(a1*(np.arange(minmaxM[0], minmaxM[1], 0.1) - Mc)), '--b', label=lb1)
                # ax.plot(np.arange(minmaxM[0],minmaxM[1],0.1), k2*10**(a2*(np.arange(minmaxM[0],minmaxM[1],0.1) - Mc)),'--k',label=lb2)
                
                ax.scatter(m[IyesF1], nf[IyesF1], 30, alpha=0.7, facecolor='b', edgecolors='k')#, label=r'$Mainshock^{*}$')
                ax.scatter(m[InoFS], np.ones(sum(InoFS)) * 0.2, 30, alpha=0.7, facecolor='y', edgecolors='k')# label='No FS'
                # ax.scatter(m[IyesF2], nf[IyesF2],30,alpha=0.7, facecolor='gray',edgecolors='k')#,label=r'$Other^{*}$')
                ax.plot(Mcenter1, nMrange1, 's-b')
                # ax.plot(Mcenter2, nMrange2, 's-',color='gray')

                ax.set_xlabel('Mainshock magnitud Mw')
                ax.set_ylabel('Number of foreshocks')
                ax.set_yscale('log')
                ax.grid()
                # ax.set_ylim([1, 1000])
                ax.set_xlim(minmaxM)
                ylims = ax.get_ylim()
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
            #fig.tight_layout(pad=3.0)
            fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99, bottom=0.05, top=0.95)
        
        elif Ptype == 'AftershockProductivity_c_type':
            fig = plb.figure(2060)
            for ic in range(nCats):

                ax = fig.add_subplot(2,3,ic+1)

                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                    na   = clust['na'].values
                    nf   = clust['nf'].values
                    tp   = clust['c_type60'].values
                    days_cluster = self.DT.days
                else:
                    na   = clust['na_all'].values
                    nf   = clust['nf_all'].values
                    tp   = clust['c_type'].values
                    days_cluster = np.inf

                
                c_type_names = ['not-defined','Mainshocks','Doublets','Triplets','Swarms']
                c_type_symbol = 'o*v^s'
                minmaxM = np.array([Mc+self.dM, max(m)])
                # fit k for const a - all
                k1,a1,rms1,res1,res2 = fitK(m, na, Mc, 1)
                lb1 = 'k:%2.3f rms:%2.2f' % (k1, rms1)
                ax.plot(minmaxM, k1*10**(a1*minmaxM - Mc),c='k', alpha=0.5)
                for ii in range(len(c_type_names)):
                    Ic = tp == ii
                    ax.scatter(m[Ic], na[Ic], 30, alpha=0.9, edgecolors='k',marker=c_type_symbol[ii], label=c_type_names[ii])

                
                ax.set_xlabel('Mainshock magnitud Mw')
                if ic ==0:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                elif ic == 3:
                    ax.set_ylabel('Number of aftershocks %2.0f days' % days_cluster)
                ax.set_yscale('log')
                ax.set_ylim([1, 1000])
                ax.set_xlim(minmaxM)
                ax.legend(title=self.CATS[ic].name0,loc='upper left')
            #fig.tight_layout(pad=3.0)
            fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99)

        elif Ptype == 'K_mainshocks':
            f_per = np.zeros(nCats)
            fig = plb.figure(6321)
            ax1 = fig.add_subplot(1, 2, 1)
            for ic in range(nCats):
                clust = self.CATS[ic].CLUST
                nf = clust['nf_all'].values
                Idm = clust['c_type'].values == 1
                nf = nf[Idm]
                f_per[ic] = sum(nf >= 1) / len(nf) * 100
                ax1.scatter(self.CATS[ic].k_data_fs.k, f_per[ic], c='k')
                ax1.text(self.CATS[ic].k_data_fs.k, f_per[ic], self.CATS[ic].name0)
            ax1.grid()
            ax1.set_xlabel('Aftershocks productivity k')
            ax1.set_ylabel('Mainshocks* with foreshocks [%]')
                      
        elif Ptype == 'foreshock_map_fms':
            fig = plb.figure(2337)
            nFS = 5
            for ic in range(nCats):

                if self.CATS[ic].name0 == 'GCMT':
                    clust = self.CATS[ic].CLUST
                    nf = clust['nf_all'].values
                    Idm = clust['c_type'].values == 1
                    IyesF = np.logical_and(nf >= nFS, Idm)
                    clust = clust[IyesF]

                    for ii in range(len(clust)):
                        ax = fig.add_subplot(5, 6, ii+1, projection=ccrs.Mercator(central_longitude=180))
                        deCLUST = self.CATS[ic].deCLUST
                        m0 = clust['m0'].values[ii]
                        t0 = UTCDateTime(clust['ot0'].values[ii])
                        Lats0 = clust['lat0'].values[ii]
                        Lons0 = clust['lon0'].values[ii]
                        deCLUST = deCLUST[deCLUST['clID'].values == clust.cid.values[ii]]
                        T = np.array([UTCDateTime(jj).timestamp for jj in deCLUST['Time'].values])
                        deCLUST = deCLUST[T <= t0]
                        cat = self.CATS[ic].cat
                        fms = np.zeros((len(deCLUST), 3))
                        for jj in range(len(fms)):
                            It = np.abs(cat.ot - T[jj]) < 10 # sec
                            Im = np.abs(cat.M - deCLUST['MAG'].values[jj]) < 0.2
                            I = It & Im
                            if sum(I > 0):
                                fms[jj, 0] = cat.strike[I][0]
                                fms[jj, 1] = cat.dip[I][0]
                                fms[jj, 2] = cat.rake[I][0]
                        posM0 = np.argmin(np.abs(T - t0.timestamp))
                        [r_predicted, _] = WnCfaultL(m0, ifault=0)
                        fact_r = AdjustEffectiveRadius(m0)
                        r_predicted = fact_r * r_predicted
                        Ro = kilometers2degrees(r_predicted / 1000)
                        [xr, yr, xyr] = MakeCircleUTM(r_predicted, Lons0, Lats0)
                        xr = Make360(xr)
                        ax.plot(xr, yr, c='m', transform=ccrs.PlateCarree())#  label='E.R.: %d km' % int(r_predicted / 1000)

                        difx = Ro*2.1; Lats = [Lats0-difx, Lats0+difx]; Lons = [Lons0-difx, Lons0+difx]


                        ax.set_extent([min(Lons), max(Lons), min(Lats), max(Lats)])
                        ax.add_feature(cartopy.feature.LAND)
                        ax.add_feature(cartopy.feature.OCEAN)
                        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)

                        projection = ccrs.Mercator(central_longitude=180.0)
                        dllf = np.max([np.diff([min(Lons), max(Lons)]), np.diff([min(Lats), max(Lats)])]) * np.max([np.abs(Lats0) / 20, 1])
                        cfms = 'krgb'
                        for jj in range(len(fms)):
                            if sum(fms[jj] > 0):
                                fms2 = aux_plane(fms[jj, 0], fms[jj, 1], fms[jj, 2])
                                fstyle1, fstyle = ftypeShearer(fms[jj, 2], fms2[2])
                                x0, y0 = projection.transform_point(x=deCLUST.Lon.values[jj], y=deCLUST.Lat.values[jj], src_crs=ccrs.Geodetic())
                                if jj == posM0:
                                    b = beach(fms[jj], xy=(x0, y0), width=dllf/15*100000, linewidth=1, facecolor=cfms[fstyle], edgecolor='m', zorder=10)
                                else:
                                    b = beach(fms[jj], xy=(x0, y0), width=dllf/15*100000, linewidth=0.1, facecolor=cfms[fstyle],  zorder=10)
                                ax.add_collection(b)


                        fnames = ['ridge', 'trench', 'transform']
                        c = 'gbr'
                        for pp in range(len(fnames)):
                            shapfile1 = self.work_folder + 'shapefiles/Global_TL_Faults/%s.shp' % fnames[pp]
                            pltshape(ax, shapfile1, c[pp], ccrs=ccrs.PlateCarree(), Lons=Lons, Lats=Lats)
                        ax.set_title('%s Mw %2.1f' % (t0.date, m0))
                        ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
                    fig.subplots_adjust(wspace=0.5, left=0.05, right=0.99)


        elif Ptype == 'productive_foreshocks':
            fig = plb.figure(2237)
            nFS = 5
            for ic in range(nCats):

                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                na = clust['na_all_apr'].values
                nf = clust['nf_all'].values

                m_lon = Make360(clust.lon0.values)
                m_lat = clust.lat0.values

                Idm = clust['c_type'].values == 1
                IyesF = np.logical_and(nf >= nFS, Idm)
                
                ax = fig.add_subplot(nCats, 1, ic+1, projection=ccrs.PlateCarree(central_longitude=180))
                ax.set_extent([180, -180, min(m_lat)-5, max(m_lat)+5], crs=ccrs.PlateCarree())
                ax.stock_img()

                # Plot All
                ax.scatter(m_lon, m_lat, 20, zorder=14, c='grey', transform=ccrs.PlateCarree())
                ax.scatter(m_lon[Idm & IyesF], m_lat[Idm & IyesF], 25, zorder=15, facecolors='none', edgecolors='b', label='FS ≥ %d' % nFS, transform=ccrs.PlateCarree())

                ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
            fig.suptitle(self.cmethod)


        elif Ptype == 'plot_fs_moment_ratios':
            x, y = shape2xy2(self.work_folder + 'shapefiles/EW_Pacific.shp')
            x[0] = Make360(x[0])
            x[1] = Make360(x[1])
            fig = plb.figure(1301)
            for ic in range(nCats):
                for ic in range(nCats):
                    if nCats == 2:
                        ax = fig.add_subplot(2, 1, ic+1, projection=ccrs.PlateCarree(central_longitude=180))
                        ax.set_extent([max(self.CATS[ic].lonlims), min(self.CATS[ic].lonlims), min(self.CATS[ic].latlims), max(self.CATS[ic].latlims)], crs=ccrs.PlateCarree())
                    elif nCats == 1:
                        if self.CATS[ic].b_data.Mc >= 5.0:
                            ax = fig.add_subplot(1, 1, ic+1, projection=ccrs.PlateCarree(central_longitude=180))
                            ax.set_extent([max(self.CATS[ic].lonlims), min(self.CATS[ic].lonlims), min(self.CATS[ic].latlims), max(self.CATS[ic].latlims)], crs=ccrs.PlateCarree())
                        else:
                            ax = fig.add_subplot(1, 1, ic+1, projection=ccrs.PlateCarree())
                            ax.set_extent([max(self.CATS[ic].lonlims), min(self.CATS[ic].lonlims), min(self.CATS[ic].latlims), max(self.CATS[ic].latlims)], crs=ccrs.PlateCarree())

                    else:
                        ax = fig.add_subplot(4, 2, ic+1)
                    Idm = self.CATS[ic].CLUST['c_type'].values == 1
                    ax.stock_img()
                    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
                    ax.set_ylim(self.CATS[ic].latlims)
                    ax.set_xlim(self.CATS[ic].lonlims)
                    fnames = ['ridge', 'trench', 'transform']
                    c = 'gbr'
                    for pp in range(len(fnames)):
                        shapfile1 = self.work_folder + 'shapefiles/Global_TL_Faults/%s.shp' % fnames[pp]
                    I1 = (self.CATS[ic].CLUST['m01_2'].values > 0) & Idm # decreasing.
                    I2 = (self.CATS[ic].CLUST['m01_2'].values < 0) & Idm # increasing.
                    I0 = (self.CATS[ic].CLUST['m01_2'].values == 0) & Idm # no data.
                    ax.scatter(self.CATS[ic].CLUST['lon0'][Idm], self.CATS[ic].CLUST['lat0'][Idm], s=20, edgecolor='k', facecolors='none', label='no data', transform=ccrs.PlateCarree())
                    ax.scatter(self.CATS[ic].CLUST['lon0'][I1], self.CATS[ic].CLUST['lat0'][I1], s=20, c='g', label='decreasing', transform=ccrs.PlateCarree())
                    ax.scatter(self.CATS[ic].CLUST['lon0'][I2], self.CATS[ic].CLUST['lat0'][I2], s=20, c='m', label='increasing', transform=ccrs.PlateCarree())
                    ax.plot(x[0], y[0], transform=ccrs.PlateCarree())
                    ax.plot(x[1], y[1], transform=ccrs.PlateCarree())
                    m_lon = Make360(self.CATS[ic].CLUST.lon0.values)
                    m_lat = self.CATS[ic].CLUST.lat0.values
                    Iw = inpolygon(m_lon, m_lat, x[0], y[0]) & Idm
                    Ie = inpolygon(m_lon, m_lat, x[1], y[1]) & Idm
                    # bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),)

                    ax.text(90, 50, 'EP increasing %d %%\nEP decreasing %d %%\nEP no data %d %%'
                            % ((sum(Ie & I2) / sum(Ie)*100), (sum(Ie & I1) / sum(Ie)*100), (sum(Ie & I0) / sum(Ie)*100)),
                            bbox=dict(boxstyle="round", ec=(0.5, 0.5, 0.5), fc=(0.8, 0.8, 0.8), alpha=0.5))
                    ax.text(-110, 50, 'WP increasing %d %%\nWP decreasing %d %%\nWP no data %d %%'
                            % ((sum(Iw & I2) / sum(Iw)*100), (sum(Iw & I1) / sum(Iw)*100), (sum(Iw & I0) / sum(Iw)*100)),
                            bbox=dict(boxstyle="round", ec=(0.5, 0.5, 0.5), fc=(0.8, 0.8, 0.8), alpha=0.5))

                    # set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                    set_legend_title(ax, self.cmethod, 14, 'Impact')
                fig.tight_layout()


        elif Ptype == 'EW_Pacific':
            x, y = shape2xy2(self.work_folder + 'shapefiles/EW_Pacific.shp')
            fig = plb.figure(2227)
            nFS = 1
            for ic in range(nCats):

                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                na = clust['na_all_apr'].values
                nf = clust['nf_all_apr'].values

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

                print('West Pacific Forshocks %d %% %d / %d ' % (sum(IwF) / sum(Iw)*100, sum(IwF), sum(Iw)))
                print('East Pacific Forshocks %d %% %d / %d' % (sum(IeF) / sum(Ie)*100, sum(IeF), sum(Ie)))

                ax = fig.add_subplot(nCats, 1, ic+1, projection=ccrs.PlateCarree(central_longitude=180))
                ax.set_extent([max(m_lon[Ie])+5, min(m_lon[Iw])-5, min(m_lat)-5, max(m_lat)+5], crs=ccrs.PlateCarree())
                ax.stock_img()

                # Plot All
                ax.scatter(m_lon[Idm], m_lat[Idm], 20, zorder=14, c='grey', transform=ccrs.PlateCarree(), label='No FS')
                ax.scatter(m_lon[Idm & IyesF], m_lat[Idm & IyesF], 25, zorder=15, facecolors='none', edgecolors='b', label='FS ≥ %d' % nFS, transform=ccrs.PlateCarree())
                ax.plot(x[0], y[0], transform=ccrs.PlateCarree())
                ax.plot(x[1], y[1], transform=ccrs.PlateCarree())
                ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
                ax.text(-70, 50, 'WP %d %% %d / %d ' % (sum(IwF) / sum(Iw)*100, sum(IwF), sum(Iw)))
                ax.text(90, 50, 'EP %d %% %d / %d ' % (sum(IeF) / sum(Ie)*100, sum(IeF), sum(Ie)))

                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
            fig.suptitle(self.cmethod)

            # fig2 = plb.figure(2228)
            # nFS = 1
            # for ic in range(nCats):
            #
            #     clust = self.CATS[ic].CLUST
            #     Mc = self.CATS[ic].b_data.Mc
            #     m = clust['m0'].values
            #     na = clust['na_all_apr'].values
            #     nf = clust['nf_all_apr'].values
            #
            #     x[0] = Make360(x[0])
            #     x[1] = Make360(x[1])
            #
            #     m_lon = Make360(clust.lon0.values)
            #     m_lat = clust.lat0.values
            #
            #     Iw = inpolygon(m_lon, m_lat, x[0], y[0])
            #     Ie = inpolygon(m_lon, m_lat, x[1], y[1])
            #
            #     Idm = clust['c_type'].values == 1
            #     Irev = clust['fstyle'].values == 3
            #     Idm = Idm & Irev
            #     IyesF = np.logical_and(nf >= nFS, Idm)
            #     InoF = nf == 0
            #
            #     Iw = np.logical_and(Iw, Idm)
            #     Ie = np.logical_and(Ie, Idm)
            #
            #     IwF = np.logical_and(Iw, IyesF)
            #     IeF = np.logical_and(Ie, IyesF)
            #
            #     print('West Pacific Forshocks %d perc.' % (sum(IwF) / sum(Iw)*100))
            #     print('East Pacific Forshocks %d perc.' % (sum(IeF) / sum(Ie)*100))
            #
            #     ax = fig2.add_subplot(nCats, 1, ic+1, projection=ccrs.PlateCarree(central_longitude=180))
            #     ax.set_extent([max(m_lon[Ie])+5, min(m_lon[Iw])-5, min(m_lat)-5, max(m_lat)+5], crs=ccrs.PlateCarree())
            #     ax.stock_img()
            #
            #     ax.scatter(m_lon[Idm], m_lat[Idm], 20, zorder=14, c='grey', transform=ccrs.PlateCarree(), label='No FS')
            #     ax.scatter(m_lon[Idm & IyesF], m_lat[Idm & IyesF], 25, zorder=15, facecolors='none', edgecolors='b', label='FS ≥ %d' % nFS, transform=ccrs.PlateCarree())
            #
            #     ax.plot(x[0], y[0], transform=ccrs.PlateCarree())
            #     ax.plot(x[1], y[1], transform=ccrs.PlateCarree())
            #     ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
            #     ax.text(90, 50, 'EP %d %%' % (sum(IeF) / sum(Ie)*100))
            #     ax.text(-70, 50, 'WP %d %%' % (sum(IwF) / sum(Iw)*100))
            #     set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
            # fig2.suptitle('Thrust %s ' % self.cmethod)

        elif Ptype == 'c_typePie':
            fig = plb.figure(1750)
            # c_type_names = ['not-defined','Mainshocks','Doublets','Triplets','Swarms-I','Swarms-II']
            c_type_names = ['Mainshocks', 'Doublets', 'Triplets', 'Swarms-I', 'Swarms-II']
            for ic in range(nCats):
                if nCats == 2:
                    ax = fig.add_subplot(2, 1, ic+1)
                else:
                    ax = fig.add_subplot(4, 2, ic+1)
                clust = self.CATS[ic].CLUST
                if self.cmethod == 'WnC':
                    tp = clust['c_type60'].values
                else:
                    tp = clust['c_type'].values

                LowSeis     = sum(tp ==-1)
                not_defined = sum(tp == 0)
                Mainshocks  = sum(tp == 1)
                Doublets    = sum(tp == 2)
                Triplets    = sum(tp == 3)
                Swarms1      = sum(tp == 4)
                Swarms2      = sum(tp == 5)
                
                # Vtp = [not_defined,Mainshocks,Doublets,Triplets,Swarms1,Swarms2]
                Vtp = [Mainshocks, Doublets, Triplets, Swarms1, Swarms2]
                if ic == 0:
                    ax.pie(Vtp, labels=c_type_names, autopct='%1.0f%%', shadow=True, startangle=90)
                else:
                    ax.pie(Vtp, autopct='%1.0f%%', shadow=True, startangle=90)
                    
                ax.set_aspect('equal')
                ax.text(-0.5, -1.2, 'N: %d, N≥4: %d' % (len(tp), len(tp) - LowSeis))
                ax.set_title(self.CATS[ic].name0)
                
            fig.subplots_adjust(hspace=0.41,wspace=0.16,left=0.05, right=0.97)
        
        elif Ptype == 'c_type':
            fig = plb.figure(1700)
            c_type_names = ['not-defined','Mainshocks','Doublets','Triplets','Swarms-I','Swarms-II']
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                if self.cmethod == 'WnC':
                    tp = clust['c_type60'].values
                else:
                    tp = clust['c_type'].values

                not_defined = sum(tp == 0)
                Mainshocks  = sum(tp == 1)
                Doublets    = sum(tp == 2)
                Triplets    = sum(tp == 3)
                Swarms1      = sum(tp == 4)
                Swarms2      = sum(tp == 5)
                
                
                ax.hist(tp,[-0.5,0.5,1.5,2.5,3.5,4.5,5.5],edgeColor = 'black',label='not-defined %d' % not_defined)
                for ii in range(1,len(c_type_names,Mainshocks)):
                    ax.text(ii, 1+sum(tp == ii), '%d' % sum(tp == ii))

                ax.set_xticks(np.arange(len(c_type_names)))
                ax.set_xticklabels(c_type_names)
                plb.xticks(rotation=-45)
                ax.legend(title=self.CATS[ic].name0,loc='upper right')
                ax.set_xlim([0.5,4.5])
                ax.set_ylim([0, 1.2*max([Mainshocks, Doublets, Triplets, Swarms1,Swarms2])])
            fig.subplots_adjust(hspace=0.41,wspace=0.16,left=0.05, right=0.97)
        
        elif Ptype == 'c_type_map':
            fig = plb.figure(1701)
            c_type_names = ['not-defined','Mainshocks','Doublets','Triplets','Swarms']
            for ic in range(nCats):
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                mag = clust['m0'].values
                m_lat = clust['lat0'].values
                m_lon = clust['lon0'].values
                na = clust['na'].values
                nf = clust['nf'].values
                dm_a = clust['dm_a'].values
                dm_f = clust['dm_f'].values
                
                if self.cmethod == 'WnC':
                    tp = clust['c_type60'].values
                else:
                    tp = clust['c_type'].values

                ax = fig.add_subplot(2, 3, ic+1)
                if sum([self.CATS[ic].name == 'GCMT', self.CATS[ic].name == 'NEIC']) > 0:
                    m_lon = Make360(m_lon)
                    m = Basemap(llcrnrlat=-70, urcrnrlat=max(self.CATS[ic].latlims), llcrnrlon=70, urcrnrlon=340, lat_ts=1)
                else:
                    m = Basemap(projection='merc', llcrnrlat=min(self.CATS[ic].latlims), urcrnrlat=max(self.CATS[ic].latlims), llcrnrlon=min(self.CATS[ic].lonlims), urcrnrlon=max(self.CATS[ic].lonlims), lat_ts=1, resolution='i')
                m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
                m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
                m.drawcoastlines(linewidth=0.1, color="white")
                x, y = m(m_lon, m_lat)
                ctc = 'wrcgk'
                for ii in range(len(ctc)):
                    Ictc = tp == ii
                    ax.scatter(x[Ictc],y[Ictc],10,c=ctc[ii],zorder=10+ii,label=c_type_names[ii]) # , edgecolors='k'
                if ic == 3:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
                
                ax.set_title(self.CATS[ic].name0)

            fig.subplots_adjust(wspace=0.1,left=0.05, right=0.99)

        elif Ptype == 'Cat_poissonian':
            if self.cmethod == 'ZnBZ':
                fig = plb.figure(1450)
                for ic in range(nCats):
                    ax = fig.add_subplot(3,3,ic+1)
                    clust = self.CATS[ic].CLUST

        elif Ptype == 'ShowIndividualClusters':
            for ic in range(nCats):
                pos = np.argsort(len(self.CATS[ic].CLUST.m0))
                for ii in range(len(pos)):
                    fig = plb.figure(999)
                    ax1 = fig.add_subplot(1,2,1)
                    ax2 = fig.add_subplot(1,2,2)
                    I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                    clust = self.CATS[ic].deCLUST[I]
                    Lats = [min(clust['Lat']),max(clust['Lat'])]
                    Lons = [min(clust['Lon']),max(clust['Lon'])]
                    Lats0 = np.mean(Lats)
                    Lons0 = np.mean(Lons)
                    difLat = np.abs(np.diff(Lats))[0]
                    difLon = np.abs(np.diff(Lons))[0]
                    difx = max([difLat, difLon])*1.1

                    Lats = [Lats0-difx, Lats0+difx]
                    Lons = [Lons0-difx, Lons0+difx]
                    t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])
                    # plotTimesMclust(ax2, clust, t0,self.cmethod)

                    # plotMapC(ax1, clust, Lats, Lons, self.CATS[ic].CLUST.lon0[pos[ii]], self.CATS[ic].CLUST.lat0[pos[ii]], t0, self.CATS[ic].CLUST.m0[pos[ii]], self.CATS[ic].cat)


                    m = Basemap(projection='merc', llcrnrlat=min(clust['Lat']), urcrnrlat=max(clust['Lat']), llcrnrlon=min(clust['Lon']), urcrnrlon=max(clust['Lon']), lat_ts=1, resolution='i')
                    m.drawcoastlines(color='k', linewidth=0.2)  # add coastlines
                    x, y = m(clust['Lon'].values, clust['Lat'].values)
                    otc=[]
                    for jj in range(len(clust['Lat'].values)):
                        otc.append(UTCDateTime(clust['Time'].values[jj]))
                    otc = np.array(otc)
                    selFS = otc < t0
                    selAS = otc > t0
                    selMS = otc == t0

                    ax1.scatter(x[selFS], y[selFS], 5, marker='o', color='b',alpha=0.4)
                    ax1.scatter(x[selAS], y[selAS], 5, marker='o', color='r',alpha=0.4)
                    ax1.scatter(x[selMS], y[selMS], 50, marker='*', color='k',alpha=0.4)

                    ax2.scatter(atmp_Time[selFS], clust['Mag'].values[selFS], 5, marker='o', color='b',alpha=0.4)
                    ax2.scatter(atmp_Time[selAS], atmp_MAG[selAS], 5, marker='o', color='r',alpha=0.4)
                    ax2.scatter(atmp_Time[selMS], atmp_MAG[selMS], 50, marker='*', color='k',alpha=0.4)
                    plb.show()
                
        elif Ptype == 'Globalregions':
            fig = plb.figure(1009)
            # np.dtype(float)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim([-10, 70])
            ax.set_xlim([-180, 180])
            for ic in range(nCats):
                cat0 = self.CATS[ic].cat
                lat_lims = [cat0.Lat.min(), cat0.Lat.max()]
                lon_lims = [cat0.Long.min(), cat0.Long.max()]
                ax.add_patch(Rectangle((min(lon_lims), min(lat_lims)), (max(lon_lims) - min(lon_lims)), (max(lat_lims) - min(lat_lims)), fc='none', ec='k', lw=1))
            ax.osm = OSM(ax)
            # set_legend_title(ax, 'All', 14, 'Impact')


        elif Ptype == 'MapSeis3':
            fig = plb.figure(500)
            for ic in range(nCats):
                if nCats == 2:
                    ax = fig.add_subplot(2, 1, ic+1, projection=ccrs.Mercator(central_longitude=180))
                elif nCats == 1:
                    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(central_longitude=180))
                else:
                    ax = fig.add_subplot(4, 2, ic+1, projection=ccrs.Mercator(central_longitude=180))
                cat0 = self.CATS[ic].cat
                Mc = self.CATS[ic].b_data.Mc
                if Mc >= 5.0:
                    Msel = 7.0
                else:
                    Msel = np.max(cat0.M)-1.0
                sel7 = cat0.M >=Msel

                ax.set_extent([cat0.Long.min(), cat0.Long.max(), -75, 75], crs=ccrs.PlateCarree())
                fnames = ['ridge', 'trench', 'transform']
                c = 'gbr'
                for pp in range(len(fnames)):
                    shapfile1 = self.work_folder + 'shapefiles/Global_TL_Faults/%s.shp' % fnames[pp]
                    pltshape(ax, shapfile1, c[pp], ccrs=ccrs.PlateCarree(), Lons=[cat0.Long.min(), cat0.Long.max()], Lats=[-75, 75])

                ax.plot(cat0.Long, cat0.Lat, 'ko', ms=1, transform=ccrs.PlateCarree(), label='M≥%2.1f (%d)' % (Mc, len(sel7)))
                ax.plot(cat0.Long[sel7], cat0.Lat[sel7], 'mo', ms=3, mew=1.5, mfc='none', label='M≥%2.1f (%d)' % (Msel, sum(sel7)), transform=ccrs.PlateCarree())
                ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3, transform=ccrs.PlateCarree())
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                ax.stock_img()
            fig.tight_layout()

        elif Ptype == 'MapSeis2':
            fig = plb.figure(500)
            for ic in range(nCats):
                if nCats == 2:
                    ax = fig.add_subplot(2, 1, ic+1)
                elif nCats == 1:
                    ax = fig.add_subplot(1, 1, ic+1)
                else:
                    ax = fig.add_subplot(4, 2, ic+1)
                cat0 = self.CATS[ic].cat
                Mc = self.CATS[ic].b_data.Mc
                Msel = Mc + self.dM
                # if Mc >= 5.0:
                #     Msel = 7.0
                # else:
                #     Msel = np.max(cat0.M)-1.0
                sel7 = cat0.M >= Msel
                ax.set_ylim([cat0.Lat.min(), cat0.Lat.max()])
                ax.set_xlim([cat0.Long.min(), cat0.Long.max()])
                fnames = ['ridge', 'trench', 'transform']
                c = 'gbr'
                for pp in range(len(fnames)):
                    shapfile1 = self.work_folder + 'shapefiles/Global_TL_Faults/%s.shp' % fnames[pp]
                    pltshape(ax, shapfile1, c[pp], ccrs=False)
                ax.plot(cat0.Long, cat0.Lat, 'ko', ms=1, alpha=0.6, label='M≥%2.1f (%d)' % (Mc, len(sel7))) # , label='M≥%2.1f' % Mc)
                ax.plot(cat0.Long[sel7], cat0.Lat[sel7], 'mo', ms=3, mew=1.5, mfc='none', label='M≥%2.1f (%d)' % (Msel, sum(sel7)))
                ax.osm = OSM(ax)
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')

            fig.tight_layout()

        elif Ptype == 'MapSeis':
            fig = plb.figure(500)
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                cat0 = self.CATS[ic].cat
                projection = 'cyl'
                xmin,xmax = cat0.Long.min(), cat0.Long.max()
                ymin,ymax = cat0.Lat.min(), cat0.Lat.max()
                # setup equi distance basemap.
                m = Basemap( llcrnrlat  =  ymin,urcrnrlat  =  ymax,
                             llcrnrlon  =  xmin,urcrnrlon  =  xmax,
                             projection = projection,lat_0=(ymin+ymax)*.5,lon_0=(xmin+xmax)*.5,
                             resolution = 'i')
                m.drawstates( linewidth = 1)
                m.drawcoastlines( linewidth= 2)
                m.drawmeridians( np.linspace( int(xmin), xmax, 4),labels=[False,False,False,True],
                                 fontsize = 12, fmt = '%.1f')
                m.drawparallels( np.linspace( int(ymin), ymax, 4),labels=[True,False,False,False],
                                 fontsize = 12, fmt = '%.2f')
                a_x, a_y = m( cat0.Long, cat0.Lat)
                ax.plot( a_x, a_y, 'ko', ms = 1, alpha=0.5)
                Msel = np.max(cat0.M)-1.0
                sel7 = cat0.M >=Msel
                ax.plot( a_x[sel7], a_y[sel7], 'ro', ms = 8, mew= 1.5, mfc = 'none', label = 'M>%2.1f' % Msel)
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
       
        
        elif Ptype == 'M0M1-productivity':
            fig = plb.figure(1100)
            markers = 'vs*^.DXP'
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                na = clust['na'].values
                # b_a = clust['b_a'].values
                # b_f = clust['b_f'].values
                m = clust['m0'].values
                Mc = self.CATS[ic].b_data.Mc
                dm_a = clust['dm_a'].values
                dm_f = clust['dm_f'].values
                k1,a1,rms1,res = fitK(m, na, Mc, 1)
                I=na>0
                ax.scatter(res,dm_a[I],marker=markers[ic],c='r',alpha=0.5,label='Aftershocks')
                ax.scatter(res,dm_f[I],marker=markers[ic],c='b',alpha=0.5,label='Foreshocks')
                ax.set_xlabel('Residual Productivity')
                ax.set_ylabel('M0 - M1')
                ax.set_ylim([0, 2])
                ax.legend(title=self.CATS[ic].name0,loc='upper left')
                

        elif Ptype == 'Foreshocks_b_value_FMS':
            fig1 = plb.figure(50051)
            fig2 = plb.figure(50061)
            fig3 = plb.figure(50071)
            labels = 'Oblique', 'Strike-slip', 'Normal', 'Reverse'
            for ic in range(nCats):
                if self.CATS[ic].name == 'GCMT' or self.CATS[ic].name == 'GCMT2':
                    clust = self.CATS[ic].CLUST
                    tp = clust['c_type'].values
                    Idm = tp == 1  # types_c = 'Mainshocks'
                    clust = clust[Idm]
                    for jj in range(4): # odd, ss, nor, rev
                        Ifms = clust['fstyle_m'].values == jj
                        clustf = clust[Ifms]
                        ForeshocksM = []
                        AftershocksM = []
                        pos = clustf.index
                        for ii in range(len(pos)):
        
                            I = self.CATS[ic].deCLUST['clID'] == clustf['cid'][pos[ii]]
                            clustM = self.CATS[ic].deCLUST[I]

                            t0 = UTCDateTime(clustf.ot0[pos[ii]])

                            otc=[]
                            for kk in range(len(clustM['Lat'].values)):
                                otc.append(UTCDateTime(clustM['Time'].values[kk]))
                            otc = np.array(otc)
                            selFS = otc < t0
                            selAS = otc > t0
                            selMS = otc == t0
                            for kk in range(len(clustM[selFS].MAG.values)):
                                ForeshocksM.append(clustM[selFS].MAG.values[kk])
                            for kk in range(len(clustM[selAS].MAG.values)):
                                AftershocksM.append(clustM[selAS].MAG.values[kk])

                        ForeshocksM = np.array(ForeshocksM)
                        AftershocksM = np.array(AftershocksM)
                        b_dataF = calc_b_val(ForeshocksM, 0.1, self.CATS[ic].M_bval_min, 6.2)
                        b_dataA = calc_b_val(AftershocksM, 0.1, self.CATS[ic].M_bval_min, 6.2)
                        ax1 = fig1.add_subplot(2, 2, jj+1)
                        ax2 = fig2.add_subplot(2, 2, jj+1)
                        ax3 = fig3.add_subplot(2, 2, jj+1)
                        if len(ForeshocksM) > 0:
                            print_b_val(b_dataF, min(ForeshocksM),max(ForeshocksM), ax1, ax2,labels[jj],'#1E90FF','b')
                        if len(AftershocksM) > 0:
                            print_b_val(b_dataA, min(AftershocksM),max(AftershocksM), ax1, ax3,labels[jj],'#F08080','r')

                fig1.suptitle('GCMT G-R %s' % self.cmethod)
                fig2.suptitle('GCMT G-R %s' % self.cmethod)

        
        elif Ptype == 'Aftershocks_b_value_yes_no_fs':
            fig1 = plb.figure(5095)
            fig2 = plb.figure(5096)
            fig3 = plb.figure(5097)
            fig4 = plb.figure(5098)
            for ic in range(nCats):
                Aftershocks_nfs = []
                Aftershocks_yfs = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]
    
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])
    
                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        selAS = otc > t0
                        selMS = otc == t0
                        
                        if sum(selFS) >= 1:
                            for jj in range(len(clust[selAS].MAG.values)):
                                Aftershocks_yfs.append(clust[selAS].MAG.values[jj])
                        else:
                            for jj in range(len(clust[selAS].MAG.values)):
                                Aftershocks_nfs.append(clust[selAS].MAG.values[jj])

                Aftershocks_yfs = np.array(Aftershocks_yfs)
                Aftershocks_nfs = np.array(Aftershocks_nfs)

                ax1 = fig1.add_subplot(4, 2, ic+1)
                ax2 = fig2.add_subplot(4, 2, ic+1)
                ax3 = fig3.add_subplot(4, 2, ic+1)
                ax4 = fig4.add_subplot(4, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5

                try:
                    b_dataF = calc_b_val(Aftershocks_yfs, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(Aftershocks_yfs), max(Aftershocks_yfs), ax1, ax2, self.CATS[ic].name0, 'gray', 'k')
                    ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], c='k')
                    ax4.scatter(b_dataF.Mc, b_dataF.b_val, c='k')
                except:
                    print('No data!')

                try:
                    b_dataA = calc_b_val(Aftershocks_nfs, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(Aftershocks_nfs), max(Aftershocks_nfs), ax1, ax3, self.CATS[ic].name0, '#F08080', 'r')
                    ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], c='r')#, label='Aftershocks')
                    ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')

                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])

                ax4.set_xlim([Mc, Mc + dm1])
                ax4.set_ylabel('b-value')
                ax4.set_xlabel('Magnitude')
                ax4.set_ylim([0.4, 1.6])
                ax4.grid()
                set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')
                
            fig1.suptitle(' G-R %s' % self.cmethod)
            fig2.suptitle(' G-R %s' % self.cmethod)
        

        elif Ptype == 'FMS_spindles':
            max_theta = 30.0
            for ic in range(nCats):
                if self.CATS[ic].name == 'NEIC' or self.CATS[ic].name == 'GCMT2':
                    theta_f = []
                    theta_a = []
                    r_f = []
                    r_a = []
                    type_a = []
                    type_f = []
                    depth_m0_f = []
                    depth_m0_a = []
                    in_CP_f = []
                    in_CP_a = []
                    domain_a = []
                    domain_f = []
                    print('Calculate FMS for spindles')
                    for ii in range(len(self.CATS[ic].CLUST.m0)):
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][ii]
                        clust = self.CATS[ic].deCLUST[I]
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[ii])
                        otc = []
                        for jj in range(len(clust['Time'].values)):
                            # otc.append((t0 - UTCDateTime(clust['Time'].values[jj])) / (24.0 * 3600.0))
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        If = otc < t0
                        Ia = otc > t0

                        pos0 = np.argmin(np.abs(self.CATS[ic].cat.ot - t0))
                        FMS0 = [self.CATS[ic].cat.strike[pos0], self.CATS[ic].cat.dip[pos0], self.CATS[ic].cat.rake[pos0]]
                        fstyle, dP, dT, dB = mechaStyle([FMS0[0]], [FMS0[1]], [FMS0[2]])

                        # m_lon = Make360(self.CATS[ic].CLUST.lon0[ii])
                        # m_lat = self.CATS[ic].CLUST.lat0[ii]
                        # Iw = inpolygon(m_lon, m_lat, x[0], y[0])
                        # Ie = inpolygon(m_lon, m_lat, x[1], y[1])
                        # in_CP = 0
                        # if Iw == True:
                        #     in_CP = 1
                        # if Ie == True:
                        #     in_CP = 1
                        in_CP = self.CATS[ic].CLUST.InterIntra[ii]
                        domain = get_ll_domain(t0, self.work_folder)

                        if (sum(If) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1) & (FMS0[0] > -999):
                            for jj in range(sum(If)):
                                t1 = otc[If][jj]
                                pos1 = np.argmin(np.abs(self.CATS[ic].cat.ot - t1))
                                FMS1 = [self.CATS[ic].cat.strike[pos1], self.CATS[ic].cat.dip[pos1], self.CATS[ic].cat.rake[pos1]]
                                theta = compare_fms(FMS0, FMS1, max_theta)
                                Rm0 = DistLatLonUTM(self.CATS[ic].cat.Lat[pos1], self.CATS[ic].cat.Long[pos1], self.CATS[ic].CLUST['lat0'][ii], self.CATS[ic].CLUST['lon0'][ii])
                                Rm0 = -(Rm0 / 1000) / self.CATS[ic].CLUST['Rpred'][ii]
                                if FMS1[0] > -999:
                                    theta_f.append(theta)
                                    r_f.append(Rm0)
                                    type_f.append(fstyle[0])
                                    depth_m0_f.append(self.CATS[ic].CLUST['depth0'][ii])
                                    in_CP_f.append(in_CP)
                                    domain_f.append(domain)

                        if (sum(Ia) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1) & (FMS0[0] > -999):
                            for jj in range(sum(Ia)):
                                t1 = otc[Ia][jj]
                                pos1 = np.argmin(np.abs(self.CATS[ic].cat.ot - t1))
                                FMS1 = [self.CATS[ic].cat.strike[pos1], self.CATS[ic].cat.dip[pos1], self.CATS[ic].cat.rake[pos1]]
                                theta = compare_fms(FMS0, FMS1, max_theta)
                                Rm0 = DistLatLonUTM(self.CATS[ic].cat.Lat[pos1], self.CATS[ic].cat.Long[pos1], self.CATS[ic].CLUST['lat0'][ii], self.CATS[ic].CLUST['lon0'][ii])
                                Rm0 = (Rm0 / 1000) / self.CATS[ic].CLUST['Rpred'][ii]
                                if FMS1[0] > -999:
                                    theta_a.append(theta)
                                    r_a.append(Rm0)
                                    type_a.append(fstyle[0])
                                    depth_m0_a.append(self.CATS[ic].CLUST['depth0'][ii])
                                    in_CP_a.append(in_CP)
                                    domain_a.append(domain)

                    theta_f = np.array(theta_f)
                    theta_a = np.array(theta_a)
                    r_f = np.array(r_f)
                    r_a = np.array(r_a)
                    type_a = np.array(type_a)
                    type_f = np.array(type_f)
                    depth_m0_a = np.array(depth_m0_a)
                    depth_m0_f = np.array(depth_m0_f)
                    in_CP_a = np.array(in_CP_a)
                    in_CP_f = np.array(in_CP_f)
                    domain_f = np.array(domain_f)
                    domain_a = np.array(domain_a)

                    # I_A_a = depth_m0_a < 15
                    # I_A_f = depth_m0_f < 15
                    # I_B_a = (depth_m0_a >= 15) & (depth_m0_a < 35)
                    # I_B_f = (depth_m0_f >= 15) & (depth_m0_f < 35)
                    # I_C_a = depth_m0_a >= 35
                    # I_C_f = depth_m0_f >= 35
                    
                    I_A_a = domain_a == 1
                    I_A_f = domain_f == 1
                    I_B_a = domain_a == 2
                    I_B_f = domain_f == 2
                    I_C_a = domain_a == 3
                    I_C_f = domain_f == 3

                    
                    dbin = 0.05

                    fig0 = plb.figure(91876)
                    ax1 = fig0.add_subplot(2, 3, 1)
                    ax2 = fig0.add_subplot(2, 3, 2)
                    ax3 = fig0.add_subplot(2, 3, 4)
                    ax4 = fig0.add_subplot(2, 3, 5)

                    ax5 = fig0.add_subplot(3, 3, 3)
                    ax6 = fig0.add_subplot(3, 3, 6)
                    ax7 = fig0.add_subplot(3, 3, 9)


                    def plot_spindle(ax1, theta_f, theta_a, r_f, r_a, dbin, max_theta, title):
                        if (len(theta_f) + len(theta_a)) > 0:
                            # make hist foreshocks
                            maxX = 1.2
                            binsF = np.arange(-maxX, dbin, dbin)
                            Iiter = theta_f < max_theta
                            countFinter = histw(r_f[Iiter], binsF)
                            countFintra = histw(r_f[~Iiter], binsF)

                            perFinter = 0
                            perFintra = 0
                            if len(Iiter) > 0:
                                perFinter = sum(Iiter) / len(Iiter) * 100
                                perFintra = sum(~Iiter) / len(Iiter) * 100

                            # make hist aftershocks
                            binsA = np.arange(-dbin, maxX, dbin)
                            Iiter = theta_a < max_theta
                            countAinter = histw(r_a[Iiter], binsA)
                            countAintra = histw(r_a[~Iiter], binsA)
                            perAinter = 0
                            perAintra = 0
                            if len(Iiter) > 0:
                                perAinter = sum(Iiter) / len(Iiter) * 100
                                perAintra = sum(~Iiter) / len(Iiter) * 100


                            countFinter = np.log10(countFinter)
                            countFintra = np.log10(countFintra)
                            countAinter = np.log10(countAinter)
                            countAintra = np.log10(countAintra)

                            ax1.bar(binsF[1:]-dbin/2, countFinter, color='b', alpha=0.9, width=dbin, edgecolor='k')
                            ax1.bar(binsF[1:]-dbin/2, -countFintra, color='b', alpha=0.4, width=dbin, edgecolor='k')
                            ax1.bar(binsA[:-1]+dbin/2, countAinter, color='r', alpha=0.9, width=dbin, edgecolor='k')
                            ax1.bar(binsA[:-1]+dbin/2, -countAintra, color='r', alpha=0.4, width=dbin, edgecolor='k')
                            ax1.grid()
                            # ax1.set_title(title)
                            maxY = max([max(countFinter), max(countFintra), max(countAinter), max(countAintra)]) * 1.1
                            ax1.set_ylim([-maxY, maxY])
                            ax1.set_xlim([-maxX, maxX])
                            ax1.set_xlabel(r'$\Delta r_{i}/R_{j}$')
                            # ax1.set_ylabel('Different from Mainshock  log(N)  Similar to Mainshock')

                            ax1.text(-maxX+dbin, maxY*0.85, r'$%s$' % title)
                            ax1.text(-maxX+dbin, maxY*0.7, '%d %%' % perFinter)
                            ax1.text(-maxX+dbin, -maxY*0.7, '%d %%' % perFintra)
                            ax1.text(maxX-dbin, maxY*0.7, '%d %%' % perAinter, horizontalalignment='right')
                            ax1.text(maxX-dbin, -maxY*0.7, '%d %%' % perAintra, horizontalalignment='right')
                        ax1.set_ylabel('Different    log(N)    Similar')
                        ax1.set_ylabel(r'Different     $log_{10}(N_{i})$     Similar')
                    
                    plot_spindle(ax1, theta_f, theta_a, r_f, r_a, dbin, max_theta, 'All')
                    plot_spindle(ax2, theta_f[type_f == 1], theta_a[type_a == 1], r_f[type_f == 1], r_a[type_a == 1], dbin, max_theta, 'Strike-Slip')
                    plot_spindle(ax3, theta_f[type_f == 2], theta_a[type_a == 2], r_f[type_f == 2], r_a[type_a == 2], dbin, max_theta, 'Normal')
                    plot_spindle(ax4, theta_f[type_f == 3], theta_a[type_a == 3], r_f[type_f == 3], r_a[type_a == 3], dbin, max_theta, 'Thrusts')

                    plot_spindle(ax5, theta_f[(type_f == 3) & I_A_f], theta_a[(type_a == 3) & I_A_a], r_f[(type_f == 3) & I_A_f], r_a[(type_a == 3) & I_A_a], dbin, max_theta, 'Thrusts (A)')
                    plot_spindle(ax6, theta_f[(type_f == 3) & I_B_f], theta_a[(type_a == 3) & I_B_a], r_f[(type_f == 3) & I_B_f], r_a[(type_a == 3) & I_B_a], dbin, max_theta, 'Thrusts (B)')
                    plot_spindle(ax7, theta_f[(type_f == 3) & I_C_f], theta_a[(type_a == 3) & I_C_a], r_f[(type_f == 3) & I_C_f], r_a[(type_a == 3) & I_C_a], dbin, max_theta, 'Thrusts (C)')

                    # plot_spindle(ax8, theta_f[((type_f == 3) & I_A_f) & (in_CP_f == 1)], theta_a[((type_a == 3) & I_A_a) & (in_CP_a == 1)], r_f[((type_f == 3) & I_A_f) & (in_CP_f == 1)], r_a[((type_a == 3) & I_A_a) & (in_CP_a == 1)], dbin, max_theta, 'Thrusts PB (A)')
                    # plot_spindle(ax9, theta_f[((type_f == 3) & I_B_f) & (in_CP_f == 1)], theta_a[((type_a == 3) & I_B_a) & (in_CP_a == 1)], r_f[((type_f == 3) & I_B_f) & (in_CP_f == 1)], r_a[((type_a == 3) & I_B_a) & (in_CP_a == 1)], dbin, max_theta, 'Thrusts PB (B)')
                    # plot_spindle(ax10, theta_f[((type_f == 3) & I_C_f) & (in_CP_f == 1)], theta_a[((type_a == 3) & I_C_a) & (in_CP_a == 1)], r_f[((type_f == 3) & I_C_f) & (in_CP_f == 1)], r_a[((type_a == 3) & I_C_a) & (in_CP_a == 1)], dbin, max_theta, 'Thrusts PB (C)')

                    
                    fig0.suptitle("%s %s" % (self.cmethod, self.CATS[ic].name0))

        
        elif Ptype == 'FMS_spindles2':
            maxX = 1.2
            max_theta = 30.0
            dbin = 0.05
            binsF = np.arange(-maxX, dbin, dbin)
            binsA = np.arange(-dbin, maxX, dbin)

            for ic in range(nCats):
                if self.CATS[ic].name == 'NEIC' or self.CATS[ic].name == 'GCMT2':

                    depth_m0_f = []
                    depth_m0_a = []
                    in_CP_f = []
                    in_CP_a = []
                    Reff_f = []
                    Reff_a = []
                    type_a = []
                    type_f = []

                    countFinter = [] # np.zeros((len(self.CATS[ic].CLUST.m0), len(binsF) - 1))
                    countFintra = [] # np.zeros((len(self.CATS[ic].CLUST.m0), len(binsF) - 1))
                    countAinter = [] # np.zeros((len(self.CATS[ic].CLUST.m0), len(binsA) - 1))
                    countAintra = [] # np.zeros((len(self.CATS[ic].CLUST.m0), len(binsA) - 1))
                    
                    domain_a = []
                    domain_f = []
                    print('Calculate FMS for spindles')
                    # nia = 0
                    # nif = 0
                    for ii in range(len(self.CATS[ic].CLUST.m0)):
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][ii]
                        clust = self.CATS[ic].deCLUST[I]
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[ii])
                        otc = []
                        for jj in range(len(clust['Time'].values)):
                            # otc.append((t0 - UTCDateTime(clust['Time'].values[jj])) / (24.0 * 3600.0))
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        If = otc < t0
                        Ia = otc > t0
                        # Ia = (otc > t0) & (otc < t0 + (30 * 24.0 * 3600.0))

                        pos0 = np.argmin(np.abs(self.CATS[ic].cat.ot - t0))
                        FMS0 = [self.CATS[ic].cat.strike[pos0], self.CATS[ic].cat.dip[pos0], self.CATS[ic].cat.rake[pos0]]
                        fstyle, dP, dT, dB = mechaStyle([FMS0[0]], [FMS0[1]], [FMS0[2]])
                        in_CP = self.CATS[ic].CLUST.InterIntra[ii]
                        domain = get_ll_domain(t0, self.work_folder)

                        theta_f = []
                        theta_a = []
                        r_f = []
                        r_a = []
                        

                        if (sum(If) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1) & (FMS0[0] > -999):
                            for jj in range(sum(If)):
                                t1 = otc[If][jj]
                                pos1 = np.argmin(np.abs(self.CATS[ic].cat.ot - t1))
                                FMS1 = [self.CATS[ic].cat.strike[pos1], self.CATS[ic].cat.dip[pos1], self.CATS[ic].cat.rake[pos1]]
                                if (FMS1[0] > -999):
                                    theta = compare_fms(FMS0, FMS1, max_theta)
                                    Rm0 = DistLatLonUTM(self.CATS[ic].cat.Lat[pos1], self.CATS[ic].cat.Long[pos1], self.CATS[ic].CLUST['lat0'][ii], self.CATS[ic].CLUST['lon0'][ii])
                                    Rm0 = -(Rm0 / 1000) / self.CATS[ic].CLUST['Rpred'][ii]
                                    theta_f.append(theta)
                                    r_f.append(Rm0)
                            theta_f = np.array(theta_f)
                            r_f = np.array(r_f)
                            type_f.append(fstyle[0])
                            depth_m0_f.append(self.CATS[ic].CLUST['depth0'][ii])
                            in_CP_f.append(in_CP)
                            Reff_f.append(self.CATS[ic].CLUST['Rpred'][ii])
                            Iiter = theta_f < max_theta
                            # countFinter.append(histw(r_f[Iiter], binsF))
                            # countFintra.append(histw(r_f[~Iiter], binsF))
                            countFinter.append(histw(r_f[Iiter], binsF) / self.CATS[ic].CLUST['Rpred'][ii])
                            countFintra.append(histw(r_f[~Iiter], binsF) / self.CATS[ic].CLUST['Rpred'][ii])
                            domain_f.append(domain)
                            # nif = nif + 1


                        if (sum(Ia) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1) & (FMS0[0] > -999):
                            for jj in range(sum(Ia)):
                                t1 = otc[Ia][jj]
                                pos1 = np.argmin(np.abs(self.CATS[ic].cat.ot - t1))
                                FMS1 = [self.CATS[ic].cat.strike[pos1], self.CATS[ic].cat.dip[pos1], self.CATS[ic].cat.rake[pos1]]
                                if (FMS1[0] > -999):
                                    theta = compare_fms(FMS0, FMS1, max_theta)
                                    Rm0 = DistLatLonUTM(self.CATS[ic].cat.Lat[pos1], self.CATS[ic].cat.Long[pos1], self.CATS[ic].CLUST['lat0'][ii], self.CATS[ic].CLUST['lon0'][ii])
                                    Rm0 = (Rm0 / 1000) / self.CATS[ic].CLUST['Rpred'][ii]
                                    theta_a.append(theta)
                                    r_a.append(Rm0)
                            theta_a = np.array(theta_a)
                            r_a = np.array(r_a)
                            type_a.append(fstyle[0])
                            depth_m0_a.append(self.CATS[ic].CLUST['depth0'][ii])
                            in_CP_a.append(in_CP)
                            Reff_a.append(self.CATS[ic].CLUST['Rpred'][ii])
                            Iiter = theta_a < max_theta
                            # countAinter.append(histw(r_a[Iiter], binsA))
                            # countAintra.append(histw(r_a[~Iiter], binsA))
                            countAinter.append(histw(r_a[Iiter], binsA) / self.CATS[ic].CLUST['Rpred'][ii])
                            countAintra.append(histw(r_a[~Iiter], binsA) / self.CATS[ic].CLUST['Rpred'][ii])
                            domain_a.append(domain)
                            # nia = nia + 1
                    # countFinter = countAinter[0:nif-1, :]
                    # countFintra = countAinter[0:nif-1, :]
                    # countAinter = countAinter[0:nia-1, :]
                    # countAintra = countAinter[0:nia-1, :]

                    # Mainshock
                    domain_a = np.array(domain_a)
                    domain_f = np.array(domain_f)
                    type_a = np.array(type_a)
                    type_f = np.array(type_f)
                    depth_m0_a = np.array(depth_m0_a)
                    depth_m0_f = np.array(depth_m0_f)
                    in_CP_a = np.array(in_CP_a)
                    in_CP_f = np.array(in_CP_f)
                    Reff_f = np.array(Reff_f)
                    Reff_a = np.array(Reff_a)

                    I_A_a = depth_m0_a < 15
                    I_A_f = depth_m0_f < 15
                    I_B_a = (depth_m0_a >= 15) & (depth_m0_a < 35)
                    I_B_f = (depth_m0_f >= 15) & (depth_m0_f < 35)
                    I_C_a = depth_m0_a >= 35
                    I_C_f = depth_m0_f >= 35

                    # I_A_a = domain_a == 1
                    # I_A_f = domain_f == 1
                    # I_B_a = domain_a == 2
                    # I_B_f = domain_f == 2
                    # I_C_a = domain_a == 3
                    # I_C_f = domain_f == 3

                    fig0 = plb.figure(92876)
                    ax1 = fig0.add_subplot(2, 3, 1)
                    ax2 = fig0.add_subplot(2, 3, 2)
                    ax3 = fig0.add_subplot(2, 3, 4)
                    ax4 = fig0.add_subplot(2, 3, 5)

                    ax5 = fig0.add_subplot(3, 3, 3)
                    ax6 = fig0.add_subplot(3, 3, 6)
                    ax7 = fig0.add_subplot(3, 3, 9)

                    fig1 = plb.figure(82876)
                    ax1a = fig1.add_subplot(4, 3, 1)
                    ax1b = fig1.add_subplot(4, 3, 4, sharex=ax1a)
                    ax2a = fig1.add_subplot(4, 3, 2)
                    ax2b = fig1.add_subplot(4, 3, 5, sharex=ax2a)
                    ax3a = fig1.add_subplot(4, 3, 7)
                    ax3b = fig1.add_subplot(4, 3, 10)
                    ax4a = fig1.add_subplot(4, 3, 8)
                    ax4b = fig1.add_subplot(4, 3, 11)

                    ax5a = fig1.add_subplot(6, 3, 3)
                    ax5b = fig1.add_subplot(6, 3, 6)
                    ax6a = fig1.add_subplot(6, 3, 9)
                    ax6b = fig1.add_subplot(6, 3, 12)
                    ax7a = fig1.add_subplot(6, 3, 15)
                    ax7b = fig1.add_subplot(6, 3, 18)


                    
                    def norm_list_by_Rf(list, Rf):
                        for ii in range(len(list)):
                            l = list[ii] / Rf[ii]
                            list[ii] = l
                        return list


                    def list2log(list):
                        for ii in range(len(list)):
                            l = np.log10(list[ii])
                            l[l == -inf] = 0
                            list[ii] = l
                        return list

                    def bool4list(list, bool):
                        list1 = []
                        for ii in range(len(bool)):
                            if bool[ii] == True:
                                list1.append(list[ii])
                        return list1


                    def plot_spindle2(ax1, countAinter, countAintra, countFinter, countFintra, binsF, binsA, dbin, title):
                        maxY = 0
                        sumFinter = 0
                        sumFintra = 0
                        sumAinter = 0
                        sumAintra = 0
                        for ii in range(len(countAinter)):
                            sumAinter += sum(countAinter[ii])
                        for ii in range(len(countAintra)):
                            sumAintra += sum(countAintra[ii])
                        for ii in range(len(countFinter)):
                            sumFinter += sum(countFinter[ii])
                        for ii in range(len(countFintra)):
                            sumFintra += sum(countFintra[ii])

                        if len(countFinter) > 0:
                            ax1.bar(binsF[1:]-dbin/2, countFinter[0], color='b', alpha=0.9, width=dbin, edgecolor='k', linewidth=0.5)
                            bottom = countFinter[0]
                            for ii in range(1, len(countFinter)):
                                ax1.bar(binsF[1:]-dbin/2, countFinter[ii], bottom=bottom, color='b', alpha=0.9, width=dbin, edgecolor='k', linewidth=0.5)
                                bottom = bottom + countFinter[ii]
                                maxY = max([maxY, max(np.abs(bottom))])
                        if len(countFintra) > 0:
                            ax1.bar(binsF[1:]-dbin/2, -countFintra[0], color='b', alpha=0.4, width=dbin, edgecolor='k', linewidth=0.5)
                            bottom = -countFintra[0]
                            for ii in range(1, len(countFintra)):
                                ax1.bar(binsF[1:]-dbin/2, -countFintra[ii], bottom=bottom, color='b', alpha=0.4, width=dbin, edgecolor='k', linewidth=0.5)
                                bottom = bottom - countFintra[ii]
                                maxY = max([maxY, max(np.abs(bottom))])
                        if len(countAinter) > 0:
                            ax1.bar(binsA[:-1]+dbin/2, countAinter[0], color='r', alpha=0.9, width=dbin, edgecolor='k', linewidth=0.5)
                            bottom = countAinter[0]
                            for ii in range(1, len(countAinter)):
                                ax1.bar(binsA[:-1]+dbin/2, countAinter[ii], bottom=bottom, color='r', alpha=0.9, width=dbin, edgecolor='k', linewidth=0.5)
                                bottom = bottom + countAinter[ii]
                                maxY = max([maxY, max(np.abs(bottom))])
                        if len(countAintra) > 0:
                            ax1.bar(binsA[:-1]+dbin/2, -countAintra[0], color='r', alpha=0.4, width=dbin, edgecolor='k', linewidth=0.5)
                            bottom = -countAintra[0]
                            for ii in range(1, len(countAintra)):
                                ax1.bar(binsA[:-1]+dbin/2, -countAintra[ii], bottom=bottom, color='r', alpha=0.4, width=dbin, edgecolor='k', linewidth=0.5)
                                bottom = bottom - countAintra[ii]
                                maxY = max([maxY, max(np.abs(bottom))])

                        ax1.grid()
                        # ax1.set_title(title)
                        # maxY = max([max(max(countFinter)), max(max(countFintra)), max(max(countAinter)), max(max(countAintra))]) * 1.1
                        # maxY = 2.5
                        ax1.set_ylim([-maxY, maxY])
                        ax1.set_xlim([-maxX, maxX])
                        ax1.set_xlabel(r'$\Delta r_{i}/R_{j}$')
                        # ax1.set_ylabel('Different from Mainshock  log(N)  Similar to Mainshock')
                        ax1.set_ylabel(r'Different     $N_{i,j}/R_{j}$     Similar')
                        ax1.text(-maxX+dbin, maxY*0.85, r'$%s$' % title)
                        sumF = (sumFintra + sumFinter)
                        if sumF == 0:
                            ax1.text(-maxX+dbin, maxY*0.7, '%d %%' % 0)
                            ax1.text(-maxX+dbin, -maxY*0.7, '%d %%' % 0)
                        else:
                            ax1.text(-maxX+dbin, maxY*0.7, '%d %%' % (sumFinter / sumF * 100))
                            ax1.text(-maxX+dbin, -maxY*0.7, '%d %%' % (sumFintra / sumF * 100))
                        sumA = (sumAintra + sumAinter)
                        if sumA > 0:
                            ax1.text(maxX-dbin, maxY*0.7, '%d %%' % (sumAinter / sumA * 100), horizontalalignment='right')
                            ax1.text(maxX-dbin, -maxY*0.7, '%d %%' % (sumAintra / sumA * 100), horizontalalignment='right')
                        else:
                            ax1.text(maxX-dbin, maxY*0.7, '%d %%' % 0, horizontalalignment='right')
                            ax1.text(maxX-dbin, -maxY*0.7, '%d %%' % 0, horizontalalignment='right')


                    def plot_spindle3(ax1, ax2, countAinter, countAintra, countFinter, countFintra, binsF, binsA, Reff_f, Reff_a, dbin, title):
                        maxY = 0
                        minY = 0
                        sumFinter = 0
                        sumFintra = 0
                        sumAinter = 0
                        sumAintra = 0

                        for ii in range(len(countAinter)):
                            sumAinter += sum(countAinter[ii])
                        for ii in range(len(countAintra)):
                            sumAintra += sum(countAintra[ii])
                        for ii in range(len(countFinter)):
                            sumFinter += sum(countFinter[ii])
                        for ii in range(len(countFintra)):
                            sumFintra += sum(countFintra[ii])

                        if len(countFinter) > 0:
                            countFinter_sum = np.zeros(len(countFinter[0]))
                            for ii in range(len(countFinter)):
                                countFinter_sum += countFinter[ii] / Reff_f[ii]
                            ax1.bar(binsF[1:]-dbin/2, countFinter_sum, color='b', alpha=0.9, width=dbin, edgecolor='k', linewidth=0.5)
                            maxY = max([maxY, max(np.abs(countFinter_sum))])
                            minY = min([minY, min(np.abs(countFinter_sum))])

                        if len(countAinter) > 0:
                            countAinter_sum = np.zeros(len(countAinter[0]))
                            for ii in range(len(countAinter)):
                                countAinter_sum += countAinter[ii] / Reff_a[ii]
                            ax1.bar(binsA[:-1]+dbin/2, countAinter_sum, color='r', alpha=0.9, width=dbin, edgecolor='k', linewidth=0.5)
                            maxY = max([maxY, max(np.abs(countAinter_sum))])
                            minY = min([minY, min(np.abs(countAinter_sum))])
                            
                        if len(countFintra) > 0:
                            countFintra_sum = np.zeros(len(countFintra[0]))
                            for ii in range(len(countFintra)):
                                countFintra_sum += countFintra[ii] / Reff_f[ii]
                            ax2.bar(binsF[1:]-dbin/2, countFintra_sum, color='b', alpha=0.4, width=dbin, edgecolor='k', linewidth=0.5)
                            maxY = max([maxY, max(np.abs(countFintra_sum))])
                            minY = min([minY, min(np.abs(countFintra_sum))])

                        if len(countAintra) > 0:
                            countAintra_sum = np.zeros(len(countAintra[0]))
                            for ii in range(len(countAintra)):
                                countAintra_sum += countAintra[ii] / Reff_a[ii]
                            ax2.bar(binsA[:-1]+dbin/2, countAintra_sum, color='r', alpha=0.4, width=dbin, edgecolor='k', linewidth=0.5)
                            maxY = max([maxY, max(np.abs(countAintra_sum))])
                            minY = min([minY, min(np.abs(countAintra_sum))])

                        ax1.scatter(0, maxY, alpha=0.0)
                        ax1.scatter(0, minY, alpha=0.0)
                        ax2.scatter(0, maxY, alpha=0.0)
                        ax2.scatter(0, minY, alpha=0.0)
                        ax1.set_yscale('log')
                        ax2.set_yscale('log')

                        # ax1.set_ylim([minY, maxY])
                        # ax2.set_ylim([minY, maxY])
                        ax1.set_ylim([5*10**-6, 5*10**-3])
                        ax2.set_ylim([5*10**-6, 5*10**-3])
                        ax2.invert_yaxis()
                        ax1.grid()
                        ax2.grid()
                        ax1.get_shared_x_axes().join(ax1, ax2)
                        ax2.set_xlabel(r'$\Delta r_{i}/R_{j}$')
                        ax1.set_xticklabels([])
                        # ax1.set_ylabel('Different from Mainshock  log(N)  Similar to Mainshock')
                        ax1.set_ylabel(r'Different     $N_{i,j}/R_{j}$     Similar')
                        ax1.text(-maxX+dbin, maxY*0.85, r'$%s$' % title)
                        ax1.set_xlim([-maxX, maxX])
                        ax2.set_xlim([-maxX, maxX])
                        # sumF = (sumFintra + sumFinter)
                        # if sumF == 0:
                        #     ax1.text(-maxX+dbin, maxY*0.7, '%d %%' % 0)
                        #     ax1.text(-maxX+dbin, -maxY*0.7, '%d %%' % 0)
                        # else:
                        #     ax1.text(-maxX+dbin, maxY*0.7, '%d %%' % (sumFinter / sumF * 100))
                        #     ax1.text(-maxX+dbin, -maxY*0.7, '%d %%' % (sumFintra / sumF * 100))
                        # sumA = (sumAintra + sumAinter)
                        # if sumA > 0:
                        #     ax1.text(maxX-dbin, maxY*0.7, '%d %%' % (sumAinter / sumA * 100), horizontalalignment='right')
                        #     ax1.text(maxX-dbin, -maxY*0.7, '%d %%' % (sumAintra / sumA * 100), horizontalalignment='right')
                        # else:
                        #     ax1.text(maxX-dbin, maxY*0.7, '%d %%' % 0, horizontalalignment='right')
                        #     ax1.text(maxX-dbin, -maxY*0.7, '%d %%' % 0, horizontalalignment='right')


                    plot_spindle3(ax1a, ax1b, countAinter, countAintra, countFinter, countFintra, binsF, binsA, Reff_f, Reff_a, dbin, 'All')
                    plot_spindle3(ax2a, ax2b, bool4list(countAinter, type_a == 1), bool4list(countAintra, type_a == 1), bool4list(countFinter, type_f == 1), bool4list(countFintra, type_f == 1), binsF, binsA, Reff_f[type_f == 1], Reff_a[type_a == 1], dbin, 'Strike-Slip')
                    plot_spindle3(ax3a, ax3b, bool4list(countAinter, type_a == 2), bool4list(countAintra, type_a == 2), bool4list(countFinter, type_f == 2), bool4list(countFintra, type_f == 2), binsF, binsA, Reff_f[type_f == 2], Reff_a[type_a == 2], dbin, 'Normal')
                    plot_spindle3(ax4a, ax4b, bool4list(countAinter, type_a == 3), bool4list(countAintra, type_a == 3), bool4list(countFinter, type_f == 3), bool4list(countFintra, type_f == 3), binsF, binsA, Reff_f[type_f == 3], Reff_a[type_a == 3], dbin, 'Thrusts')
                    
                    plot_spindle3(ax5a, ax5b, bool4list(countAinter, (type_a == 3) & I_A_a), bool4list(countAintra, (type_a == 3) & I_A_a), bool4list(countFinter, (type_f == 3) & I_A_f), bool4list(countFintra, (type_f == 3) & I_A_f), binsF, binsA, Reff_f[(type_f == 3) & I_A_f], Reff_a[(type_a == 3) & I_A_a], dbin, 'Thrusts (A)')
                    plot_spindle3(ax6a, ax6b, bool4list(countAinter, (type_a == 3) & I_B_a), bool4list(countAintra, (type_a == 3) & I_B_a), bool4list(countFinter, (type_f == 3) & I_B_f), bool4list(countFintra, (type_f == 3) & I_B_f), binsF, binsA, Reff_f[(type_f == 3) & I_B_f], Reff_a[(type_a == 3) & I_B_a], dbin, 'Thrusts (B)')
                    plot_spindle3(ax7a, ax7b, bool4list(countAinter, (type_a == 3) & I_C_a), bool4list(countAintra, (type_a == 3) & I_C_a), bool4list(countFinter, (type_f == 3) & I_C_f), bool4list(countFintra, (type_f == 3) & I_C_f), binsF, binsA, Reff_f[(type_f == 3) & I_C_f], Reff_a[(type_a == 3) & I_C_a], dbin, 'Thrusts (C)')

                    plot_spindle2(ax1, countAinter, countAintra, countFinter, countFintra, binsF, binsA, dbin, 'All')
                    plot_spindle2(ax2, bool4list(countAinter, type_a == 1), bool4list(countAintra, type_a == 1), bool4list(countFinter, type_f == 1), bool4list(countFintra, type_f == 1), binsF, binsA, dbin, 'Strike-Slip')
                    plot_spindle2(ax3, bool4list(countAinter, type_a == 2), bool4list(countAintra, type_a == 2), bool4list(countFinter, type_f == 2), bool4list(countFintra, type_f == 2), binsF, binsA, dbin, 'Normal')
                    plot_spindle2(ax4, bool4list(countAinter, type_a == 3), bool4list(countAintra, type_a == 3), bool4list(countFinter, type_f == 3), bool4list(countFintra, type_f == 3), binsF, binsA, dbin, 'Thrusts')
                    
                    plot_spindle2(ax5, bool4list(countAinter, (type_a == 3) & I_A_a), bool4list(countAintra, (type_a == 3) & I_A_a), bool4list(countFinter, (type_f == 3) & I_A_f), bool4list(countFintra, (type_f == 3) & I_A_f), binsF, binsA, dbin, 'Thrusts (A)')
                    plot_spindle2(ax6, bool4list(countAinter, (type_a == 3) & I_B_a), bool4list(countAintra, (type_a == 3) & I_B_a), bool4list(countFinter, (type_f == 3) & I_B_f), bool4list(countFintra, (type_f == 3) & I_B_f), binsF, binsA, dbin, 'Thrusts (B)')
                    plot_spindle2(ax7, bool4list(countAinter, (type_a == 3) & I_C_a), bool4list(countAintra, (type_a == 3) & I_C_a), bool4list(countFinter, (type_f == 3) & I_C_f), bool4list(countFintra, (type_f == 3) & I_C_f), binsF, binsA, dbin, 'Thrusts (C)')

                    fig0.suptitle("%s %s" % (self.cmethod, self.CATS[ic].name0))
                    fig0.subplots_adjust(hspace=0.28, wspace=0.25, left=0.09, right=0.97)
                    fig1.suptitle("%s %s" % (self.cmethod, self.CATS[ic].name0))

        elif Ptype == 'FMS_spindles3':
            for ic in range(nCats):
                if self.CATS[ic].name == 'NEIC' or self.CATS[ic].name == 'GCMT2':
                    theta_f = []
                    theta_a = []
                    r_f = []
                    r_a = []
                    type_a = []
                    type_f = []
                    depth_m0_f = []
                    depth_m0_a = []
                    in_CP_f = []
                    in_CP_a = []
                    Reff_f = []
                    Reff_a = []
                    print('Calculate FMS for spindles')
                    for ii in range(len(self.CATS[ic].CLUST.m0)):
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][ii]
                        clust = self.CATS[ic].deCLUST[I]
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[ii])
                        otc = []
                        for jj in range(len(clust['Time'].values)):
                            # otc.append((t0 - UTCDateTime(clust['Time'].values[jj])) / (24.0 * 3600.0))
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        If = otc < t0
                        Ia = otc > t0

                        pos0 = np.argmin(np.abs(self.CATS[ic].cat.ot - t0))
                        FMS0 = [self.CATS[ic].cat.strike[pos0], self.CATS[ic].cat.dip[pos0], self.CATS[ic].cat.rake[pos0]]
                        fstyle, dP, dT, dB = mechaStyle([FMS0[0]], [FMS0[1]], [FMS0[2]])

                        # m_lon = Make360(self.CATS[ic].CLUST.lon0[ii])
                        # m_lat = self.CATS[ic].CLUST.lat0[ii]
                        # Iw = inpolygon(m_lon, m_lat, x[0], y[0])
                        # Ie = inpolygon(m_lon, m_lat, x[1], y[1])
                        # in_CP = 0
                        # if Iw == True:
                        #     in_CP = 1
                        # if Ie == True:
                        #     in_CP = 1
                        in_CP = self.CATS[ic].CLUST.InterIntra[ii]

                        if (sum(If) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1):
                            for jj in range(sum(If)):
                                t1 = otc[If][jj]
                                pos1 = np.argmin(np.abs(self.CATS[ic].cat.ot - t1))
                                FMS1 = [self.CATS[ic].cat.strike[pos1], self.CATS[ic].cat.dip[pos1], self.CATS[ic].cat.rake[pos1]]
                                theta = compare_fms(FMS0, FMS1, max_theta)
                                Rm0 = DistLatLonUTM(self.CATS[ic].cat.Lat[pos1], self.CATS[ic].cat.Long[pos1], self.CATS[ic].CLUST['lat0'][ii], self.CATS[ic].CLUST['lon0'][ii])
                                Rm0 = -(Rm0 / 1000) / self.CATS[ic].CLUST['Rpred'][ii]
                                theta_f.append(theta)
                                r_f.append(Rm0)
                                type_f.append(fstyle[0])
                                depth_m0_f.append(self.CATS[ic].CLUST['depth0'][ii])
                                in_CP_f.append(in_CP)
                                Reff_f.append(self.CATS[ic].CLUST['Rpred'][ii])


                        if (sum(Ia) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1):
                            for jj in range(sum(Ia)):
                                t1 = otc[Ia][jj]
                                pos1 = np.argmin(np.abs(self.CATS[ic].cat.ot - t1))
                                FMS1 = [self.CATS[ic].cat.strike[pos1], self.CATS[ic].cat.dip[pos1], self.CATS[ic].cat.rake[pos1]]
                                theta = compare_fms(FMS0, FMS1, max_theta)
                                Rm0 = DistLatLonUTM(self.CATS[ic].cat.Lat[pos1], self.CATS[ic].cat.Long[pos1], self.CATS[ic].CLUST['lat0'][ii], self.CATS[ic].CLUST['lon0'][ii])
                                Rm0 = (Rm0 / 1000) / self.CATS[ic].CLUST['Rpred'][ii]
                                theta_a.append(theta)
                                r_a.append(Rm0)
                                type_a.append(fstyle[0])
                                depth_m0_a.append(self.CATS[ic].CLUST['depth0'][ii])
                                in_CP_a.append(in_CP)
                                Reff_a.append(self.CATS[ic].CLUST['Rpred'][ii])

                    theta_f = np.array(theta_f)
                    theta_a = np.array(theta_a)
                    r_f = np.array(r_f)
                    r_a = np.array(r_a)
                    type_a = np.array(type_a)
                    type_f = np.array(type_f)
                    depth_m0_a = np.array(depth_m0_a)
                    depth_m0_f = np.array(depth_m0_f)
                    in_CP_a = np.array(in_CP_a)
                    in_CP_f = np.array(in_CP_f)
                    Reff_f = np.array(Reff_f)
                    Reff_a = np.array(Reff_a)

                    I_A_a = depth_m0_a < 15
                    I_A_f = depth_m0_f < 15

                    I_B_a = (depth_m0_a >= 15) & (depth_m0_a < 35)
                    I_B_f = (depth_m0_f >= 15) & (depth_m0_f < 35)

                    I_C_a = depth_m0_a >= 35
                    I_C_f = depth_m0_f >= 35

                    max_theta = 25.0
                    dbin = 0.1

                    fig0 = plb.figure(9876)
                    fig1 = plb.figure(9877)
                    fig2 = plb.figure(9875)
                    ax1 = fig0.add_subplot(2, 2, 1)
                    ax2 = fig0.add_subplot(2, 2, 2)
                    ax3 = fig0.add_subplot(2, 2, 3)
                    ax4 = fig0.add_subplot(2, 2, 4)

                    ax5 = fig1.add_subplot(2, 3, 1)
                    ax6 = fig1.add_subplot(2, 3, 2)
                    ax7 = fig1.add_subplot(2, 3, 3)

                    ax8 = fig1.add_subplot(2, 3, 4)
                    ax9 = fig1.add_subplot(2, 3, 5)
                    ax10 = fig1.add_subplot(2, 3, 6)

                    
                    def plot_spindle(ax1, theta_f, theta_a, r_f, r_a, dbin, max_theta, title, Reff_f, Reff_a):
                        # make hist foreshocks
                        maxX = 1.2
                        binsF = np.arange(-maxX, dbin, dbin)
                        Iiter = theta_f < max_theta
                        countFinter = histw(r_f[Iiter], binsF)
                        countFintra = histw(r_f[~Iiter], binsF)
                        countFinter = countFinter / Reff_f[Iiter]
                        countFintra = countFintra / Reff_f[~Iiter]

                        # make hist aftershocks
                        binsA = np.arange(-dbin, maxX, dbin)
                        Iiter = theta_a < max_theta
                        countAinter = histw(r_a[Iiter], binsA)
                        countAintra = histw(r_a[~Iiter], binsA)
                        countAinter = countAinter / Reff_a[Iiter]
                        countAintra = countAintra / Reff_a[~Iiter]

                        countFinter = np.log10(countFinter)
                        countFintra = np.log10(countFintra)
                        countAinter = np.log10(countAinter)
                        countAintra = np.log10(countAintra)

                        ax1.bar(binsF[1:]-dbin/2, countFinter, color='b', alpha=0.9, width=dbin, edgecolor='k')
                        ax1.bar(binsF[1:]-dbin/2, -countFintra, color='b', alpha=0.4, width=dbin, edgecolor='k')
                        ax1.bar(binsA[:-1]+dbin/2, countAinter, color='r', alpha=0.9, width=dbin, edgecolor='k')
                        ax1.bar(binsA[:-1]+dbin/2, -countAintra, color='r', alpha=0.4, width=dbin, edgecolor='k')
                        ax1.grid()
                        # ax1.set_title(title)
                        maxY = max([max(countFinter), max(countFintra), max(countAinter), max(countAintra)]) * 1.1
                        ax1.set_ylim([-maxY, maxY])
                        ax1.set_xlim([-maxX, maxX])
                        ax1.set_xlabel(r'$dr/R_{WnC}$')
                        # ax1.set_ylabel('Different from Mainshock  log(N)  Similar to Mainshock')
                        ax1.set_ylabel('Different    log(N)/Reff    Similar')
                        ax1.text(-maxX+dbin, maxY*0.85, r'$%s$' % title)
                    
                    plot_spindle(ax1, theta_f, theta_a, r_f, r_a, dbin, max_theta, 'All', Reff_f, Reff_a)
                    plot_spindle(ax2, theta_f[type_f == 1], theta_a[type_a == 1], r_f[type_f == 1], r_a[type_a == 1], dbin, max_theta, 'Strike-Slip',Reff_f[type_f == 1], Reff_a[type_a == 1])
                    plot_spindle(ax3, theta_f[type_f == 2], theta_a[type_a == 2], r_f[type_f == 2], r_a[type_a == 2], dbin, max_theta, 'Normal',Reff_f[type_f == 2], Reff_a[type_a == 2])
                    plot_spindle(ax4, theta_f[type_f == 3], theta_a[type_a == 3], r_f[type_f == 3], r_a[type_a == 3], dbin, max_theta, 'Thrusts',Reff_f[type_f == 3], Reff_a[type_a == 3])

                    plot_spindle(ax5, theta_f[(type_f == 3) & I_A_f], theta_a[(type_a == 3) & I_A_a], r_f[(type_f == 3) & I_A_f], r_a[(type_a == 3) & I_A_a], dbin, max_theta, 'Thrusts (A)',Reff_f[(type_f == 3) & I_A_f], Reff_a[(type_a == 3) & I_A_a])
                    plot_spindle(ax6, theta_f[(type_f == 3) & I_B_f], theta_a[(type_a == 3) & I_B_a], r_f[(type_f == 3) & I_B_f], r_a[(type_a == 3) & I_B_a], dbin, max_theta, 'Thrusts (B)',Reff_f[(type_f == 3) & I_B_f], Reff_a[(type_a == 3) & I_B_a])
                    plot_spindle(ax7, theta_f[(type_f == 3) & I_C_f], theta_a[(type_a == 3) & I_C_a], r_f[(type_f == 3) & I_C_f], r_a[(type_a == 3) & I_C_a], dbin, max_theta, 'Thrusts (C)',Reff_f[(type_f == 3) & I_C_f], Reff_a[(type_a == 3) & I_C_a])

                    # plot_spindle(ax8, theta_f[((type_f == 3) & I_A_f) & (in_CP_f == 1)], theta_a[((type_a == 3) & I_A_a) & (in_CP_a == 1)], r_f[((type_f == 3) & I_A_f) & (in_CP_f == 1)], r_a[((type_a == 3) & I_A_a) & (in_CP_a == 1)], dbin, max_theta, 'Thrusts PB (A)',Reff_f, Reff_a)
                    # plot_spindle(ax9, theta_f[((type_f == 3) & I_B_f) & (in_CP_f == 1)], theta_a[((type_a == 3) & I_B_a) & (in_CP_a == 1)], r_f[((type_f == 3) & I_B_f) & (in_CP_f == 1)], r_a[((type_a == 3) & I_B_a) & (in_CP_a == 1)], dbin, max_theta, 'Thrusts PB (B)',Reff_f, Reff_a)
                    # plot_spindle(ax10, theta_f[((type_f == 3) & I_C_f) & (in_CP_f == 1)], theta_a[((type_a == 3) & I_C_a) & (in_CP_a == 1)], r_f[((type_f == 3) & I_C_f) & (in_CP_f == 1)], r_a[((type_a == 3) & I_C_a) & (in_CP_a == 1)], dbin, max_theta, 'Thrusts PB (C)',Reff_f, Reff_a)
                    fig0.suptitle("%s %s" % (self.cmethod, self.CATS[ic].name0))
                    fig1.suptitle("%s %s" % (self.cmethod, self.CATS[ic].name0))
        elif Ptype == 'Foreshocks_Omori_EW':
            x, y = shape2xy2(self.work_folder + 'shapefiles/EW_Pacific.shp')
            x[0] = Make360(x[0])
            x[1] = Make360(x[1])
            fig1 = plb.figure(1303)
            fig2 = plb.figure(1304)
            for ic in range(nCats):
                ForeshocksT_E = []
                ForeshocksT_W = []
                for ii in range(len(self.CATS[ic].CLUST.m0)):
                    I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][ii]
                    clust = self.CATS[ic].deCLUST[I]
                    t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[ii])
                    otc = []
                    for jj in range(len(clust['Time'].values)):
                        otc.append((t0 - UTCDateTime(clust['Time'].values[jj])) / (24.0 * 3600.0))
                    otc = np.array(otc)
                    otc = otc[otc > 0]
                    if (len(otc) > 0) & (self.CATS[ic].CLUST['c_type'][ii] == 1):
                        m_lon = Make360(self.CATS[ic].CLUST.lon0[ii])
                        m_lat = self.CATS[ic].CLUST.lat0[ii]
                        Iw = inpolygon(m_lon, m_lat, x[0], y[0])
                        Ie = inpolygon(m_lon, m_lat, x[1], y[1])
                        if Iw:
                            for jj in range(len(otc)):
                                ForeshocksT_W.append(otc[jj])
                        if Ie:
                            for jj in range(len(otc)):
                                ForeshocksT_E.append(otc[jj])
                ForeshocksT_E = np.array(ForeshocksT_E)
                ForeshocksT_W = np.array(ForeshocksT_W)
                dday = 1
                bins = np.arange(0.1, 100 + dday, dday)
                # bins[0] = bins[0] + 0.01
                countsE, binsE = np.histogram(ForeshocksT_E, bins)
                countsW, binsW = np.histogram(ForeshocksT_W, bins)
                binsW = (binsW[:-1] + binsW[1:]) / 2
                binsE = (binsE[:-1] + binsE[1:]) / 2
                I0 = countsW > 0
                countsW = countsW[I0]
                binsW = binsW[I0]
                I0 = countsE > 0
                countsE = countsE[I0]
                binsE = binsE[I0]

                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                # countsW = countsW / sum(countsW)
                # countsE = countsE / sum(countsE)
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                # ax1.plot(binsW, countsW, label='WP')
                # cp = ax1.get_lines()[0].get_color()
                ax1.scatter(binsW, countsW, label='WP') # c=cp
                # ax1.plot(binsE, countsE, label='EP')
                # cp = ax1.get_lines()[1].get_color()
                ax1.scatter(binsE, countsE, label='EP') # c=cp
                ax1.set_xlabel('Days before t0')
                ax1.set_ylabel('Foreshocks per %d days' % dday)
                ax1.set_xlim([min(bins), max(bins)])
                ax1.set_ylim([0.8, 100])
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.grid()
                set_legend_title(ax1, self.cmethod, 14, 'Impact', pos='upper right')
                #set_box_aspect(ax1, 1.0)
                #                       c         k        p
                a_limit = np.array([[.02, 2], [1, 300], [.2, 2.5]])
                a_par0 = np.array([.22, 50, .95])
                ForeshocksT_E.sort()
                ForeshocksT_W.sort()
                # at_bin_FS_E, aN_bin_FS_E = seis_utils.eqRate(ForeshocksT_E, 10)
                # at_bin_FS_W, aN_bin_FS_W = seis_utils.eqRate(ForeshocksT_W, 10)
                at_bin_FS_E = binsE; aN_bin_FS_E = countsE
                at_bin_FS_W = binsW; aN_bin_FS_W = countsW
                dOmW = omori.fit_omoriMLE(ForeshocksT_W, bounds=a_limit, par0=a_par0, disp = 0)
                dOmE = omori.fit_omoriMLE(ForeshocksT_E, bounds=a_limit, par0=a_par0, disp = 0)

                a_OMrateE = omori.fct_omori(at_bin_FS_E, [dOmE['c'], dOmE['K'], dOmE['p']])
                a_OMrateW = omori.fct_omori(at_bin_FS_W, [dOmW['c'], dOmW['K'], dOmW['p']])
                r2E = r2_score(aN_bin_FS_E, a_OMrateE)
                r2W = r2_score(aN_bin_FS_W, a_OMrateW)
                print( 'Omori-fit, MLE')
                
                #==================================4================================================
                #                                 plots
                #===================================================================================

                ax2.loglog(at_bin_FS_W, aN_bin_FS_W, 'o')
                ax2.loglog(at_bin_FS_E, aN_bin_FS_E, 'o')
                ax2.loglog(at_bin_FS_W, a_OMrateW, '--', c=[0.12156862745098039, 0.4666666666666667, 0.7058823529411765], label = 'WP: $c$= %.2f, $K$=%.2f,  $p$=%.2f, $r^2=%.2f$'% (dOmW['c'], dOmW['K'], dOmW['p'], r2W))
                ax2.loglog(at_bin_FS_E, a_OMrateE, '--', c=[1.0, 0.4980392156862745, 0.054901960784313725], label = 'EP: $c$= %.2f, $K$=%.2f,  $p$=%.2f, $r^2=%.2f$'%(dOmE['c'], dOmE['K'], dOmE['p'], r2E))
                ax2.set_xlabel( 'Time [day]')
                ax2.set_ylabel( 'events/day')
                set_legend_title(ax2, self.cmethod, 14, 'Impact', pos='upper right')
                ax2.set_ylim([0.8, 100])
                ax2.set_xlim(right=100)
                # ax2.set_xlim([min(bins), max(bins)])
                ax2.grid()


        elif Ptype == 'Foreshocks_Bath':
            fig1 = plb.figure(4022)
            
            for ic in range(nCats):
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                dMfv = []
                dMav = []
                dMfv_med = []
                dMav_med = []
                dMfvMS = []
                dMavMS = []
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]
    
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])
                        m0 = self.CATS[ic].CLUST.m0[pos[ii]]
    
                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        
                        # Aftershocks!
                        selAS = otc > t0
                        if sum(selAS) >= 1:
                            Ma = clust[selAS].MAG.values
                            dMa = m0 - np.max(Ma)
                            dMav.append(dMa)
                            dMavMS.append(m0)
                            dMav_med.append(m0 - np.median(Ma))
                            
                        # Foreshocks!
                        selFS = otc < t0
                        if sum(selFS) >= 2:
                            Mf = clust[selFS].MAG.values
                            tf = otc[selFS]
                            posF = np.argmax(Mf)
    
                            # get the foreshocks after largest foreshock
                            If = tf >= tf[posF]
                            Mf = Mf[If]
                            # tf = tf[If]
                            Mf = np.sort(Mf)[::-1]
                            if len(Mf) >= 2:
                                dMf = Mf[0] - Mf[1]
                                dMfv.append(dMf)
                                dMfvMS.append(Mf[0])
                                dMfv_med.append(Mf[0] - np.median(Mf[1:]))
                dMfv = np.array(dMfv)
                dMfv_med = np.array(dMfv_med)
                dMfvMS = np.array(dMfvMS)
                
                dMav = np.array(dMav)
                dMav_med = np.array(dMav_med)
                dMavMS = np.array(dMavMS)
                
                ax = fig1.add_subplot(4, 2, ic+1)
                ax.scatter(dMfvMS, dMfv, s=10, c='b')#, label='MF0 - MF1')
                ax.scatter(dMfvMS, dMfv_med, s=40, c='b', alpha=0.4)#, label='MF0 - med(MF1:)')
                ax.scatter(dMavMS, dMav, s=10, c='r')#, label='MF0 - MF1')
                ax.scatter(dMavMS, dMav_med, s=40, c='r', alpha=0.4)#, label='MF0 - med(MF1:)')
                xlims = ax.get_xlim()
                ax.plot(xlims, [np.median(dMfv_med), np.median(dMfv_med)], '-b', alpha=0.4, label='%2.2f' % np.median(dMfv_med))
                ax.plot(xlims, [np.median(dMav_med), np.median(dMav_med)], '-r', alpha=0.4, label='%2.2f' % np.median(dMav_med))
                ax.plot(xlims, [np.median(dMfv), np.median(dMfv)], '-b', label='%2.2f' % np.median(dMfv))
                ax.plot(xlims, [np.median(dMav), np.median(dMav)], '-r', label='%2.2f' % np.median(dMav))
                ax.set_ylim([0, 4])
                ax.set_xlabel('Mw')
                ax.set_ylabel('M0 - M1')
                ax.set_xlim(xlims)
                ax.grid()
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                        
        elif Ptype == 'Foreshocks_b_value':
            fig1 = plb.figure(5005)
            fig2 = plb.figure(5006)
            fig3 = plb.figure(5007)
            fig4 = plb.figure(5008)
            for ic in range(nCats):
                ForeshocksM = []
                AftershocksM = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:  # this line was not included in the "Regional" paper 
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]

                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])

                        otc=[]
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        selAS = otc > t0
                        # if sum(selFS) >= 2:
                        for jj in range(len(clust[selFS].MAG.values)):
                            ForeshocksM.append(clust[selFS].MAG.values[jj])
                        for jj in range(len(clust[selAS].MAG.values)):
                            AftershocksM.append(clust[selAS].MAG.values[jj])

                ForeshocksM = np.array(ForeshocksM)
                AftershocksM = np.array(AftershocksM)
                b_dataF = calc_b_val(ForeshocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                b_dataA = calc_b_val(AftershocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                ax3 = fig3.add_subplot(nrows, 2, ic+1)
                print_b_val(b_dataF, min(ForeshocksM), max(ForeshocksM), ax1, ax2, '', '#1E90FF', 'b') # self.CATS[ic].name0
                print_b_val(b_dataA, min(AftershocksM), max(AftershocksM), ax1, ax3, '', '#F08080', 'r') # self.CATS[ic].name0
                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])
                ax1 = fig4.add_subplot(nrows, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5
                ax1.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], c='r')#, label='Aftershocks')
                ax1.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], c='b')#, label='Foreshocks')
                ax1.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                ax1.scatter(b_dataF.Mc, b_dataF.b_val, c='b')
                ax1.set_xlim([Mc, Mc + dm1])
                ax1.set_ylabel('b-value')
                ax1.set_xlabel('Magnitude')
                ax1.set_ylim([0.4, 1.6])
                ax1.grid()
                # set_legend_title(ax1, self.CATS[ic].name0, 14, 'Impact')
                set_box_aspect(ax1, 0.5)

            # fig1.suptitle(' G-R %s' % self.cmethod)
            # fig2.suptitle(' G-R %s' % self.cmethod)


        elif Ptype == 'Foreshocks_b_value-1a':
            fig1 = plb.figure(5025)
            fig2 = plb.figure(5026)
            fig3 = plb.figure(5027)
            fig4 = plb.figure(5028)
            for ic in range(nCats):
                ForeshocksM = []
                AftershocksM = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])
                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        selAS = otc > t0
                        selMS = otc == t0
                        if sum(selAS) > 0:
                            if max(clust[selAS].MAG.values) >= self.CATS[ic].CLUST['m0'][pos[ii]]:
                                print('problem with mainshock selection!!')
                            
                        if sum(selFS) >= 2:
                            Mf = clust[selFS].MAG.values
                            tf = otc[selFS]
                            posF = np.argmax(Mf)
    
                            # get the foreshocks after largest foreshock
                            If = tf > tf[posF]
                            Mf = Mf[If]
                        
                            for jj in range(len(Mf)):
                                ForeshocksM.append(Mf[jj])

                        for jj in range(len(clust[selAS].MAG.values)):
                            AftershocksM.append(clust[selAS].MAG.values[jj])
                        # AftershocksM.append(self.CATS[ic].CLUST['m0'][pos[ii]])

                ForeshocksM = np.array(ForeshocksM)
                AftershocksM = np.array(AftershocksM)
                
                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                ax3 = fig3.add_subplot(nrows, 2, ic+1)
                ax4 = fig4.add_subplot(nrows, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5

                try:
                    b_dataF = calc_b_val(ForeshocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(ForeshocksM), max(ForeshocksM), ax1, ax2, '', '#1E90FF', 'b') #
                    ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], c='b')#, label='Foreshocks')
                    ax4.scatter(b_dataF.Mc, b_dataF.b_val, c='b')
                except:
                    print('No Foreshocks!')

                try:
                    b_dataA = calc_b_val(AftershocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(AftershocksM), max(AftershocksM), ax1, ax3, '', '#F08080', 'r') # self.CATS[ic].name0
                    ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], c='r')#, label='Aftershocks')
                    ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')

                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])

                ax4.set_xlim([Mc, Mc + dm1])
                ax4.set_ylabel('b-value')
                ax4.set_xlabel('Magnitude')
                ax4.set_ylim([0.4, 1.6])
                set_box_aspect(ax4, 0.5)
                ax4.grid()
                # set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')
                
            # fig1.suptitle(' G-R %s' % self.cmethod)
            # fig2.suptitle(' G-R %s' % self.cmethod)

        elif Ptype == 'Foreshocks_b_value-1a_EW':
            x, y = shape2xy2(self.work_folder + 'shapefiles/EW_Pacific.shp')
            x[0] = Make360(x[0])
            x[1] = Make360(x[1])
            fig1 = plb.figure(50250)
            fig2 = plb.figure(50260)
            fig3 = plb.figure(50270)
            fig4 = plb.figure(50280)
            for ic in range(nCats):
                ForeshocksM_E = []
                ForeshocksM_W = []
                AftershocksM_E = []
                AftershocksM_W = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        m_lon = Make360(self.CATS[ic].CLUST.lon0[pos[ii]])
                        m_lat = self.CATS[ic].CLUST.lat0[pos[ii]]
                        Iw = inpolygon(m_lon, m_lat, x[0], y[0])
                        Ie = inpolygon(m_lon, m_lat, x[1], y[1])
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])
                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        selAS = otc > t0
                        selMS = otc == t0
                        if sum(selAS) > 0:
                            if max(clust[selAS].MAG.values) >= self.CATS[ic].CLUST['m0'][pos[ii]]:
                                print('problem with mainshock selection!!')

                        if sum(selFS) >= 2:
                            Mf = clust[selFS].MAG.values
                            tf = otc[selFS]
                            posF = np.argmax(Mf)

                            # get the foreshocks after largest foreshock
                            If = tf > tf[posF]
                            Mf = Mf[If]

                            if Iw == True:
                                for jj in range(len(Mf)):
                                    ForeshocksM_W.append(Mf[jj])
                            if Ie == True:
                                for jj in range(len(Mf)):
                                    ForeshocksM_E.append(Mf[jj])
                        if Iw == True:
                            for jj in range(len(clust[selAS].MAG.values)):
                                AftershocksM_W.append(clust[selAS].MAG.values[jj])
                        if Ie == True:
                            for jj in range(len(clust[selAS].MAG.values)):
                                AftershocksM_E.append(clust[selAS].MAG.values[jj])

                ForeshocksM_E = np.array(ForeshocksM_E)
                ForeshocksM_W = np.array(ForeshocksM_W)
                AftershocksM_E = np.array(AftershocksM_E)
                AftershocksM_W = np.array(AftershocksM_W)

                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                ax3 = fig3.add_subplot(nrows, 2, ic+1)
                ax4 = fig4.add_subplot(nrows, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5

                try:
                    b_dataF = calc_b_val(ForeshocksM_E, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(ForeshocksM_E), max(ForeshocksM_E), ax1, ax4, '', '#1E90FF', 'b')
                    # ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], c='b')#, label='Foreshocks')
                    # ax4.scatter(b_dataF.Mc, b_dataF_E.b_val, c='b')
                    ax1.set_title('East')
                except:
                    print('No Foreshocks!')
                    
                try:
                    b_dataF = calc_b_val(ForeshocksM_W, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(ForeshocksM_W), max(ForeshocksM_W), ax2, ax4, '', '#1E90FF', 'b')
                    # ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], c='b')#, label='Foreshocks')
                    # ax4.scatter(b_dataF.Mc, b_dataF_E.b_val, c='b')
                    ax2.set_title('West')
                except:
                    print('No Foreshocks!')

                try:
                    b_dataA = calc_b_val(AftershocksM_E, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(AftershocksM_E), max(AftershocksM_E), ax1, ax3, '', '#F08080', 'r') # self.CATS[ic].name0
                    # ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], c='r')#, label='Aftershocks')
                    # ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')
                    
                try:
                    b_dataA = calc_b_val(AftershocksM_W, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(AftershocksM_W), max(AftershocksM_W), ax2, ax3, '', '#F08080', 'r') # self.CATS[ic].name0
                    # ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], c='r')#, label='Aftershocks')
                    # ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')

                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])

                ax4.set_xlim([Mc, Mc + dm1])
                ax4.set_ylabel('b-value')
                ax4.set_xlabel('Magnitude')
                ax4.set_ylim([0.4, 1.6])
                set_box_aspect(ax4, 0.5)
                ax4.grid()
                # set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')

            # fig1.suptitle(' G-R %s' % self.cmethod)
            # fig2.suptitle(' G-R %s' % self.cmethod)
        
        elif Ptype == 'Foreshocks_b_value-1a_equal_t':
            fig1 = plb.figure(5025)
            fig2 = plb.figure(5026)
            fig3 = plb.figure(5027)
            fig4 = plb.figure(5028)
            for ic in range(nCats):
                ForeshocksM = []
                AftershocksM = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]
    
                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])
    
                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        
                        selMS = otc == t0
                        
                        if sum(selFS) >= 2:
                            Mf = clust[selFS].MAG.values
                            tf = otc[selFS]
                            posF = np.argmax(Mf)
    
                            # get the foreshocks after largest foreshock
                            If = tf > tf[posF]
                            Mf = Mf[If]
                            
                            dtF = t0 - tf[posF]
                            selAS = (otc > t0) & (otc < t0 + dtF)
                        
                            for jj in range(len(Mf)):
                                ForeshocksM.append(Mf[jj])

                            for jj in range(len(clust[selAS].MAG.values)):
                                AftershocksM.append(clust[selAS].MAG.values[jj])

                ForeshocksM = np.array(ForeshocksM)
                AftershocksM = np.array(AftershocksM)

                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                ax3 = fig3.add_subplot(nrows, 2, ic+1)
                ax4 = fig4.add_subplot(nrows, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5

                try:
                    b_dataF = calc_b_val(ForeshocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(ForeshocksM), max(ForeshocksM), ax1, ax2, self.CATS[ic].name0, '#1E90FF', 'b')
                    ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm >= Mc, b_dataF.Mm <= Mc+dm1)], c='b')#, label='Foreshocks')
                    ax4.scatter(b_dataF.Mc, b_dataF.b_val, c='b')
                except:
                    print('No Foreshocks!')

                try:
                    b_dataA = calc_b_val(AftershocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(AftershocksM), max(AftershocksM), ax1, ax3, self.CATS[ic].name0, '#F08080', 'r')
                    ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm >= Mc, b_dataA.Mm <= Mc+dm1)], c='r')#, label='Aftershocks')
                    ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')

                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])

                ax4.set_xlim([Mc, Mc + dm1])
                ax4.set_ylabel('b-value')
                ax4.set_xlabel('Magnitude')
                ax4.set_ylim([0.4, 1.6])
                ax4.grid()
                set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')
                
            # fig1.suptitle(' G-R %s' % self.cmethod)
            # fig2.suptitle(' G-R %s' % self.cmethod)
            fig1.tight_layout()

        elif Ptype == 'Foreshocks_b_value-1b':
            fig1 = plb.figure(5035)
            fig2 = plb.figure(5036)
            fig3 = plb.figure(5037)
            fig4 = plb.figure(5038)
            for ic in range(nCats):
                ForeshocksM = []
                AftershocksM = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]

                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])

                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        selAS = otc > t0
                        selMS = otc == t0
                        if sum(selAS) > 0:
                            if max(clust[selAS].MAG.values) >= self.CATS[ic].CLUST['m0'][pos[ii]]:
                                print('problem with mainshock selection!!')

                        if sum(selFS) >= 1:
                            Mf = clust[selFS].MAG.values
                            tf = otc[selFS]
                            posF = np.argmax(Mf)

                            # get the foreshocks after largest foreshock
                            If = tf > tf[posF]
                            Mf = Mf[If]

                            for jj in range(len(Mf)):
                                ForeshocksM.append(Mf[jj])

                        for jj in range(len(clust[selAS].MAG.values)):
                            AftershocksM.append(clust[selAS].MAG.values[jj])
                        AftershocksM.append(self.CATS[ic].CLUST['m0'][pos[ii]])

                ForeshocksM = np.array(ForeshocksM)
                AftershocksM = np.array(AftershocksM)

                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                ax3 = fig3.add_subplot(nrows, 2, ic+1)
                ax4 = fig4.add_subplot(nrows, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5

                try:
                    b_dataF = calc_b_val(ForeshocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(ForeshocksM), max(ForeshocksM), ax1, ax2, self.CATS[ic].name0,'#1E90FF','b')
                    ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm>=Mc, b_dataF.Mm<=Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm>=Mc, b_dataF.Mm<=Mc+dm1)], c='b')#, label='Foreshocks')
                    ax4.scatter(b_dataF.Mc, b_dataF.b_val, c='b')
                except:
                    print('No Foreshocks!')

                try:
                    b_dataA = calc_b_val(AftershocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(AftershocksM), max(AftershocksM), ax1, ax3, self.CATS[ic].name0,'#F08080','r')
                    ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm>=Mc, b_dataA.Mm<=Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm>=Mc, b_dataA.Mm<=Mc+dm1)], c='r')#, label='Aftershocks')
                    ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')

                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])

                ax4.set_xlim([Mc, Mc + dm1])
                ax4.set_ylabel('b-value')
                ax4.set_xlabel('Magnitude')
                ax4.set_ylim([0.4, 1.6])
                ax4.grid()
                set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')

            # fig1.suptitle(' G-R %s' % self.cmethod)
            # fig2.suptitle(' G-R %s' % self.cmethod)
            fig1.tight_layout()
            
        elif Ptype == 'Foreshocks_b_value-1c':
            fig1 = plb.figure(5045)
            fig2 = plb.figure(5046)
            fig3 = plb.figure(5047)
            fig4 = plb.figure(5048)
            for ic in range(nCats):
                ForeshocksM = []
                AftershocksM = []
                pos = np.argsort(self.CATS[ic].CLUST.m0)
                for ii in range(len(pos)):
                    if self.CATS[ic].CLUST['c_type'][pos[ii]] == 1:
                        I = self.CATS[ic].deCLUST['clID'] == self.CATS[ic].CLUST['cid'][pos[ii]]
                        clust = self.CATS[ic].deCLUST[I]

                        t0 = UTCDateTime(self.CATS[ic].CLUST.ot0[pos[ii]])

                        otc = []
                        for jj in range(len(clust['Lat'].values)):
                            otc.append(UTCDateTime(clust['Time'].values[jj]))
                        otc = np.array(otc)
                        selFS = otc < t0
                        selAS = otc >= t0
                        selMS = otc == t0
                        if sum(selAS) > 0:
                            if max(clust[selAS].MAG.values) >= self.CATS[ic].CLUST['m0'][pos[ii]]:
                                print('problem with mainshock selection!!')

                        if sum(selFS) >= 1:
                            Mf = clust[selFS].MAG.values
                            for jj in range(len(Mf)):
                                ForeshocksM.append(Mf[jj])

                        for jj in range(len(clust[selAS].MAG.values)):
                            AftershocksM.append(clust[selAS].MAG.values[jj])
                        # AftershocksM.append(self.CATS[ic].CLUST['m0'][pos[ii]])

                ForeshocksM = np.array(ForeshocksM)
                AftershocksM = np.array(AftershocksM)

                if self.CATS[ic].b_data.Mc >= 5:
                    nrows = 2
                else:
                    nrows = 4
                ax1 = fig1.add_subplot(nrows, 2, ic+1)
                ax2 = fig2.add_subplot(nrows, 2, ic+1)
                ax3 = fig3.add_subplot(nrows, 2, ic+1)
                ax4 = fig4.add_subplot(nrows, 2, ic+1)
                Mc = self.CATS[ic].b_data.Mc
                dm1 = 1.5

                try:
                    b_dataF = calc_b_val(ForeshocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataF, min(ForeshocksM), max(ForeshocksM), ax1, ax2, self.CATS[ic].name0,'#1E90FF','b')
                    ax4.plot(b_dataF.Mm[np.logical_and(b_dataF.Mm>=Mc, b_dataF.Mm<=Mc+dm1)], b_dataF.b[np.logical_and(b_dataF.Mm>=Mc, b_dataF.Mm<=Mc+dm1)], c='b')#, label='Foreshocks')
                    ax4.scatter(b_dataF.Mc, b_dataF.b_val, c='b')
                except:
                    print('No Foreshocks!')

                try:
                    b_dataA = calc_b_val(AftershocksM, 0.1, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max)
                    print_b_val(b_dataA, min(AftershocksM), max(AftershocksM), ax1, ax3, self.CATS[ic].name0,'#F08080','r')
                    ax4.plot(b_dataA.Mm[np.logical_and(b_dataA.Mm>=Mc, b_dataA.Mm<=Mc+dm1)], b_dataA.b[np.logical_and(b_dataA.Mm>=Mc, b_dataA.Mm<=Mc+dm1)], c='r')#, label='Aftershocks')
                    ax4.scatter(b_dataA.Mc, b_dataA.b_val, c='r')
                except:
                    print('no aftershocks!')

                ax1.set_xlim([self.CATS[ic].b_data.Mc, 7.0])
                ax1.set_ylim([1, 10000])

                ax4.set_xlim([Mc, Mc + dm1])
                ax4.set_ylabel('b-value')
                ax4.set_xlabel('Magnitude')
                ax4.set_ylim([0.4, 1.6])
                ax4.grid()
                set_legend_title(ax4, self.CATS[ic].name0, 14, 'Impact')

            # fig1.suptitle(' G-R %s' % self.cmethod)
            # fig2.suptitle(' G-R %s' % self.cmethod)
            fig1.tight_layout()
            
        elif Ptype == 'foreshocks_times':
            fig1 = plb.figure(3791)
            fig2 = plb.figure(3792)
            for ic in range(nCats):
                fig3 = plb.figure(3770 + ic + 1)
                ax1 = fig1.add_subplot(4, 2, ic+1)
                ax2 = fig2.add_subplot(4, 2, ic+1)
                foreshocks_time, foreshocks_times_01, Mfs = calc_foreshocks_time_scated(self.CATS[ic].CLUST, self.CATS[ic].deCLUST, 5.0)
                meanFS = []
                for ii in range(len(foreshocks_time)):
                    ax1.scatter(-foreshocks_time[ii], Mfs[ii], s=10)
                    ax3 = fig3.add_subplot(5, 5, ii+1)
                    ax3.stem(foreshocks_time[ii], Mfs[ii])
                for ii in range(len(foreshocks_times_01)):
                    ax2.scatter(-foreshocks_times_01[ii], Mfs[ii], s=10)
                    #meanFS.append(np.mean(-foreshocks_times_01[ii]))
                fig3.suptitle(self.CATS[ic].name0)
                ax1.set_ylabel('Foreshocks Mw')
                ax1.set_xscale('log')
                ax1.set_xlim([0.1, 100])
                ax2.set_xlim([-1, 0])
                ax2.set_ylabel('Foreshocks Mw')
                ax1.set_xlabel('Time before mainshock*')
                ax2.set_xlabel('Scaled time before mainshock*')
                ax1.grid()
                ax2.grid()
                # ax2.plot([np.mean()], [min(Mfs), max(Mfs)])
                set_legend_title(ax1, self.CATS[ic].name0, 14, 'Impact')
                set_legend_title(ax2, self.CATS[ic].name0, 14, 'Impact')
                #set_box_aspect(ax1, 1)


        elif Ptype == 'dM_M':
            fig = plb.figure(3780)
            for ic in range(nCats):
                ax = fig.add_subplot(4, 2, ic+1)
                clust = self.CATS[ic].CLUST
                tp = clust['c_type'].values
                dM_f = clust['dm_f_all'].values
                dM_a = clust['dm_a_all'].values
                Idm = tp == 1 
                m0 = clust['m0'].values
                ax.scatter(m0[Idm], dM_a[Idm], c='r')
                ax.scatter(m0[Idm], dM_f[Idm], c='b')
                ax.plot([min(m0), max(m0)], [np.median(dM_a[Idm & (dM_a > 0)]), np.median(dM_a[Idm & (dM_a > 0)])], c='r')
                ax.plot([min(m0), max(m0)], [np.median(dM_f[Idm & (dM_f > 0)]), np.median(dM_f[Idm & (dM_f > 0)])], c='b')
                ax.set_ylim([0, 4])
                ax.set_xlabel('Mainshock* Mw')
                ax.set_ylabel('M0 - M1')
                ax.set_xlim([min(m0), max(m0)])
                ax.grid()
                set_box_aspect(ax, 1)
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')

            
        elif Ptype == 'foreshocks-Pie60d':
            fig = plb.figure(7410)
            colors = ['whitesmoke','skyblue','dodgerblue']
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                tp = clust['c_type'].values
                nf = clust['nf'].values
                Idm = tp == 1  # types_c = 'Mainshocks'

                nf = nf[Idm]
                nfh0 = sum(nf == 0)
                nfh1 = sum(np.logical_and(nf >= 1, nf < 5))
                nfh5 = sum(nf >= 5)
                # labels = '0≤N<1', '1≤N<5', '5≤N<Inf'
                
                ax.pie([nfh0, nfh1, nfh5], autopct='%1.0f%%',shadow=True, startangle=90, colors=colors)
                ax.set_title('%s %d eq' % (self.CATS[ic].name0, sum(Idm)))
                ax.set_aspect('equal')
                
        elif Ptype == 'foreshocks-Pie60dapr':
            fig = plb.figure(7420)
            colors = ['whitesmoke','skyblue','dodgerblue']
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                tp   = clust['c_type'].values
                nf = clust['nf_apr'].values
                

                Idm = tp == 1  # types_c = 'Mainshocks'

                nf = nf[Idm]
                nfh0 = sum(nf == 0)
                nfh1 = sum(np.logical_and(nf >= 1, nf < 5))
                nfh5 = sum(nf >= 5)
                # labels = '0≤N<1', '1≤N<5', '5≤N<Inf'
                
                ax.pie([nfh0, nfh1, nfh5], autopct='%1.0f%%',shadow=True, startangle=90, colors=colors)
                ax.set_title('%s %d eq' % (self.CATS[ic].name0, sum(Idm)))
                ax.set_aspect('equal')
        
        
        
        elif Ptype == 'foreshocks-Pie':
            fig = plb.figure(7400)
            colors = ['whitesmoke','skyblue','dodgerblue']
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                if self.cmethod == 'WnC':
                    # m = clust['m0'].values
                    tp   = clust['c_type'].values
                    nf = clust['nf'].values
                else:
                    # m = clust['m0'].values
                    tp   = clust['c_type'].values
                    nf = clust['nf_all'].values

                Idm = tp == 1  # types_c = 'Mainshocks'

                nf = nf[Idm]
                nfh0 = sum(nf == 0)
                nfh1 = sum(np.logical_and(nf >= 1, nf < 5))
                nfh5 = sum(nf >= 5)
                labels = '0≤N<1', '1≤N<5', '5≤N<Inf'
                if ic == 1:
                    ax.pie([nfh0, nfh1, nfh5], labels=labels, autopct='%1.0f%%',shadow=True, startangle=90, colors=colors)
                else:
                    ax.pie([nfh0, nfh1, nfh5], autopct='%1.0f%%',shadow=True, startangle=90, colors=colors)
                ax.set_title('%s %d eq' % (self.CATS[ic].name0, sum(Idm)))
                ax.set_aspect('equal')
        
        elif Ptype == 'FMS-Pie':
            
            colors2 = ['whitesmoke','skyblue','dodgerblue']
            labels2 = '0≤N<1', '1≤N<5', '5≤N<Inf'
            colors = ['gray','r','g','b']
            labels = 'Oblique', 'Strike-slip', 'Normal', 'Reverse'
            for ic in range(nCats):
                if self.CATS[ic].name == 'GCMT' or self.CATS[ic].name == 'GCMT2':
                    fig = plb.figure(7450)
                    fig2 = plb.figure(7452)
                    clust = self.CATS[ic].CLUST
                    tp   = clust['c_type'].values
                    Idm = tp == 1  # types_c = 'Mainshocks'
                    clust = clust[Idm]
                    Iodd = clust['fstyle_m'].values == 0
                    Iss  = clust['fstyle_m'].values == 1
                    Inor = clust['fstyle_m'].values == 2
                    Irev = clust['fstyle_m'].values == 3
                    
                    

                    nf = clust['nf'].values
                    nfh0 = nf == 0
                    nfh1 = np.logical_and(nf >= 1, nf < 5)
                    nfh5 = nf >= 5

                    # mainshocks with no foreshocks
                    ax = fig.add_subplot(1,3,1)
                    ax.pie([sum(np.logical_and(nfh0,Iodd)), sum(np.logical_and(nfh0,Iss)), sum(np.logical_and(nfh0,Inor)), sum(np.logical_and(nfh0,Irev))],
                           labels=labels,autopct='%2.0f%%', shadow=True, startangle=90, colors=colors)

                    # ax.pie([nfh0, nfh1, nfh5], autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
                    ax.set_title('Mainshocks with \nno foreshocks')
                    ax.set_aspect('equal')

                    ax = fig.add_subplot(1,3,2)
                    ax.pie([sum(np.logical_and(nfh1,Iodd)), sum(np.logical_and(nfh1,Iss)), sum(np.logical_and(nfh1,Inor)), sum(np.logical_and(nfh1,Irev))],
                           labels=labels,autopct='%2.0f%%', shadow=True, startangle=90, colors=colors)

                    # ax.pie([nfh0, nfh1, nfh5], autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
                    ax.set_title('Mainshocks with \n1≤N<5 foreshocks')
                    ax.set_aspect('equal')

                    ax = fig.add_subplot(1,3,3)
                    ax.pie([sum(np.logical_and(nfh5,Iodd)), sum(np.logical_and(nfh5,Iss)), sum(np.logical_and(nfh5,Inor)), sum(np.logical_and(nfh5,Irev))],
                           labels=labels, autopct='%2.0f%%',shadow=True, startangle=90, colors=colors)

                    # ax.pie([nfh0, nfh1, nfh5], autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
                    ax.set_title('Mainshocks with \n5≤N<Inf foreshocks')
                    ax.set_aspect('equal')

                    fig.suptitle(self.CATS[ic].name0)



                    for ifs in range(4):
                        Ifms = clust['fstyle_m'].values == ifs
                        # mainshocks with no foreshocks
                        ax = fig2.add_subplot(2,2,ifs+1)
                        ax.pie([sum(np.logical_and(nfh0,Ifms)), sum(np.logical_and(nfh1,Ifms)), sum(np.logical_and(nfh5,Ifms))],
                               autopct='%1.0f%%',shadow=True, startangle=90, colors=colors2) # labels=labels2,

                        # ax.pie([nfh0, nfh1, nfh5], autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
                        ax.set_title(labels[ifs])
                        ax.set_aspect('equal')

                    fig2.suptitle(self.CATS[ic].name0)








        elif Ptype == 'b-val-productivity':
            fig = plb.figure(1200)
            fig2 = plb.figure(1201)
            ax2 = fig2.add_subplot(1,1,1)
            markers = 'vs*^.DXP'
            for ic in range(nCats):
                ax = fig.add_subplot(2,3,ic+1)
                clust = self.CATS[ic].CLUST
                na = clust['na'].values
                b_a = clust['b_a'].values
                b_f = clust['b_f'].values
                m = clust['m0'].values
                Mc = self.CATS[ic].b_data.Mc
                I=na>0
                k1,a1,rms1,res = fitK(m, na, Mc, 1)
                ax.scatter(res,b_a[I],marker=markers[ic],c='r',alpha=0.5,label='Aftershocks')
                ax.scatter(res,b_f[I],marker=markers[ic],c='b',alpha=0.5,label='Foreshocks')
                ax2.scatter(res,b_a[I],marker=markers[ic],c='r',alpha=0.5,label='Aftershocks %s' % self.CATS[ic].name)
                ax2.scatter(res,b_f[I],marker=markers[ic],c='b',alpha=0.5,label='Foreshocks %s' % self.CATS[ic].name)
                ax.set_xlabel('Residual Productivity')
                ax.set_ylabel('b-value')
                ax.set_ylim([0, 2])
                ax.legend(title=self.CATS[ic].name,loc='upper left')
            ax2.set_xlabel('Residual Productivity')
            ax2.set_ylabel('b-value')
            ax2.set_ylim([0, 2])
            ax2.legend(loc='upper left')


        elif Ptype == 'AftershockWithForeshocksProductivity':
            cmap = cm.get_cmap('cool',5)
            if fig==None:
                fig = plb.figure(2000)
            grid = plb.GridSpec(2, 9)
            for ic in range(nCats):

                ax = fig.add_subplot(grid[3*ic: 3*ic+2])
                axb = fig.add_subplot(grid[3*ic+2])

                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                if self.cmethod == 'WnC':
                    na = clust['na'].values
                    nf = clust['nf'].values
                    dm_a = clust['dm_a'].values
                    dm_f = clust['dm_f'].values
                else:
                    na = clust['na_all'].values
                    nf = clust['nf_all'].values
                    dm_a = clust['dm_a_all'].values
                    dm_f = clust['dm_f_all'].values

                
                # Idm = np.logical_and(dm_a > self.dmSwarm, dm_f > self.dmSwarm)
                
                # IyesF = np.logical_and(nf > 0, Idm)
                IyesF = nf > 0
                InoF = nf == 0

                minmaxM = np.array([Mc+self.dM, max(m)])
                # fit k for const a - all
                k1,a1,rms1,res1,r2 = fitK(m, na, Mc, 1)
                lb1 = 'k:%2.3f r2:%2.2f' % (k1, r2)
                ax.plot(minmaxM, k1*10**(a1*minmaxM - Mc),c='k', alpha=0.5,label=lb1)
                
                # fit k for const a - yes foreshocks
                k0_yes,a1,rms_k_yes,_,r2_yes = fitK(m[IyesF], na[IyesF], Mc, 1)
                
                # fit k for const a - no foreshocks
                k0_no,a1,rms_k_no,_,r2_no = fitK(m[InoF], na[InoF], Mc, 1)
                
                # bootstrap data
                k_yes, a_yes, _ = bootstrap(m[IyesF],na[IyesF],Mc,1,self.nbs)
                k_no,  a_no,  _ = bootstrap(m[InoF], na[InoF], Mc,1,self.nbs)

                # set histogram range
                nbins = 30
                minmaxK = minmax(np.concatenate([k_yes,k_no]))
                hist_edges = np.arange(minmaxK[0],minmaxK[1],abs(np.diff(minmaxK)/nbins))

                # Make histograms
                axb.hist(k_yes, hist_edges, orientation="horizontal",color=cmap(1), alpha=0.5,edgeColor = 'black')
                axb.hist(k_no,  hist_edges, orientation="horizontal",color=cmap(3), alpha=0.5,edgeColor = 'black')

                k_yes95 = np.quantile(k_yes, 0.95)
                k_yes05 = np.quantile(k_yes, 0.05)
                k_no95 = np.quantile(k_no, 0.95)
                k_no05 = np.quantile(k_no, 0.05)

                xlims = axb.get_xlim()
                axb.plot(xlims, [k0_yes,k0_yes],c='k')
                axb.plot(xlims, [k0_no,k0_no],c='k')
                axb.plot(xlims, [k_yes95,k_yes95],c=cmap(1))
                axb.plot(xlims, [k_no95,k_no95],c=cmap(3))
                axb.plot(xlims, [k_yes05,k_yes05],c=cmap(1))
                axb.plot(xlims, [k_no05,k_no05],c=cmap(3))
                plb.yticks(rotation=45)


                lb_yes = 'With FS - k:%2.3f r2:%2.2f' % (k0_yes, r2_yes)
                ax.plot(minmaxM, k0_yes*10**(a1*minmaxM - Mc),c=cmap(1), alpha=0.5,label=lb_yes)
                ax.scatter(m[IyesF], na[IyesF], c=cmap(1), alpha=0.5, edgecolors='k')

                lb_no = 'No FS - k:%2.3f r2:%2.2f' % (k0_no, r2_no)
                ax.plot(minmaxM, k1*10**(a1*minmaxM - Mc),c=cmap(3), alpha=1.0,label=lb_no)
                ax.scatter(m[InoF], na[InoF], facecolors='none', edgecolors=cmap(3), alpha=1.0)

                ax.set_xlabel('Mainshock magnitud Mw')
                if ic ==0:
                    ax.set_ylabel('Number of aftershocks %d days' % self.DT.days)
                elif ic == 3:
                    ax.set_ylabel('Number of aftershocks %d days' % self.DT.days)
                ax.set_yscale('log')
                ax.set_ylim([1, 1000])
                ax.set_xlim(minmaxM)
                ax.legend(title=self.CATS[ic].name,loc='upper left')
            #fig.tight_layout(pad=3.0)
            fig.suptitle('Minimum dif. between M and largest A.S. and F.S: %2.2f' % self.dmSwarm)
            fig.subplots_adjust(wspace=0.5,left=0.05, right=0.99)

        
        elif Ptype=='ForeshocksDuration':
            foreshocks_time = []
            fig = plb.figure(4160)
            bins = np.arange(-100, 0, 10)
            for ic in range(nCats):
                foreshocks_time = calc_cforeshocks_duration(self.CATS[ic].CLUST, self.CATS[ic].deCLUST, 5.0)
                ax = fig.add_subplot(int(np.ceil(nCats/2)), 4, 2*ic+1)
                ax.hist(foreshocks_time, bins, density=False, facecolor='g', alpha=0.75, width=7)
                ax.set_xlabel('Days')
                set_box_aspect(ax, 1)
                set_legend_title(ax, self.CATS[ic].name0, 14, 'Impact')
                        

        elif Ptype == 'Poissonian':
            fig = plb.figure(3000)
            for ic in range(nCats):
                # fig = plb.figure(3000 + ic)
                
                # ax1 = fig.add_subplot(nCats,2,1 + ic*2)
                # ax1 = fig.add_subplot(nCats,2,1 + ic*2)
                if nCats > 1:
                    ax1 = fig.add_subplot(int(np.ceil(nCats/2)), 4, 2*ic+1)
                    ax2 = fig.add_subplot(int(np.ceil(nCats/2)), 4, ic*2 + 2)
                else:
                    ax1 = fig.add_subplot(2, 2, 1)
                    ax2 = fig.add_subplot(2, 2, 2)
                I = self.CATS[ic].deCLUST['eType'] != 1.0
                BG = self.CATS[ic].deCLUST[I]
                ddays = 60
                dt = ddays*3600*24
                times0 = BG.Time.values
                Mmax = max(self.CATS[ic].deCLUST['MAG'].values) - 1
                Imax = self.CATS[ic].deCLUST['MAG'] > Mmax
                timeMax = self.CATS[ic].deCLUST['Time'][Imax].values
                Lon = self.CATS[ic].deCLUST['Lon'].values
                Lat = self.CATS[ic].deCLUST['Lat'].values
                timeMax0 = []
                mMax0 = self.CATS[ic].deCLUST['MAG'][Imax].values
                for ii in range(len(timeMax)):
                    timeMax0.append(float(UTCDateTime(timeMax[ii])))
                timeMax0 = np.array(timeMax0)
                times = []
                for ii in range(len(times0)):
                    times.append(UTCDateTime(times0[ii]))
                times = np.array(times)
                # times = times - min(times)
                dtv = np.arange(min(times),max(times),dt)
                ll = len(dtv) - 1
                nEv = np.zeros((ll,2))
                for ii in range(ll):
                    nEv[ii,0] = sum(np.logical_and(times >=dtv[ii], times < dtv[ii+1]))
                    nEv[ii,1] = dtv[ii+1] - dt/2
                nEv[:,1] = 1970 + nEv[:,1] / (365*3600*24)
                timeMax0 = 1970 + timeMax0 / (365*3600*24)
                
                meanBG = np.mean(nEv[:,0])
                stdBG = np.std(nEv[:,0])
                ax1.scatter(nEv[:,1],nEv[:,0])
                for ii in range(len(timeMax0)):
                    ax1.plot([timeMax0[ii], timeMax0[ii]], [0, max(nEv[:,0])],c='k')
                    ax1.text(timeMax0[ii],max(nEv[:,0]),'%2.1f' % mMax0[ii])
                ax1.plot([min(nEv[:,1]),max(nEv[:,1])],[meanBG,meanBG],'-k')
                ax1.plot([min(nEv[:,1]),max(nEv[:,1])],[meanBG+stdBG,meanBG+stdBG],':k')
                ax1.plot([min(nEv[:,1]),max(nEv[:,1])],[meanBG-stdBG,meanBG-stdBG],':k')
                ax1.text(np.mean(nEv[:,1]),meanBG+3,'%2.1f' % meanBG)
                ax1.text(np.mean(nEv[:,1]),meanBG+3+stdBG,'%2.1f' % stdBG)
                ax1.set_xlim([min(nEv[:,1]),max(nEv[:,1])])
                ax1.set_xlabel('Years')
                ax1.set_ylabel('EQ in %d days' % ddays)
                set_box_aspect(ax1, 0.5)
                

                # ax2 = fig.add_subplot(nCats,4,3+ic*4)

                data = nEv[:,0]
                bin_interval = int((np.max(data)-np.min(data))/10)
                bins = np.arange(np.max([0,np.min(data)-bin_interval]), np.max(data) + 5*bin_interval,bin_interval) - 0.5*bin_interval
                # entries, bin_edges, patches = ax2.hist(data, bins=bins, density=True, edgeColor = 'black', label='Data')
                entries, bin_edges, patches = ax2.hist(data, bins=bins, density=True, label=self.CATS[ic].name0)
 
                bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                def fit_function(k, lamb):
                    '''poisson function, parameter lamb is the fit parameter'''
                    return poisson.pmf(k, lamb)

                # fit with curve_fit
                print('poisson fit for %s ' % self.CATS[ic].name)
                parameters, cov_matrix = curve_fit(fit_function, bin_middles, entries, p0=[np.mean(data)])

                # plot poisson-deviation with fitted parameter
                x_plot = bins + 0.5*bin_interval # np.arange(0, 15)
                # ax2.hist(nEv[:,0],10)
                # ax2.plot(x_plot,fit_function(x_plot, *parameters),marker='o', linestyle='',label='Lambda: %d' % parameters[0])
                ax2.plot(x_plot,fit_function(x_plot, *parameters),marker='o', linestyle='', label=r'$\lambda: %d$' % parameters[0])
                ax2.plot(np.arange(0,max(data)+5),fit_function(np.arange(0,max(data)+5), *parameters),'-k')
                ax2.set_xlabel('Number of earthquakes')
                ax2.set_ylabel('Occurrence')
                ylims = ax2.get_ylim()
                ax2.set_ylim([0, max(ylims)*1.3])
                set_box_aspect(ax2, 0.75)
                set_legend_title(ax2, self.CATS[ic].name0, 14, 'Impact')
                
                # write to file:
                file_name = '%s%s.poisson.%s.csv' % (self.fileRes, self.CATS[ic].name, self.cmethod)
                # CF = pd.read_csv(file_name)
                class p_data():
                    pass
                p_data.Lambda = round(parameters[0])
                p_data.meanBG = round(meanBG)
                p_data.stdBG = stdBG
                PF = {'Lambda':[p_data.Lambda], 'meanBG':[p_data.meanBG], 'stdBG':[round(p_data.stdBG,2)]}

                PF = pd.DataFrame(data=PF)
                PF.to_csv(file_name)
                self.CATS[ic].p_data = p_data
                
                
                
                # fig.suptitle(self.CATS[ic].name0)
                # ax2.set_title(self.CATS[ic].name0)

                # ax4 = fig.add_subplot(nCats,4,4+ic*4)
                # if sum([self.CATS[ic].name == 'GCMT', self.CATS[ic].name == 'NEIC']) > 0:
                #     Lon = Make360(Lon)
                #     m = Basemap(llcrnrlat=-70, urcrnrlat=max(self.CATS[ic].latlims), llcrnrlon=70, urcrnrlon=340, lat_ts=1)
                # else:
                #     m = Basemap(projection='merc', llcrnrlat=min(self.CATS[ic].latlims), urcrnrlat=max(self.CATS[ic].latlims), llcrnrlon=min(self.CATS[ic].lonlims), urcrnrlon=max(self.CATS[ic].lonlims), lat_ts=1, resolution='i')
                # m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
                # m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
                # # m.readshapefile('/Volumes/GoogleDrive/My Drive/ARCMAP files/gem_active_faults/gem_active_faults','gem_active_faults')
                # m.drawcoastlines(linewidth=0.1, color="white")
                # x, y = m(Lon, Lat)
                #
                # ax4.scatter(x[Imax],y[Imax],50,c='k',zorder=10,label='M > %2.1f' % Mmax)
                # ax4.scatter(x[I],y[I],1,c='b',zorder=9,label='declustetred')
                # l = ax4.legend(loc='north east')
                # l.set_zorder(20)
            #  fig.tight_layout()


        elif Ptype == 'Map_M0M1':
            
            fig = plb.figure(2600)
            for ic in range(nCats):
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                mag = clust['m0'].values
                m_lat = clust['lat0'].values
                m_lon = clust['lon0'].values
                na = clust['na'].values
                nf = clust['nf'].values
                dm_a = clust['dm_a'].values
                dm_f = clust['dm_f'].values


                IyesF = nf > 0
                InoF = nf == 0

                ax = fig.add_subplot(2,3,ic+1)
                if sum([self.CATS[ic].name == 'GCMT', self.CATS[ic].name == 'NEIC']) > 0:
                    m_lon = Make360(m_lon)
                    m = Basemap(llcrnrlat=-70, urcrnrlat=max(self.CATS[ic].latlims), llcrnrlon=70, urcrnrlon=340, lat_ts=1)
                else:
                    m = Basemap(projection='merc', llcrnrlat=min(self.CATS[ic].latlims), urcrnrlat=max(self.CATS[ic].latlims), llcrnrlon=min(self.CATS[ic].lonlims), urcrnrlon=max(self.CATS[ic].lonlims), lat_ts=1, resolution='i')
                m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
                m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
                # m.readshapefile('/Volumes/GoogleDrive/My Drive/ARCMAP files/gem_active_faults/gem_active_faults','gem_active_faults')
                m.drawcoastlines(linewidth=0.1, color="white")
                x, y = m(m_lon, m_lat)
                scat1 = ax.scatter(x[IyesF],y[IyesF],(1.5*mag[IyesF])**1.5,dm_a[IyesF], edgecolors='k', cmap=cm.bwr,zorder=10)
                # scat1 = ax.scatter(m_lon[IyesF],m_lat[IyesF],(1.5*mag[IyesF])**1.5,dm_a[IyesF], edgecolors='k', alpha=0.7,cmap=cm.bwr)
                # scat1.set_clim(0,max(dm_a[IyesF]))
                scat1.set_clim(0,2)

                # ax.set_ylim(self.CATS[ic].latlims)
                # ax.set_xlim(np.array(self.CATS[ic].lonlims))
                # ax.osm = OSM(ax)
                ax.set_title(self.CATS[ic].name)
                if ic == 0:
                    cbar = fig.colorbar(scat1)

            fig.suptitle('Foreshocks')
            fig.subplots_adjust(wspace=0.1,left=0.05, right=0.99)
        
        elif Ptype == 'Map_AftershockWithForeshocksProductivity':
            cmap = cm.get_cmap('cool',5)
            fig = plb.figure(2500)
            for ic in range(nCats):
                clust = self.CATS[ic].CLUST
                Mc = self.CATS[ic].b_data.Mc
                m = clust['m0'].values
                m_lat = clust['lat0'].values
                m_lon = clust['lon0'].values
                na = clust['na'].values
                nf = clust['nf'].values
                dm_a = clust['dm_a'].values
                dm_f = clust['dm_f'].values

                Idm = np.logical_and(dm_a > self.dmSwarm, dm_f > self.dmSwarm)

                IyesF = np.logical_and(nf > 0, Idm)
                InoF = nf == 0

                ax = fig.add_subplot(2,3,ic+1)
                ax.scatter(m_lon[InoF],m_lat[InoF],20,facecolors=cmap(3), edgecolors='k', alpha=0.7)
                ax.scatter(m_lon[IyesF],m_lat[IyesF],20,facecolors=cmap(1), edgecolors='k', alpha=0.7)
                ax.set_ylim(self.CATS[ic].latlims)
                ax.set_xlim(self.CATS[ic].lonlims)
                ax.osm = OSM(ax)
                ax.set_title(self.CATS[ic].name)


        elif Ptype == 'MapCatClusters':
            fig = plb.figure(501)
            for ii in range(len(self.CATS)):
                ax = fig.add_subplot(2,3,ii+1)
                # cid = self.CATS[ii].CLUST.cid[1]
                # Ic = self.CATS[ii].deC .c > 0
                I = self.CATS[ii].deCLUST.clID > 0
                I0 = self.CATS[ii].deCLUST.clID == 0
                ax.scatter(self.CATS[ii].deCLUST.Lon[I0], self.CATS[ii].deCLUST.Lat[I0], 1,c='gray',label=None)
                ax.scatter(self.CATS[ii].deCLUST.Lon[I], self.CATS[ii].deCLUST.Lat[I], 5, self.CATS[ii].deCLUST.clID[I],cmap='hsv',label=None)
                ax.set_ylim(self.CATS[ii].latlims)
                ax.set_xlim(self.CATS[ii].lonlims)
                ax.osm = OSM(ax)
                set_legend_title(ax, self.CATS[ii].name0, 14, 'Impact')


    def GnK(self, ic):
        # 'My Way... :-)'
        cat0 = self.CATS[ic].cat
        sortM = np.argsort(cat0.M)
        sortM = sortM[::-1]
        cat0.M       = cat0.M[sortM]
        cat0.Depth   = cat0.Depth[sortM]
        cat0.Lat     = cat0.Lat[sortM]
        cat0.Long    = cat0.Long[sortM]
        cat0.N       = cat0.N[sortM]
        cat0.datenum = cat0.datenum[sortM]
        cat0.ot      = cat0.ot[sortM]
        len1         = len(cat0.M)

        def windowsGnK(M):
            ''' from http://www.corssa.org/export/sites/corssa/.galleries/articles-pdf/vanStiphout_et_al.pdf'''
            d = 10**(0.1238*M + 0.983) / 2
            t = 10**(0.5409*M - 0.547)
            I = M >= 6.5
            t[I] = 10**(0.032*M[I] + 2.7389)
            t = np.array(np.ceil(t))
            d = np.array(np.ceil(d))
            t = t.astype(int)
            d = d.astype(int)
            return t, d


        def CheckInListC(list1, ClusterList):
            llist = len(list1)
            cols = []
            # run over all events in new list
            for jj in range(llist):
                # check in all clusters in ClusterList
                for mm in range(len(ClusterList)):
                    # locate position of event in Cluster
                    cols1 = np.argwhere(np.array(ClusterList[mm]) == list1[jj])
                    # if event is found in a Cluster
                    if len(cols1) > 0:
                        # save the Cluster id
                        for nn in range(len(cols1)):
                            cols.append(mm)
            cols = np.unique(cols)
            return cols

        [tw_AS, rpt] = windowsGnK(cat0.M)
        Mat_dt = np.zeros((len1, len1))

        for ii in range(len1):
            Mat_dt[ii, :] = cat0.datenum - cat0.datenum[ii]
        Mat_dt = np.ceil(Mat_dt)
        Mat_dt.astype(int)

        print('Calculating distances...')
        MatDist = np.zeros((len1, len1)).astype(int)
        Mat_AS = np.zeros((len1, len1)).astype(int)
        lonlat = np.zeros((len(cat0.Lat), 2))
        lonlat[:, 0] = cat0.Long
        lonlat[:, 1] = cat0.Lat
        for ii in range(len1):
            [_, _, xyr] = MakeCircleUTM(rpt[ii]*1000, cat0.Long[ii], cat0.Lat[ii])
            p = path.Path(xyr)
            MatDist[ii, :] = p.contains_points(lonlat)
            Mat_AS[ii, :] = np.logical_and(Mat_dt[ii, :] > 0, Mat_dt[ii, :] < tw_AS[ii])
        I = np.logical_and(MatDist, Mat_AS)

        print('Cultering using GnK!')
        IDS = np.zeros((sum(sum(I)), 5)); print('ok 1')
        IDS[:, 4] = Mat_dt[I]; del Mat_dt; print('ok 2')
        IDS[:, 2] = MatDist[I]; del MatDist; print('ok 3')

        [iq, jq] = np.meshgrid(np.arange(0, len1), np.arange(0, len1)); print('ok 1')

        IDS[:, 0] = iq[I]; del iq; print('ok 4')
        IDS[:, 1] = jq[I]; del jq; print('ok 5')

        del I
        IDV = np.unique(IDS[:, 0:2])
        ClusterList = []
        print('Associating events to clusters')
        for ii in range(len(IDV)):
        # for ii in range(500):
            ID = IDV[ii]
            # Make list for all events in row2 associated with ID
            list1 = IDS[IDS[:, 0] == ID, 1]

            # add ID to list
            list1 = np.append(list1,ID)

            # Check for ID in Clusterlist
            cols = CheckInListC(list1, ClusterList)

            if len(cols) == 0: # NOT exists list
                list0 = np.unique(list1)

            else: # Exists in list
                list0 = []
                for mm in range(len(cols)):
                    list2 = ClusterList[cols[mm]]
                    for vv in range(len(list2)):
                        list0.append(list2[vv])
                for vv in range(len(list1)):
                    list0.append(list1[vv])
                list0 = np.array(list0)
                list0 = np.unique(list0)

                # remove colums
                ClusterList1 = []
                for nn in range(len(ClusterList)):
                    if nn not in cols:
                        ClusterList1.append(ClusterList[nn])
                ClusterList = ClusterList1

            ClusterList.append(np.array(list0))
            print('%d / %d %d' % (ii, len(IDV), len(ClusterList)))

        print('Add cluster info to EQ table')

        # all events
        c = np.zeros(len(cat0.Lat))
        n = np.zeros(len(cat0.Lat))
        posM = []
        clist = []
        for ii in range(len(ClusterList)):
            posC = ClusterList[ii].astype(int)
            c[posC] = ii+1
            n[posC] = len(ClusterList[ii])
            magsC = cat0.M[ClusterList[ii].astype(int)]
            posM.append(posC[np.argmax(magsC)])
            clist.append(ii+1)
        self.CATS[ic].cat.n = n
        self.CATS[ic].cat.c = c
        self.CATS[ic].posM = np.array(posM)
        self.CATS[ic].clist = np.array(clist)

    # def GnK(self, ic):
    #     # 'Using Gardner and Knopoff 1974 Declustering method'
    #     cat0 = self.CATS[ic].cat
    #     sortM = np.argsort(cat0.M)
    #     sortM = sortM[::-1]
    #     cat0.M       = cat0.M[sortM]
    #     cat0.Depth   = cat0.Depth[sortM]
    #     cat0.Lat     = cat0.Lat[sortM]
    #     cat0.Long    = cat0.Long[sortM]
    #     cat0.N       = cat0.N[sortM]
    #     cat0.datenum = cat0.datenum[sortM]
    #     cat0.ot      = cat0.ot[sortM]
    #
    #     def windowsGnK(M):
    #         ''' from http://www.corssa.org/export/sites/corssa/.galleries/articles-pdf/vanStiphout_et_al.pdf'''
    #         d = 10**(0.1238*M + 0.983) / 2
    #         t = 10**(0.5409*M - 0.547)
    #         I = M >= 6.5
    #         t[I] = 10**(0.032*M[I] + 2.7389)
    #         # if M >= 6.5:
    #         #     t = 10**(0.032*M + 2.7389)
    #         # else:
    #         #     t = 10**(0.5409*M - 0.547)
    #         t = np.array(np.round(t))
    #         d = np.array(np.round(d))
    #         t = t.astype(int)
    #         d = d.astype(int)
    #         return t, d
    #
    #     def CheckInListC(list1, ClusterList):
    #         llist = len(list1)
    #         cols = []
    #         # run over all events in new list
    #         for jj in range(llist):
    #             # check in all clusters in ClusterList
    #             for mm in range(len(ClusterList)):
    #                 # locate position of event in Cluster
    #                 cols1 = np.argwhere(np.array(ClusterList[mm]) == list1[jj])
    #                 # if event is found in a Cluster
    #                 if len(cols1) > 0:
    #                     # save the Cluster id
    #                     for nn in range(len(cols1)):
    #                         cols.append(mm)
    #         cols = np.unique(cols)
    #         return cols
    #
    #     len1 = len(cat0.M)
    #     lonlat = np.zeros((len1,2))
    #     lonlat[:, 0] = cat0.Long
    #     lonlat[:, 1] = cat0.Lat
    #     [tw_AS, rpt] = windowsGnK(cat0.M)
    #     Mat_dt = np.zeros((len1, len1))
    #
    #     for ii in range(len1):
    #         Mat_dt[ii,:]  = cat0.datenum[ii] - cat0.datenum
    #
    #     # for ii in range(len1):
    #     #     MaxTime[ii,:] = tw_AS[ii] * np.ones(len1)
    #     MaxTime = np.ones((len1, 1)) * tw_AS
    #     MaxTime = MaxTime.T
    #     MatRPT = np.ones((len1, 1)) * rpt
    #     # for ii in range(len1):
    #     #     MatRPT[ii,:]  = rpt
    #
    #
    #
    #     print('Calculating distances...')
    #     MatDist = np.zeros((len1, len1))
    #     for ii in range(len1):
    #         MatDist[ii, :] = DistFromEQ(cat0, ii)
    #
    #     I_t = np.logical_and(Mat_dt > 0, Mat_dt < MaxTime)
    #
    #     I_R = MatRPT > MatDist
    #
    #     I = np.logical_and(I_t, I_R)
    #     del I_R
    #     del I_t
    #
    #     print('Cultering using WetzClust!')
    #     [iq, jq] = np.meshgrid(np.arange(0, len1), np.arange(0, len1))
    #     IDS = np.zeros((sum(sum(I)), 5))
    #     IDS[:, 0] = iq[I]
    #     IDS[:, 1] = jq[I]
    #     IDS[:, 2] = MatDist[I]
    #     IDS[:, 3] = MatRPT[I]
    #     IDS[:, 4] = Mat_dt[I]
    #     IDV = np.unique(IDS[:, 0:2])
    #     ClusterList = []
    #     print('Associating events to clusters')
    #     for ii in range(len(IDV)):
    #     # for ii in range(500):
    #         ID = IDV[ii]
    #         # Make list for all events in row2 associated with ID
    #         list1 = IDS[IDS[:, 0] == ID, 1]
    #
    #         # add ID to list
    #         list1 = np.append(list1, ID)
    #
    #         # Check for ID in Clusterlist
    #         cols = CheckInListC(list1, ClusterList)
    #
    #         if len(cols) == 0: # NOT exists list
    #             list0 = np.unique(list1)
    #
    #         else: # Exists in list
    #             list0 = []
    #             for mm in range(len(cols)):
    #                 list2 = ClusterList[cols[mm]]
    #                 for vv in range(len(list2)):
    #                     list0.append(list2[vv])
    #             for vv in range(len(list1)):
    #                 list0.append(list1[vv])
    #             list0 = np.array(list0)
    #             list0 = np.unique(list0)
    #
    #             # remove colums
    #             ClusterList1 = []
    #             for nn in range(len(ClusterList)):
    #                 if nn not in cols:
    #                     ClusterList1.append(ClusterList[nn])
    #             ClusterList = ClusterList1
    #
    #         ClusterList.append(np.array(list0))
    #         print('%d / %d %d' % (ii, len(IDV), len(ClusterList)))
    #
    #     print('Add cluster info to EQ table')
    #
    #     # all events
    #     c = np.zeros(len(cat0.Lat))
    #     n = np.zeros(len(cat0.Lat))
    #     posM = []
    #     clist = []
    #     for ii in range(len(ClusterList)):
    #         posC = ClusterList[ii].astype(int)
    #         c[posC] = ii+1
    #         n[posC] = len(ClusterList[ii])
    #         magsC = cat0.M[ClusterList[ii].astype(int)]
    #         posM.append(posC[np.argmax(magsC)])
    #         clist.append(ii+1)
    #     self.CATS[ic].cat.n = n
    #     self.CATS[ic].cat.c = c
    #     self.CATS[ic].posM = np.array(posM)
    #     self.CATS[ic].clist = np.array(clist)

    def TMC(self, ic):
        # 'My Way... :-)'
        cat0 = self.CATS[ic].cat
        sortM = np.argsort(cat0.M)
        sortM = sortM[::-1]
        cat0.M       = cat0.M[sortM]
        cat0.Depth   = cat0.Depth[sortM]
        cat0.Lat     = cat0.Lat[sortM]
        cat0.Long    = cat0.Long[sortM]
        cat0.N       = cat0.N[sortM]
        cat0.datenum = cat0.datenum[sortM]
        cat0.ot      = cat0.ot[sortM]
        len1         = len(cat0.M)
        
        def windowsTMC(M):
            # t = 0.002*M**7.5
            # t = 0.75*np.exp(M)
            t = 1.0*np.exp(M)
            t[t > 365] = 365
            [r_predicted, _] = WnCfaultL(M, ifault=0)
            d = (AdjustEffectiveRadius(M) * r_predicted) / 1000
            t = np.array(np.ceil(t))
            d = np.array(np.ceil(d))
            t = t.astype(int)
            d = d.astype(int)
            return t, d


        def CheckInListC(list1, ClusterList):
            llist = len(list1)
            cols = []
            # run over all events in new list
            for jj in range(llist):
                # check in all clusters in ClusterList
                for mm in range(len(ClusterList)):
                    # locate position of event in Cluster
                    cols1 = np.argwhere(np.array(ClusterList[mm]) == list1[jj])
                    # if event is found in a Cluster
                    if len(cols1) > 0:
                        # save the Cluster id
                        for nn in range(len(cols1)):
                            cols.append(mm)
            cols = np.unique(cols)
            return cols

        [tw_AS, rpt] = windowsTMC(cat0.M)
        Mat_dt = np.zeros((len1, len1))

        for ii in range(len1):
            Mat_dt[ii, :] = cat0.datenum - cat0.datenum[ii]
        Mat_dt = np.ceil(Mat_dt)
        Mat_dt.astype(int)
        # MaxTime = np.ones((1, len1)) * tw_AS
        # del tw_AS
        # MatRPT = np.ones((len1, 1)) * rpt
        # del rpt
        # MatRPT.astype(int)
        # MaxTime.astype(int)

        print('Calculating distances...')
        MatDist = np.zeros((len1, len1)).astype(int)
        Mat_AS = np.zeros((len1, len1)).astype(int)
        
        lonlat = np.zeros((len(cat0.Lat), 2))
        lonlat[:, 0] = cat0.Long
        lonlat[:, 1] = cat0.Lat
        Ro = kilometers2degrees(rpt)
        for ii in range(len1):
            [_, _, xyr] = MakeCircleUTM(rpt[ii]*1000, cat0.Long[ii], cat0.Lat[ii])
            p = path.Path(xyr)
            MatDist[ii, :] = p.contains_points(lonlat)
            Mat_AS[ii, :] = np.logical_and(Mat_dt[ii, :] > 0, Mat_dt[ii, :] < tw_AS[ii])
            
            # MatDist[ii, :] = DistFromEQ(cat0, ii)
        # MatDist.astype(int)
        # I_t = np.logical_and(Mat_dt > 0, Mat_dt < MaxTime)

        # I_R = MatRPT > MatDist
        # I_R = MatDist 

        I = np.logical_and(MatDist, Mat_AS)
        # del I_R
        # del I_t

        print('Cultering using WetzClust!')
        IDS = np.zeros((sum(sum(I)), 5)); print('ok 2')
        IDS[:, 4] = Mat_dt[I]; del Mat_dt; print('ok 7')
        # IDS[:, 3] = MatRPT[I]; del MatRPT; print('ok 6')
        IDS[:, 2] = MatDist[I]; del MatDist; print('ok 5')

        [iq, jq] = np.meshgrid(np.arange(0,len1), np.arange(0,len1)); print('ok 1')

        IDS[:, 0] = iq[I]; del iq; print('ok 3')
        IDS[:, 1] = jq[I]; del jq; print('ok 4')

        del I
        IDV = np.unique(IDS[:, 0:2])
        ClusterList = []
        print('Associating events to clusters')
        for ii in range(len(IDV)):
        # for ii in range(500):
            ID = IDV[ii]
            # Make list for all events in row2 associated with ID
            list1 = IDS[IDS[:,0]==ID,1]

            # add ID to list
            list1 = np.append(list1,ID)
        
            # Check for ID in Clusterlist
            cols = CheckInListC(list1, ClusterList)

            if len(cols) == 0: # NOT exists list
                list0 = np.unique(list1)

            else: # Exists in list
                list0 = []
                for mm in range(len(cols)):
                    list2 = ClusterList[cols[mm]]
                    for vv in range(len(list2)):
                        list0.append(list2[vv])
                for vv in range(len(list1)):
                    list0.append(list1[vv])
                list0 = np.array(list0)
                list0 = np.unique(list0)
                
                # remove colums
                ClusterList1 = []
                for nn in range(len(ClusterList)):
                    if nn not in cols:
                        ClusterList1.append(ClusterList[nn])
                ClusterList = ClusterList1

            ClusterList.append(np.array(list0))
            print('%d / %d %d' % (ii, len(IDV), len(ClusterList)))

        print('Add cluster info to EQ table')

        # all events
        c = np.zeros(len(cat0.Lat))
        n = np.zeros(len(cat0.Lat))
        posM = []
        clist = []
        for ii in range(len(ClusterList)):
            posC = ClusterList[ii].astype(int)
            c[posC] = ii+1
            n[posC] = len(ClusterList[ii])
            magsC = cat0.M[ClusterList[ii].astype(int)]
            posM.append(posC[np.argmax(magsC)])
            clist.append(ii+1)
        self.CATS[ic].cat.n = n
        self.CATS[ic].cat.c = c
        self.CATS[ic].posM = np.array(posM)
        self.CATS[ic].clist = np.array(clist)
        
    def WnC(self, ic):
        # set long term time window to exclude adtershocks in spatial window for future clustering
        # DTL = datetime.timedelta(days=0)
        cat0 = self.CATS[ic].cat
        # sort cat by magnitude
        posM = np.argsort(cat0.M)
        posM = posM[::-1]
        Im = np.sort(cat0.M)[::-1] > self.CATS[ic].b_data.Mc + self.dM
        posM = posM[Im]
        ll = len(posM)
        c = np.zeros(len(cat0.Lat))
        n = np.zeros(len(cat0.Lat))
        cid = 1
        lonlat = np.zeros((len(cat0.Lat), 2))
        lonlat[:, 0] = cat0.Long
        lonlat[:, 1] = cat0.Lat
        for ii in range(ll):
            print('%s %d / %d' % (self.CATS[ic].name, ii, ll))
            [r_predicted, _] = WnCfaultL(cat0.M[posM[ii]], ifault=0)
            r_predicted = AdjustEffectiveRadius(cat0.M[posM[ii]]) * r_predicted            
            
            # Use in polygon
            # Ro = kilometers2degrees(r_predicted / 1000)
            # [xr, yr, xyr] = makeCircle(Ro, cat0.Long[posM[ii]], cat0.Lat[posM[ii]])
            # p = path.Path(xyr)
            # Ir = p.contains_points(lonlat)
            
            # Use distance
            Rm0 = DistLatLonUTM(cat0.Lat[posM[ii]], cat0.Long[posM[ii]], cat0.Lat, cat0.Long)
            Ir = Rm0 < r_predicted
            
            # It = np.logical_and(cat0.ot > cat0.ot[posM[ii]] - self.DT, cat0.ot < cat0.ot[posM[ii]] + self.DT)
            # ItL= np.logical_and(cat0.ot > cat0.ot[posM[ii]] + self.DT, cat0.ot < cat0.ot[posM[ii]] + DTL)
            It = np.logical_and(cat0.ot > cat0.ot[posM[ii]] - self.DT, cat0.ot < cat0.ot[posM[ii]] + 2 * self.DT)
            Irt = np.logical_and(Ir, It)
            # IrtL = np.logical_and(Ir, ItL)
            Ino_c = c == 0
            Ic = np.logical_and(Ino_c, Irt)
            # if sum(Ic) < sum(Irt):
            #     print('problem')
            c[Ic] = cid
            # c[IrtL] = -1
            n[posM] = sum(Ic)
            cid = cid + 1
        self.CATS[ic].cat.n = n
        self.CATS[ic].cat.c = c
        self.CATS[ic].posM = posM
        self.CATS[ic].clist = np.arange(1, cid, 1)
        print('done clusters for %s' % self.CATS[ic].name)

    def Zclust(self,ic):
        eqCat = self.CATS[ic].cat
        b_value = self.CATS[ic].b_data.b_val
        Mc = self.CATS[ic].b_data.Mc
        D = 1.6
        aNND, vID_p, vID_c = NND_eta( eqCat, b_value, D,Mc)

        fig = plb.figure(9)
        ax = fig.add_subplot(2,3,ic+1)
        eta_binsize = .3
        f_eta_0 = -5
        NND_plot(aNND, eta_binsize, f_eta_0, Mc, len(self.CATS[ic].cat.Lat),ax)
        ax.set_title(self.CATS[ic].name)
        
        
        #==================================3=============================================
        #                       compute re-scaled interevent times and distances
        #================================================================================ 
        # catChild.copy( eqCat)
        # catParent.copy( eqCat)
        # #catChild, catPar = create_parent_child_cat( projCat, dNND)
        # catChild.selEventsFromID(    dNND['aEqID_c'], repeats = True)
        # catParent.selEventsFromID(   dNND['aEqID_p'], repeats = True)
        # print( 'size of parent catalog', catChild.size(), 'size of offspring cat', catParent.size())
        # # note that dictionary dPar here has to include 'b','D' and 'Mc'
        # a_R, a_T = clustering.rescaled_t_r( catChild, catParent, {'b':dPar['b'], 'D':dPar['D'], 'Mc':f_Mc}, correct_co_located = True)
        # RT_file = 'data/%s_RT_Mc_%.1f.mat'%( file_in.split('.')[0], f_Mc)
        # scipy.io.savemat( RT_file, {'R' : a_R, 'T': a_T}, do_compression  = True)
        # #==================================4==============================================================
        # #                       T-R density plots
        # #=================================================================================================
        # a_Tbin = np.arange( dPar['Tmin'], dPar['Tmax']+2*dPar['binx'], dPar['binx'])
        # a_Rbin = np.arange( dPar['Rmin'], dPar['Rmax']+2*dPar['biny'], dPar['biny'])
        # XX, YY, ZZ = data_utils.density_2D( np.log10( a_T), np.log10( a_R), a_Tbin, a_Rbin, sigma = dPar['sigma'])
        #
        # plt.figure(1, figsize= (8,10))
        # ax = plt.subplot(111)
        # ax.set_title( 'Nearest Neighbor Pairs in R-T')
        # #------------------------------------------------------------------------------
        # normZZ = ZZ*( dPar['binx']*dPar['biny']*eqCatMc.size())
        # plot1 = ax.pcolormesh( XX, YY, normZZ, cmap=dPar['cmap'])
        # cbar  = plt.colorbar(plot1, orientation = 'horizontal', shrink = .5, aspect = 20,)
        # #ax.plot(  np.log10( a_T), np.log10( a_R), 'wo', ms = 1.5, alpha = .2)
        # # plot eta_0 to divide clustered and background mode
        # ax.plot( [dPar['Tmin'], dPar['Tmax']],  -np.array([dPar['Tmin'], dPar['Tmax']])+dPar['eta_0'], '-', lw = 1.5, color = 'w' )
        # ax.plot( [dPar['Tmin'], dPar['Tmax']],  -np.array([dPar['Tmin'], dPar['Tmax']])+dPar['eta_0'],'--', lw = 1.5, color = '.5' )
        # #-----------------------labels and legends-------------------------------------------------------
        # #cbar.set_label( 'Event Pair Density [#ev./dRdT]')
        # cbar.set_label( 'Number of Event Pairs',labelpad=-40)
        # ax.set_xlabel( 'Rescaled Time')
        # ax.set_ylabel( 'Rescaled Distance')
        # ax.set_xlim( dPar['Tmin'], dPar['Tmax'])
        # ax.set_ylim( dPar['Rmin'], dPar['Rmax'])



    def CalcGR_Mc(self, ic, save_b):
        cat0 = self.CATS[ic].cat
        print('')
        print('Calculating b-value: %s' % self.CATS[ic].name)
        self.CATS[ic].b_data = calc_b_val(cat0.M, 0.1, 1, 6, 0)
        self.CATS[ic].b_data.MCP = False
        fig = plb.figure(5430)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        print_b_val(self.CATS[ic].b_data, self.CATS[ic].M_bval_min, self.CATS[ic].M_bval_max, ax1, ax2, self.CATS[ic].name0)
        if save_b == True:
            file_name = '%s%s.bvalue.csv' % (self.main_folder, self.CATS[ic].name)
            CF = open(file_name,"w")
            CF.write("Mc,b-value\n")
            CF.write('%2.1f,%2.2f\n' % (self.CATS[ic].b_data.Mc, self.CATS[ic].b_data.b_val))
            CF.write('\n')
            CF.close()
        print('')


        
        
    def trimM(self,ii):
        cat0 = self.CATS[ii].cat
        cat0.M = np.round(cat0.M, 1)
        I = cat0.M >= self.CATS[ii].b_data.Mc
        cat0.Lat = cat0.Lat[I]
        cat0.Long = cat0.Long[I]
        cat0.ot = cat0.ot[I]
        # cat0.ot = cat0.ot.reset_index(drop=True)
        cat0.M = cat0.M[I]
        cat0.Depth = cat0.Depth[I]
        cat0.N = cat0.N[I]
        cat0.datenum = cat0.datenum[I]
        try:
            cat0.strike = cat0.strike[I]
            cat0.rake = cat0.rake[I]
            cat0.dip = cat0.dip[I]
            cat0.fstyle = cat0.fstyle[I]
        except:
            pass
        print('%s: Removing M below: %f' % (self.CATS[ii].name, self.CATS[ii].b_data.Mc))
        print('%s - %d total number of events' % (self.CATS[ii].name,len(cat0.Lat)))
        print('')
        self.CATS[ii].cat = cat0

    
    def Save2ZMAP(self, ii, path2folder):
        cat0 = self.CATS[ii].cat
        fname = '%s/cat_ZMAP_%s.txt' % (path2folder, self.CATS[ii].name)
        CF = open(fname, 'w')
        CF.write('Date, Longitude, Latitude, Depth, Magnitude\n')
        for jj in range(len(cat0.M)):
                CF.write('%s, %3.3f, %3.3f, %d, %1.1f\n' %
                         (cat0.ot[jj], cat0.Long[jj], cat0.Lat[jj], cat0.Depth[jj], cat0.M[jj]))

        CF.close()
        print('%s - ZMAP file is ready' % self.CATS[ii].name)


    def Save2TGf(self, ii, path2folder):
        cat0 = self.CATS[ii].cat
        fname = '%s/cat_%s.txt' % (path2folder, self.CATS[ii].name)
        CF = open(fname, 'w')
        sortid = np.argsort(cat0.ot)
        if self.CATS[ii].name == 'GCMT' or self.CATS[ii].name == 'GCMT2':
            for kk in range(len(cat0.M)):
                jj = sortid[kk]
                CF.write('%d %d %d %d %d %2.2f %d %8.5f %8.5f %d %1.1f 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 %d\n' %
                         (cat0.ot[jj].year, cat0.ot[jj].month, cat0.ot[jj].day, cat0.ot[jj].hour, 
                          cat0.ot[jj].minute, cat0.ot[jj].second, jj, cat0.Lat[jj], cat0.Long[jj], cat0.Depth[jj], cat0.M[jj], cat0.fstyle[jj]))
        else:
            for kk in range(len(cat0.M)):
                jj = sortid[kk]
                CF.write('%d %d %d %d %d %2.2f %d %3.5f %3.5f %d %1.1f 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n' %
                         (cat0.ot[jj].year, cat0.ot[jj].month, cat0.ot[jj].day, cat0.ot[jj].hour, 
                          cat0.ot[jj].minute, cat0.ot[jj].second, jj, cat0.Lat[jj], cat0.Long[jj], cat0.Depth[jj], cat0.M[jj]))
            
        CF.close()
        print('%s - Final txt TG file is ready' % self.CATS[ii].name)
        
    
    
    def trimdata(self,ii):
        cat0 = self.CATS[ii].cat
        
        Idepth = cat0.Depth < self.max_depth
        Ilat1 = cat0.Lat >= self.CATS[ii].latlims[0]
        Ilat2 = cat0.Lat <= self.CATS[ii].latlims[1]
        Ilon1 = cat0.Long >= self.CATS[ii].lonlims[0]
        Ilon2 = cat0.Long <= self.CATS[ii].lonlims[1]
        Ilat = np.logical_and(Ilat1, Ilat2)
        Ilon = np.logical_and(Ilon1, Ilon2)
        Ilatlon = np.logical_and(Ilat, Ilon)
        It1 = cat0.ot >= self.CATS[ii].tinit
        It2 = cat0.ot <= self.CATS[ii].tend
        It = np.logical_and(It1, It2)
        I = np.logical_and(Ilatlon, It)
        I = np.logical_and(I, Idepth)
        
        cat0.Lat = cat0.Lat[I]
        cat0.Long = cat0.Long[I]
        cat0.ot = cat0.ot[I]
        cat0.M = cat0.M[I]
        cat0.Depth = cat0.Depth[I]
        cat0.N = cat0.N[I]
        cat0.datenum = cat0.datenum[I]
        try:
            cat0.strike = cat0.strike[I]
            cat0.rake = cat0.rake[I]
            cat0.dip = cat0.dip[I]
            cat0.fstyle = cat0.fstyle[I]
        except:
            pass
        
        self.CATS[ii].cat = cat0
        self.CATS[ii].years = np.round((self.CATS[ii].cat.datenum.max() - self.CATS[ii].cat.datenum.min()) / 365, 1)
        print('Cat durations: %f years' % self.CATS[ii].years)
        
        print('%s - %d event after cut!' % (self.CATS[ii].name, len(cat0.datenum)))


    def GetCat(self, name):

        if name == 'NEIC':
            t1 = datetime.datetime(1976, 1, 1, 0, 0, 0, 0)
            # tend = datetime.datetime(2021,3,1,0,0,0,0)
            # t1 = datetime.datetime(1991,1,1,0,0,0,0)
            tend = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)
            lats = [-90, 90]
            lons = [-180, 180]
            M_bval_min = 4.0
            M_bval_max = 6.0
            name0 = 'NEIC'
            cat = read_NEIC('%s%s.csv' % (self.main_folder, name))

        elif name == 'GCMT2back':
            t1 = datetime.datetime(1991,1,1,0,0,0,0)
            tend = datetime.datetime(2020,1,1,0,0,0,0)
            lats = [-90, 90]
            lons = [-180, 180]
            M_bval_min = 4.0
            M_bval_max = 6.0
            name0 = 'GCMT'
            cat = read_GCMT('%s%s.csv' % (self.main_folder, name))

        elif name =='OVSICORI':
            t1 = datetime.datetime(2010,1,1,0,0,0,0) # 2014
            tend = datetime.datetime(2023,1,1,0,0,0,0)
            lats = [9, 10]
            lons = [-85, -82]
            M_bval_min = 2.0
            M_bval_max = 5.0
            name0 = 'Costa Rica'
            cat = read_OVSICORI('%s%s.csv' % (self.main_folder, name))

        elif name =='CR':
            t1 = datetime.datetime(2010,1,1,0,0,0,0) # 2014
            tend = datetime.datetime(2023,1,1,0,0,0,0)
            lats = [6, 11.5]
            lons = [-87, -82]
            M_bval_min = 1.5
            M_bval_max = 5.0
            name0 = 'Navarro'
            cat = read_CR('%s%s.csv' % (self.main_folder, name))


        elif name == 'GCMT2':
            # t1 = datetime.datetime(1991,1,1,0,0,0,0)
            t1 = datetime.datetime(1976, 1, 1, 0, 0, 0, 0)
            tend = datetime.datetime(2020, 1, 1, 0, 0, 0, 0)
            lats = [-90, 90]
            lons = [-180, 180]
            M_bval_min = 4.0
            M_bval_max = 9.0
            name0 = 'GCMT'
            cat = read_GCMT('%s%s.csv' % (self.main_folder, name))

        elif name == 'GCMT':
            t1 = datetime.datetime(1990,1,1,0,0,0,0)
            tend = datetime.datetime(2020,1,1,0,0,0,0)
            lats = [-90, 90]
            lons = [-180, 180]
            M_bval_min = 4.0
            M_bval_max = 9.0
            name0 = 'GCMT'
            cat = read_GCMT('%s%s.csv' % (self.main_folder, name))

        elif name == 'KOERI':
            # t1 = datetime.datetime(1985,1,1,0,0,0,0)
            t1 = datetime.datetime(1995,1,1,0,0,0,0)
            tend = datetime.datetime(2020,1,1,0,0,0,0)
            lats = [36, 42]
            lons = [35, 45]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'Turkey'
            cat = read_KOERI('%s%s.csv' % (self.main_folder, name))

        elif name == 'KOERI_E':
            t1 = datetime.datetime(1995,1,1,0,0,0,0)
            # t1 = datetime.datetime(2018,1,1,0,0,0,0)
            tend = datetime.datetime(2022,1,1,0,0,0,0)
            lats = [35, 44]
            lons = [32.5, 46]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'E. Turkey'
            cat = read_KOERI2('%s%s.csv' % (self.main_folder, 'KOERI2'))

        elif name == 'KOERI2':
            t1 = datetime.datetime(1995,1,1,0,0,0,0)
            # t1 = datetime.datetime(2018,1,1,0,0,0,0)
            tend = datetime.datetime(2022,1,1,0,0,0,0)
            lats = [35, 44]
            lons = [25, 46]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'Turkey'
            cat = read_KOERI2('%s%s.csv' % (self.main_folder, name))

        elif name == 'NOA':
            # t1 = datetime.datetime(1985,1,1,0,0,0,0)
            t1 = datetime.datetime(2012,1,1,0,0,0,0)
            tend = datetime.datetime(2020,1,1,0,0,0,0)
            lats = [33.5, 41]
            lons = [19, 26]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'Greece'
            cat = read_NOA('%s%s.csv' % (self.main_folder, name))

        elif name == 'AUOT':
            # t1 = datetime.datetime(1985,1,1,0,0,0,0)
            t1 = datetime.datetime(1995,1,1,0,0,0,0)
            tend = datetime.datetime(2022,5,31,0,0,0)
            lats = [33.0, 43]
            lons = [19, 30]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'Greece'
            cat = read_AUOT('%s%s.csv' % (self.main_folder, name))

        elif name == 'JMA':
            t1 = datetime.datetime(1990,1,1,0,0,0,0)
            tend = datetime.datetime(2011,1,1,0,0,0,0)
            lats = [29, 46]
            lons = [127, 150]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'Japan'
            cat = read_JMA('%s%s.csv' % (self.main_folder, name))

        elif name == 'JMA2':
            t1 = datetime.datetime(2011,1,1,0,0,0,0)
            tend = datetime.datetime(2020,1,1,0,0,0,0)
            lats = [29, 46]
            lons = [127, 150]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'Japan'
            cat = read_JMA('%s%s.csv' % (self.main_folder, 'JMA'))

        elif name == 'ALASKA2':
            t1 = datetime.datetime(2010,1,1,0,0,0,0)
            tend =datetime.datetime(2022,1,1,0,0,0,0)
            lats = [47, 67]
            lons = [-180, -134]
            M_bval_min = 1.5
            M_bval_max = 4.0
            name0 = 'Alaska'
            cat = read_ALASKA('%s%s.csv' % (self.main_folder, name))

        elif name == 'NCEDC':
            t1 = datetime.datetime(1995,1,1,0,0,0,0)
            tend =datetime.datetime(2022,1,1,0,0,0,0)
            lats = [35, 40]
            lons = [-124, -117]
            M_bval_min = 1.5
            M_bval_max = 4.0
            name0 = 'N. California'
            cat = read_NCEDC('%s%s.csv' % (self.main_folder, name))

        elif name == 'SCEDC':
            t1 = datetime.datetime(1985,1,1,0,0,0,0)
            tend = datetime.datetime(2020,12,21,0,0,0,0)
            lats = [32.0, 37.0]
            lons = [-120, -114]
            M_bval_min = 2.0
            M_bval_max = 4.0
            name0 = 'S. California'
            cat = read_SCEDC('%s%s.csv' % (self.main_folder, name))

        elif name == 'INGV2':
            t1 = datetime.datetime(1990,1,1,0,0,0,0)
            tend = datetime.datetime(2021,5,22,0,0,0,0)
            lats = [35, 47] # [36, 46]#
            lons = [6, 19] # [10, 18]
            M_bval_min = 1.0
            M_bval_max = 4.0
            name0 = 'Italy'
            cat = read_INGV2('%s%s.csv' % (self.main_folder, name))

        elif name == 'samp':
            t1 = datetime.datetime(1990, 1, 1, 0, 0, 0, 0)
            tend = datetime.datetime(2000, 1, 1, 0, 0, 0, 0)
            lats = [-1.5, 1]
            lons = [-128.5, -126]
            M_bval_min = 3.0
            M_bval_max = 4.0
            name0 = 'ETAS'
            cat = read_smap('%s%s.csv' % (self.main_folder, name))

        print("%s: %d events loaded" % (name, len(cat.M)))
        Catalog = Catalog0()
        area = polygon_area(cat.Lat, cat.Long, radius=6378137)
        Catalog.latlims = lats
        Catalog.lonlims = lons
        Catalog.area = area
        Catalog.tinit = UTCDateTime(t1)
        Catalog.tend = UTCDateTime(tend)
        Catalog.cat = cat
        Catalog.M_bval_min = M_bval_min
        Catalog.M_bval_max = M_bval_max
        Catalog.name = name
        Catalog.name0 = name0
        
        result_cat = Catalog
            
        return result_cat

    def Cat2ZMAP(self, path2folder):
        for ii in range(len(self.CATS)):
            self.Save2ZMAP(ii, path2folder)

    def MakeCattxt4TG(self, path2folder):
        for ii in range(len(self.CATS)):
            self.Save2TGf(ii, path2folder)
            
        
    
    def trimcats(self):
        for ii in range(len(self.CATS)):
            self.trimdata(ii)
    def CalcGR_Mc_cats(self, loader, save_b):
        if loader == False:
            for ii in range(len(self.CATS)):
                self.CalcGR_Mc(ii, save_b)
        else:
            for ii in range(len(self.CATS)):
                file_name = '%s%s.bvalue.csv' % (self.main_folder, self.CATS[ii].name)
                CF = pd.read_csv(file_name)
                class b_data():
                    pass
                b_data.Mc = float(CF['Mc'].values[0])
                b_data.b_val = float(CF['b-value'].values[0])
                self.CATS[ii].b_data = b_data
                
    def AboveMcCat(self,Mc=-999):
        
        for ii in range(len(self.CATS)):
            if Mc > -999:
                self.CATS[ii].b_data.Mc = Mc
                self.CATS[ii].b_data.MCP = True
                print('Using Mc=%1.1f for %s' % (Mc, self.CATS[ii].name))
            self.trimM(ii)


    def writeDeclust(self,ic):
        cat0 = self.CATS[ic].cat
        namef = '%scat_%s.declust.%s.csv' % (self.fileRes, self.CATS[ic].name , self.cmethod)
        CF1 = open(namef,"w")
        CF1.write('MAG,Time,Lat,Lon,Depth,clID\n')
        for ii in range(len(cat0.M)):
            CF1.write('%2.1f,%s,%f,%f,%2.1f,%d\n' % (cat0.M[ii], cat0.ot[ii], cat0.Lat[ii],cat0.Long[ii],cat0.Depth[ii],cat0.c[ii]))
        CF1.close()
        print('Finished writing declust cat: %s' % self.CATS[ic].name)


    def calaAreaSeis(self,dxy):
        CL = pd.read_csv(self.work_folder + 'Py/GlobalCoastLine.csv')
        for ii in range(len(self.CATS)):
            [area, N, X, Y, xyn, I_N] = Calc_Area(self.CATS[ii].cat.Long,self.CATS[ii].cat.Lat,dxy,0.1)
            print('%s %d km2' % (self.CATS[ii].name, round(area)))
            # N = np.flipud(N)


            fig = plb.figure(66800)
            # ax = fig.add_subplot(1,2,1)
            # m = Basemap(projection='merc', llcrnrlat=min(Y), urcrnrlat=max(Y), llcrnrlon=min(X), urcrnrlon=max(X), lat_ts=1, resolution='i')
            # # m.drawmapboundary(fill_color='#A6CAE0', linewidth=0,zorder=1)
            # # m.fillcontinents(color='grey', alpha=0.7, lake_color='grey',zorder=2)
            # m.drawcoastlines(linewidth=0.1, color="k",zorder=2)
            # x, y = m(self.CATS[ii].cat.Long,self.CATS[ii].cat.Lat)
            # ax.scatter(x,y,1,'w',zorder=10)
            # m.imshow(N,  cmap=plb.cm.jet,interpolation='nearest',extent=[min(X), max(X)+dxy,min(Y),max(Y)+dxy],zorder=4)

            ax2 = fig.add_subplot(2,4,ii+1)
            ax2.imshow(np.flipud(I_N), cmap=plb.cm.jet, interpolation='nearest',extent=[min(X), max(X)+dxy,min(Y),max(Y)+dxy],alpha=0.5,zorder=4)
            ax2.scatter(self.CATS[ii].cat.Long,self.CATS[ii].cat.Lat,0.5,'w',label='Area %d km2' % round(area),zorder=5, alpha=0.5)
            ax2.grid()
            ax2.plot(CL.Lon.values,CL.Lat.values,'-k')
            # ax2.osm = OSM(ax2)
            ax2.set_xlim([min(X), max(X)])
            ax2.set_ylim([min(Y), max(Y)])
            set_legend_title(ax2, self.CATS[ic].name0, 14, 'Impact')



    def calcRclust(self):
        print('Calc R...')
        for ii in range(len(self.CATS)):
            MmaxFS = np.zeros(len(self.CATS[ii].CLUST))
            MmaxFS60 = np.zeros(len(self.CATS[ii].CLUST))
            dtmaxFS = np.zeros(len(self.CATS[ii].CLUST))
            dtmaxFS60 = np.zeros(len(self.CATS[ii].CLUST))
            MmaxAS = np.zeros(len(self.CATS[ii].CLUST))
            dtmaxAS = np.zeros(len(self.CATS[ii].CLUST))

            RmaxFS = np.zeros(len(self.CATS[ii].CLUST))
            RmaxAS = np.zeros(len(self.CATS[ii].CLUST))
            Rpred = np.zeros(len(self.CATS[ii].CLUST))
            
            for jj in range(len(self.CATS[ii].CLUST)):
                [r_predicted,_] = WnCfaultL(self.CATS[ii].CLUST.m0[jj],ifault=0)
                Rpred[jj] = (AdjustEffectiveRadius(self.CATS[ii].CLUST.m0[jj]) * r_predicted) / 1000
                
                if self.CATS[ii].CLUST.na[jj] > 0:
                    clust = self.CATS[ii].deCLUST[self.CATS[ii].CLUST.cid[jj] == self.CATS[ii].deCLUST.clID].reset_index(drop=True)
                    times = np.zeros(len(clust))
                    # print(jj)
                    dists = DistFromEQ2(clust,np.argmin(np.abs(clust.MAG - self.CATS[ii].CLUST.m0[jj])))
                    t0 = UTCDateTime(self.CATS[ii].CLUST.ot0[jj])
                    for tt in range(len(clust)):
                        times[tt] = UTCDateTime(clust.Time[tt])
                    I_as   = times > t0
                    I_fs   = times < t0
                    I_fs60 = np.logical_and(times < t0, times >= (t0-self.ddays*24*3600))

                    if sum(I_fs) > 0:
                        MmaxFS[jj]  = np.max(clust.MAG[I_fs])
                        dtmaxFS[jj] = t0 - UTCDateTime(times[np.argmax(clust.MAG[I_fs])])
                        RmaxFS[jj] = dists[I_fs].max()
                        try:
                            dtmaxFS60[jj] = t0 - UTCDateTime(times[np.argmax(clust.MAG[I_fs60])])
                            MmaxFS60[jj]  = np.max(clust.MAG[I_fs60])
                        except:
                            pass

                    if sum(I_as) > 0:
                        MmaxAS[jj]  = np.max(clust.MAG[I_as])
                        dtmaxAS[jj] = UTCDateTime(times[np.argmax(clust.MAG[I_as])]) - t0
                        RmaxAS[jj] = dists[I_as].max()

            self.CATS[ii].CLUST['MmaxFS'] = MmaxFS
            self.CATS[ii].CLUST['dtmaxFS'] = dtmaxFS
            self.CATS[ii].CLUST['MmaxFS60'] = MmaxFS60
            self.CATS[ii].CLUST['dtmaxFS60'] = dtmaxFS60
            self.CATS[ii].CLUST['MmaxAS'] = MmaxAS
            self.CATS[ii].CLUST['MmaxFS'] = MmaxFS
            self.CATS[ii].CLUST['dtmaxAS'] = dtmaxAS
            self.CATS[ii].CLUST['RmaxFS'] = RmaxFS
            self.CATS[ii].CLUST['RmaxAS'] = RmaxAS
            self.CATS[ii].CLUST['Rpred'] = Rpred




    def makeClusters(self, cmethod, loader):
        self.cmethod = cmethod
        for ii in range(len(self.CATS)):
            if cmethod == 'WnC':
                if loader == False:
                    print('Calc clusters for %s - %s' % (self.CATS[ii].name, cmethod))
                    self.WnC(ii)
                    self.MakaClusterData(ii)
                    self.writeDeclust(ii)
                else:
                    nameC = '%s%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ii].name, self.dmSwarm, self.cmethod)
                    CLUST = pd.read_csv(nameC)
                    print('Cdata loaded: %s' % self.CATS[ii].name)
                    self.CATS[ii].CLUST = CLUST


            elif cmethod == 'GnK':
                if loader == False:
                    print('Calc clusters for %s - %s' % (self.CATS[ii].name, cmethod))
                    self.GnK(ii)
                    self.MakaClusterData(ii)
                    self.writeDeclust(ii)
                    Im = self.CATS[ii].CLUST['m0'].values >= self.CATS[ii].b_data.Mc + self.dM
                    self.CATS[ii].CLUST = self.CATS[ii].CLUST[Im].reset_index(drop=True)

                else:
                    nameC = '%s%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ii].name, self.dmSwarm, self.cmethod)
                    try:
                        CLUST = pd.read_csv(nameC)
                        Im = CLUST['m0'].values >= self.CATS[ii].b_data.Mc + self.dM
                        CLUST = CLUST[Im].reset_index(drop=True)
                        print('Cdata loaded: %s' % self.CATS[ii].name)
                        self.CATS[ii].CLUST = CLUST
                    except:
                        print('Unable to load cluster data')
                        break
            
            elif cmethod == 'TMC':
                if loader == False:
                    print('Calc clusters for %s - %s' % (self.CATS[ii].name, cmethod))
                    self.TMC(ii)
                    self.MakaClusterData(ii)
                    self.writeDeclust(ii)
                    Im = self.CATS[ii].CLUST['m0'].values >= self.CATS[ii].b_data.Mc + self.dM
                    self.CATS[ii].CLUST = self.CATS[ii].CLUST[Im].reset_index(drop=True)

                else:
                    nameC = '%s%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ii].name, self.dmSwarm, self.cmethod)
                    try:
                        CLUST = pd.read_csv(nameC)
                        Im = CLUST['m0'].values >= self.CATS[ii].b_data.Mc + self.dM
                        CLUST = CLUST[Im].reset_index(drop=True)
                        print('Cdata loaded: %s' % self.CATS[ii].name)
                        self.CATS[ii].CLUST = CLUST
                    except:
                        print('Unable to load cluster data')
                        break


            elif cmethod == 'ZnBZ':
                nameC = '%scat_%s.clusters.a%2.1f.%s.csv' % (self.fileRes, self.CATS[ii].name, self.dmSwarm, self.cmethod)
                CLUST = pd.read_csv(nameC)
                Im = CLUST['m0'].values >= self.CATS[ii].b_data.Mc + self.dM
                CLUST = CLUST[Im]
                Id = CLUST['depth0'].values < self.max_depth
                CLUST = CLUST[Id]
                CLUST = CLUST.reset_index(drop=True)
                print('Cdata loaded: %s' % self.CATS[ii].name)
                self.CATS[ii].CLUST = CLUST

            DCname = '%scat_%s.declust.%s.csv' % (self.fileRes, self.CATS[ii].name,  cmethod)
            
            # deCLUST = pd.read_csv(DCname, index_col=[0])
            deCLUST = pd.read_csv(DCname)
            cols = deCLUST.columns
            check = 0
            for kk in range(len(cols)):
                if cols[kk] == 'eType':
                    check = 1
            if check == 0:
                deCLUST = define_Event_type(deCLUST)
                deCLUST.to_csv(DCname)
            self.CATS[ii].deCLUST = deCLUST


