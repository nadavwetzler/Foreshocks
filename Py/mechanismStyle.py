import numpy as np
# based on the Matlab code of Miroslav Hallo from: http://geo.mff.cuni.cz/~hallo/PTplots/index.html
# following: Frohlich, C. (1992): Triangle diagrams: ternary graphs to display similarity and diversity of
# earthquake focal mechanisms, Physics of the Earth and Planetary Interiors, 75, 193-198.
# Code author: nadav Wetzler
# Geological Survey of Israel, Jerusalem
# E-mail: nadavw@gsi.gov.il
# Revision 9/2021: The first version of the function.

import matplotlib.pyplot as plb
from obspy.imaging.beachball import beachball
from obspy.imaging.beachball import beach

def mechaStyle(strike, dip, rake):
    strike = np.array(strike) * np.pi / 180.
    dip    = np.array(dip) * np.pi / 180.
    rake   = np.array(rake) * np.pi / 180.

    N         = len(strike)
    n0        = np.zeros((N,3))
    u0        = np.zeros((N,3))
    P_azimuth = np.zeros(N)
    T_azimuth = np.zeros(N)
    P_theta   = np.zeros(N)
    T_theta   = np.zeros(N)
    mClass    = np.zeros(N)
    dT        = np.zeros(N)
    dP        = np.zeros(N)
    dB        = np.zeros(N)

    # Normals and slip vectors
    n0[:,0] = -np.sin(dip) * np.sin(strike)
    n0[:,1] =  np.sin(dip) * np.cos(strike)
    n0[:,2] = -np.cos(dip)
    u0[:,0] =  np.cos(rake) * np.cos(strike) + np.cos(dip) * np.sin(rake) * np.sin(strike)
    u0[:,1] =  np.cos(rake) * np.sin(strike) - np.cos(dip) * np.sin(rake) * np.cos(strike)
    u0[:,2] = -np.sin(rake) * np.sin(dip)

    # PT-axes
    P_osa = (n0-u0) / np.rot90(np.tile(np.sqrt(np.sum((n0-u0)**2,axis=1)),(3,1)))
    T_osa = (n0+u0) / np.rot90(np.tile(np.sqrt(np.sum((n0+u0)**2,axis=1)),(3,1)))
    P_osa[P_osa[:,2]>0,:] = -P_osa[P_osa[:,2]>0,:]
    T_osa[T_osa[:,2]>0,:] = -T_osa[T_osa[:,2]>0,:]

    # Compute all angles
    for i in range(N):
        # Get azimuths and dip angles
        P_azimuth[i] = np.arctan2(P_osa[i,0],P_osa[i,1])
        P_theta[i]   = np.arccos(np.abs(P_osa[i,2]))

        T_azimuth[i] = np.arctan2(T_osa[i,0],T_osa[i,1])
        T_theta[i] = np.arccos(np.abs(T_osa[i,2]))

        # Get mechanism class
        dT[i] = np.pi/2 - T_theta[i]
        dP[i] = np.pi/2 - P_theta[i]
        dB[i] = np.arcsin(np.real(np.sqrt(1 - np.sin(dT[i])**2 - np.sin(dP[i])**2)))

        if np.sin(dB[i])**2 > 0.75: # Strike-slip
            mClass[i] = 1
        elif np.sin(dP[i])**2 > 0.75: # Normal
            mClass[i] = 2
        elif np.sin(dT[i])**2 > 0.75: # Reverse
            mClass[i] = 3
        else: # Odd
            mClass[i] = 0

    return mClass, dP*180/np.pi, dT*180/np.pi, dB*180/np.pi

def deg2hv(dP,dT,dB,dN):
    z = np.arctan(np.sin(dT)/np.sin(dP))-np.deg2rad(45)
    try:
        if np.isnan(z)==True:
            z = np.arctan(1) - np.deg2rad(45)
    except:
        pass


    h1 = (np.cos(dB)*np.sin(z)) / (np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z))
    v1 = (np.cos(dN)*np.sin(dB)-np.sin(dN)*np.cos(dB)*np.cos(z))/(np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z))
    return z, h1, v1


def TernaryFMS(ax,dP0,dT0,dB0,mClass):
    dP0 = np.deg2rad(dP0)
    dT0 = np.deg2rad(dT0)
    dB0 = np.deg2rad(dB0)
    dN = np.deg2rad(35.26)

    # Border
    zero0 = 1E-10
    [z, h1, v1] = deg2hv(np.deg2rad(90),zero0,zero0,dN)
    [z, h2, v2] = deg2hv(zero0,np.deg2rad(90),zero0,dN)
    [z, h3, v3] = deg2hv(zero0,zero0,np.deg2rad(90),dN)
    ax.plot([h1, h2, h3, h1],[v1, v2, v3, v1],'-k')

    # Grid
    ax.plot(0,0,'+',c=[0.6, 0.6, 0.6])

    Np = 100
    h1_grid = []
    v1_grid = []
    for ii in range(Np):
        dP = np.deg2rad(60)
        dT = (ii)*(np.deg2rad(30)/(Np-1))
        dB = np.arcsin(np.sqrt(1 - (np.sin(dP)**2 + np.sin(dT)**2)))
        z = np.arctan(np.sin(dT)/np.sin(dP))-np.deg2rad(45)
        h1_grid.append((np.cos(dB)*np.sin(z)) / (np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z)))
        v1_grid.append((np.cos(dN)*np.sin(dB)-np.sin(dN)*np.cos(dB)*np.cos(z))/(np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z)))

    ax.plot(h1_grid,v1_grid,c=[0.6, 0.6, 0.6])

    dB = np.deg2rad(60)
    dP = np.deg2rad(60)
    h2_grid = np.zeros(Np)
    v2_grid = np.zeros(Np)
    for ii in range(Np):
        dT = (ii)*(np.deg2rad(88)/(Np-1.499))
        dP = np.sin(np.sqrt(1 - (np.sin(dP)**2 + np.sin(dT)**2)))
        z = np.arctan(np.sin(dT)/np.sin(dP))-np.deg2rad(45)
        h2_grid[ii]=((np.cos(dB)*np.sin(z)) / (np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z)))
        v2_grid[ii]=((np.cos(dN)*np.sin(dB)-np.sin(dN)*np.cos(dB)*np.cos(z))/(np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z)))
    ind = np.argsort(h2_grid)
    ax.plot(h2_grid[ind],v2_grid[ind],c=[0.6, 0.6, 0.6])

    h3_grid = []
    v3_grid = []
    for ii in range(Np):
        dT = np.deg2rad(60)
        dP = (ii+1)*(np.deg2rad(30)/(Np-1))
        dB = np.sin(np.sqrt(1 - (np.sin(dP)**2 + np.sin(dT)**2)))
        z = np.arctan(np.sin(dT)/np.sin(dP))-np.deg2rad(45)
        h3_grid.append((np.cos(dB)*np.sin(z)) / (np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z)))
        v3_grid.append((np.cos(dN)*np.sin(dB)-np.sin(dN)*np.cos(dB)*np.cos(z))/(np.sin(dN)*np.sin(dB)+np.cos(dN)*np.cos(dB)*np.cos(z)))

    ax.plot(h3_grid,v3_grid,c=[0.6, 0.6, 0.6])
    mClass = np.array(mClass)
    dv = 0.2
    beach1 = beach([0,45,-90], facecolor='g', edgecolor='k', xy=(h1, v1), width=0.20, linewidth=0.2, zorder=3)
    ax.add_collection(beach1)
    ax.text(h1-dv, v1-dv, '%2.1f' % (sum(mClass==2)/len(mClass)*100))


    beach2 = beach([0,45,90], facecolor='b', edgecolor='k', xy=(h2, v2), width=0.20, linewidth=0.2, zorder=3)
    ax.add_collection(beach2)
    ax.text(h2, v2-dv, '%2.1f' % (sum(mClass==3)/len(mClass)*100))

    beach3 = beach([45,90,180], facecolor='r', edgecolor='k', xy=(h3, v3), width=0.20, linewidth=0.2, zorder=3)
    ax.add_collection(beach3)
    ax.text(h3, v3+dv, '%2.1f' % (sum(mClass==1)/len(mClass)*100))


    [z, h, v] = deg2hv(dP0,dT0,dB0,dN)

    ax.set_aspect('equal', adjustable='box')
    clr1 = 'rgb'
    for ii in range(len(clr1)):
        I = mClass == ii+1
        ax.scatter(h[I],v[I],15,clr1[ii])
    I = mClass == 0
    ax.scatter(h[I],v[I],15,[0.5,0.5,0.5])
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    


# mClass,dP,dT,dB = mechaStyle(np.ones(15)*130,np.ones(15)*44,np.ones(15)*-80)

# mClass,dP,dT,dB = mechaStyle([23],[65],[-88])
# fig1 = plb.figure()
# ax = fig1.add_subplot(1,1,1)
# TernaryFMS(ax,dP,dT,dB,mClass)
# plb.show()
# print(mClass,dP,dT,dB)
