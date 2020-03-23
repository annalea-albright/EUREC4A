#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:25:04 2020

@author: annaleaalbright
"""

# import packages 
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units

#%% 
# =============================================================================
# Set parameters for the bulk model
# =============================================================================

r_bl  = np.array([-1.,-3.,-5.])     # BL radiative cooling rate in K/day r_bl
v      = 5.                         # background wind speed (e.g. from large-scale circulation) in m/s
th_sfc = 301.                       # surface temperature in K
th_0   = 298.                       # reference surface temperature theta_0 in K
gam    = 5./1000.                   # temperature gradient above the inversion (K/m, dtheta/dz)
r_ft  = -1.                         # FT radiative cooling rate in K/day
p_sfc  = 101300.                    # surface pressure in Pa
tau_lcl= 15.*60.                    # time scale in Eq. 10 in sec
dq     = 3e-3                       # humditiy inversion jump in g/g [3 g/kg]
C_d    = 0.001                      # drag coefficient 
A      = 0.2                       # inversion entrainment efficiency

ni = len(r_bl)
print("BL radiative cooling rate in K/day values are " + str(r_bl))
print("surface temperature in K = " + repr(th_sfc))

#%% 
# =============================================================================
# Constants and conversion to SI units
# =============================================================================

r_ft  = r_ft/86400.                 # FT radiative cooling rate: convert from K/day to K/s
r_bl = r_bl/86400                   # BL radiative cooling rate: convert from K/day to K/s
eps    = 0.61                       # for calculating virtual potential temperature in Eqs. 8 and 9
gamd   = 9.8/1000.                  # dry adiabatic lapse rate K/m
Rl     = 287.1                      # universal gas constant (J/kg/K)
g      = 9.81                       # gravitational acceleration, m/s2

dt = 5.*60.                         # time step in s (5 mins)
end_time = 6.*24.*60.*60.           # simulation time in s (6 days)
nt = np.int(np.floor(end_time/dt))
print("number of time steps is " +repr(nt))
time = np.linspace(0, dt*nt, nt+1)
print("length of time is " +repr(len(time)))

#%% 
# =============================================================================
# Bolton LCL approximation
# =============================================================================
def calc_LCL_Bolton(T, RH):
    # source: https://climlab.readthedocs.io/en/latest/_modules/climlab/utils/thermo.html
    '''Compute the Lifiting Condensation Level (LCL) for a given temperature and relative humidity
    Inputs:  T is temperature in Kelvin
            RH is relative humidity (dimensionless)
    Output: LCL in meters
    This is height (relative to parcel height) at which the parcel would become saturated during 
    adiabatic ascent. Based on approximate formula from Bolton (1980 MWR) as given by Romps (2017 JAS)
    '''
    g = 9.81   # gravity
    cpd = 1005.7   # specific heat at constant pressure of dry air, J/kg/K
    Tadj = T-55.  # in Kelvin
    return cpd/g*(Tadj - (1/Tadj - np.log(RH)/2840.)**(-1))

# Results below are insensitive to using other formulations, i.e. Romps analytic solution, iterative solution (from Metpy, for instance)


#%%
# =============================================================================
# Pre-allocate variables
# =============================================================================

q_sat = 0.
q_sat_cold = 0.
h     = np.zeros([ni, nt+1])
print("size of h (# cooling rates vs. time) " + repr(np.shape(h)))
th_bl = np.zeros([ni, nt+1])
q_bl  = np.zeros([ni, nt+1])
f_th  = np.zeros([ni, nt+1])
f_q   = np.zeros([ni, nt+1])
f_thv = np.zeros([ni, nt+1])
d_thv = np.zeros([ni, nt+1])
lcl = np.zeros([ni, nt+1])
lcl_Romps = np.zeros([ni, nt+1])
rh_bl = np.zeros([ni, nt+1])
w_m = np.zeros([ni, nt+1])
#w_e = np.zeros([ni, nt+1])

#%%

# =============================================================================
# Calculate
# =============================================================================
for k in range(0, ni): # loop over cooling rates
    q_rbl = r_bl[k]
    print("boundary layer cooling rate = " + repr(q_rbl*86400) +" K/day")
   
    #step 1: find equilibrium values without moisture
    
    th_bl[k,0] = (r_ft*th_0 + (A-(A+1)*r_ft/q_rbl)*gam*C_d*v*th_sfc)/ \
    (r_ft      + (A-(A+1)*r_ft/q_rbl)*gam*C_d*v)
    f_th[k,0]  = C_d*v*(th_sfc-th_bl[k,0])
    h[k,0]     = -(A+1)/q_rbl*f_th[k,0] 
    d_thv[k,0]  = th_0 + gam*h[k,0]  - th_bl[k,0] # thetav = theta(1+eps*q) but q =0 
    q_bl[k,0]   = 0.
    w_e        = -r_ft/gam # w_e = -w_FT in dry case
    q_ft       = 0.
    
    """
    print("dry equilibrium: h = "+ str(round(h[k,0],2)) +" m, \
          th = "+ str(round(th_bl[k,0],2)) +" K, \
          qbl = "+ str(round(q_bl[k,0] *1.e3,2))+", \
          dthv = "+ str(round(d_thv[k,0],2)) +" K, \
          entrainment velocity = "+ str(round(w_e*100,2)) +"cm/s") 
    """
    
    # calculate water vapor saturation mixing ratio as a function of pressure and temperature
    # In Naumann et al, 2019, the surface is assumed to be saturated with surface water 
    # vapor mixing ratio, qsfc = qsat(theta_surf, z=0).
    q_sat = mpcalc.saturation_mixing_ratio(p_sfc * units.Pa, th_sfc * units.K) # g/g
    #print("saturation water vapor mixing ratio: q_sat = "+ str(round(q_sat,3)) +" (in g/g)")

    # time loop to find adjustment to circulation
    
    # f_theta in dry case * 1000 ~ 5
    
    for i in range(1,nt): 
        f_th[k,i]  = C_d*v*(th_sfc-th_bl[k,i-1]) # kinematic surface sensible heat flux
        # surface assumed to be saturated, q_sfc = q_sat
        f_q[k,i]    = C_d*v*(q_sat- q_bl[k,i-1]) # the kinematic surface moisture flux
        f_thv[k,i]  = f_th[k,i] + eps*th_bl[k,i-1]*f_q[k,i] # the kinematic buoyancy flux         
        w_e        = A*f_thv[k,i]/d_thv[k,i-1] # entrainment velocity (flux-jump relation)        
        th_bl[k,i]  = th_bl[k,i-1] + (q_rbl + (1/h[k,i-1])*( w_e*(th_0+gam*h[k,i-1]-th_bl[k,i-1]) + f_th[k,i] ))*dt
        q_bl[k,i]   =  q_bl[k,i-1] + ((1/h[k,i-1])*( w_e*(q_ft- q_bl[k,i-1]) +  f_q[k,i]))*dt 
        d_thv[k,i]  = (th_0 + gam*h[k,i-1])*(1+eps*q_ft) - (th_bl[k,i]*(1+eps*q_bl[k,i]))
        # thv = theta(1 + eps*q), q = mixing ratio of water vapor
        h[k,i]      =     h[k,i-1] + (r_ft/gam + w_e )*dt
        
        # calculate the LCL    
        qsat_bl = mpcalc.saturation_mixing_ratio(1000 * units.hectopascal, th_bl[k,i] * units.K) # g/g
        #print("B.L. saturation water vapor mixing ratio: q_sat_bl = "+ str(round(qsat_bl,3)) +" (in g/g)")
        rh_bl[k,i] = q_bl[k,i]/ np.float64(qsat_bl.magnitude)
        #print("RH = "+ str(round(rh_bl[k,i],3)*100) +" %")
        lcl[k,i] = calc_LCL_Bolton(th_bl[k,i], rh_bl[k,i]) # Bolton LCL approximation
        #lcl[k,i] = LCL_Romps.lcl(p_sfc, th_bl[k,i] ,rh_bl[k,i])  # Romps analytical solution
        
        # convective mass flux triggered?
        if lcl[k,i] < h[k,i]: 
            w_m[k,i]  = - (h[k,i]-lcl[k,i])/(tau_lcl)
            h[k,i]  = h[k,i] + w_m[k,i] *dt 
            
        else:
            w_m[k,i] = 0
            h[k,i]  = h[k,i]
        
        q_ft = np.max((0., q_bl[k,i]  - dq))    # fixed dq
        
        
    print("moist equilibirum: h = "+ str(round(h[k,i-1],2)) + " m, th = "+ str(round(th_bl[k,i-1],2)) +" K, qbl = "+ str(round(q_bl[k,i-1]*1.e3,2)) +", dthv = "+ str(round(d_thv[k,i-1],2)))
    print("we = "+ str(round(w_e,2))+", dth = "+ str(round((th_0+gam*h[k,i-1]-th_bl[k,i-1]),2))+", F_B = "+ str(round(f_thv[k,i-1],2)))
    


#%% 
# =============================================================================
# Plotting 
# =============================================================================

# %% conversion for plotting


time  = time/60./60./24.        # s --> day
r_ft = r_ft*86400.              # K/s --> K/day
q_rbl = q_rbl*86400.            # K/s --> K/day
rho_air =1.16                   # density of air (kg/m3)
f_th  = f_th*rho_air*1003.4     # W/m2 (sensible heat flux = f_theta * rho * Cp)
lv0 = 2.5e6                     # latent heat of vaporization per unit mass, J/kg
f_q   = f_q*rho_air*lv0         # W/m2 (latent heat flux = rho * lv0 * f_q )
w_m = w_m * 100                 # m/s --> cm/s
w_e = A*f_thv/d_thv*1e2         # cm/s
d_theta = th_0 +(gam*h)-th_bl   # inversion strength K
q_bl = q_bl*1.e3                # g/g --> g/kg



#%% Plotting Fig. A1

# subplots: h, theta_BL, q_BL, FB, SH, LH, delta_theta_v, delta_theta, wm <0; we > 0 in cm/s 

plt.rcParams.update({'font.size': 25})
fig, ax = plt.subplots(3,3,figsize=(35,25))

ax[0,0].plot(time, h[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') # royalblue
ax[0,0].plot(time, h[1,:], linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[0,0].plot(time, h[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[0,0].set_ylabel('h (m)')
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)
ax[0,0].legend(loc='best')
ax[0,0].set_xlim([-0.3,time[-3]])

ax[1,0].plot(time, th_bl[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[1,0].plot(time, th_bl[1,:], linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[1,0].plot(time, th_bl[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[1,0].set_ylabel('$\\theta$$_{BL}$ (K)')
ax[1,0].set_ylim([296 , 302])
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)
ax[1,0].set_xlim([-0.3,time[-3]])


ax[2,0].plot(time, q_bl[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[2,0].plot(time, q_bl[1,:], linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[2,0].plot(time, q_bl[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[2,0].set_ylabel('q$_{BL}$ (g/kg)')
ax[2,0].set_xlabel('time (days)')
ax[2,0].spines['right'].set_visible(False)
ax[2,0].spines['top'].set_visible(False)
ax[2,0].set_xlim([-0.3,time[-3]])

# f_thv
ax[0,1].plot(time, f_thv[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[0,1].plot(time, f_thv[1,:], linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[0,1].plot(time, f_thv[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[0,1].set_ylabel('kinematic buoyancy flux (Km/s)')
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)
ax[0,1].set_xlim([time[2],time[-3]])
ax[0,1].set_ylim([0,0.05])


# f_th
ax[1,1].plot(time, f_th[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[1,1].plot(time, f_th[1,:],  linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[1,1].plot(time, f_th[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[1,1].set_ylabel('SH (W/m2)')
#ax[1,1].set_ylabel('SH = $\rho$c$_{p}$F$_{\\theta$_{BL}$}$ (W/m2)')
plt.sca(ax[1,1])
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)
ax[1,1].set_xlim([-0.3,time[-3]])

# f_q
ax[2,1].plot(time, f_q[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[2,1].plot(time, f_q[1,:],  linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[2,1].plot(time, f_q[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[2,1].set_ylabel('LH (W/m2)')
ax[2,1].set_xlabel('time (days)')
ax[2,1].spines['right'].set_visible(False)
ax[2,1].spines['top'].set_visible(False)
ax[2,1].set_xlim([time[2],time[-3]])
ax[2,1].set_ylim([0.1,200])

# d_thv
ax[0,2].plot(time, d_thv[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[0,2].plot(time, d_thv[1,:], linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[0,2].plot(time, d_thv[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[0,2].set_ylabel('$\Delta$$\\theta$$_v$ (K)')
ax[0,2].spines['right'].set_visible(False)
ax[0,2].spines['top'].set_visible(False)
ax[0,2].set_xlim([-0.3,time[-3]])
ax[0,2].set_ylim([0 , 5.5])

# d_theta = th_0+gam*h-th_bl
ax[1,2].plot(time, d_theta[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[1,2].plot(time, d_theta[1,:],  linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[1,2].plot(time, d_theta[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[1,2].set_ylabel('$\Delta$$\\theta$ (K)')
plt.sca(ax[1,1])
ax[1,2].spines['right'].set_visible(False)
ax[1,2].spines['top'].set_visible(False)
ax[1,2].set_xlim([-0.3,time[-3]])
ax[1,2].set_ylim([0 , 5.5])


wm0 = w_m[0,:]
wm0_nz = wm0[np.nonzero(wm0)]
wm1 = w_m[1,:]
wm1_nz = wm1[np.nonzero(wm1)]
wm2 = w_m[2,:]
wm2_nz = wm2[np.nonzero(wm2)]
ax[2,2].plot(time[np.nonzero(wm0)], wm0_nz, linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[2,2].plot(time[np.nonzero(wm1)], wm1_nz,  linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[2,2].plot(time[np.nonzero(wm2)], wm2_nz, linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[2,2].plot(time, w_e[0,:], linewidth=4, label='Q$_{BL}$ -1 K/day', color='dodgerblue') 
ax[2,2].plot(time, w_e[1,:],  linewidth=4, label='Q$_{BL}$ -3 K/day', color='mediumblue')
ax[2,2].plot(time, w_e[2,:], linewidth=4, label='Q$_{BL}$ -5 K/day', color= 'darkblue')
ax[2,2].set_ylabel('w$_m$<0 (m/s), w$_e$>0 (m/s)')
ax[2,2].set_xlabel('time (days)')
ax[2,2].spines['right'].set_visible(False)
ax[2,2].spines['top'].set_visible(False)
ax[2,2].set_xlim([-0.3,time[-3]])

