# -*- coding: utf-8 -*-

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pandas import DataFrame
import metpy.calc as mpcalc
from metpy.units import units
import seaborn as sns
plt.rcParams.update({'font.size': 18})

#%%
"""
Script to make initial plots for moisture and temperature variables (either for all, clear, or cloudy sondes). 
Option to focus on a single day, multiple days, or all sondes. Using rolling means to explore spatiotemporal 
variability (i.e. window=6 like half a circle, window=12 like a circle-mean). Exploratory analysis to estimate 
the mixed layer height (i.e. constant vertical gradients in moisture/temperature variables, height of max. RH, LCL) 
and the inversion base height; also looking at coherence among these methods.  Also takes a preliminary look at mixing 
lines (i.e. following Betts and Albrecht, 1987).
"""

#%%
# =============================================================================
# Load sondes in HALO circle + pre-process
# =============================================================================  

filepath = '/Users/annaleaalbright/Dropbox/EUREC4A/Dropsondes/Data/all_sondes_w_cloud_flag.nc'
all_sondes = xr.open_dataset(filepath)

#%% 

# choose only HALO dropsondes in circle
def HALO_circle(sondes): 
    HALO_sondes = sondes.where(sondes.Platform=='HALO')
    # HALO circle, source: http://eurec4a.eu/fileadmin/user_upload/eurec4a/documents/Flight-Planning-Coordinates.pdf
    lon_center, lat_center = -57.717,13.3
    lon_pt_circle, lat_pt_circle = -57.245,14.1903
    r_circle = np.sqrt((lon_pt_circle-lon_center)**2+(lat_pt_circle-lat_center)**2)
    buffer = 0.01
    lat_N = lat_center + r_circle + buffer*r_circle
    lat_S = lat_center - r_circle - buffer*r_circle
    lon_W = lon_center - r_circle - buffer*r_circle
    lon_E = lon_center + r_circle + buffer*r_circle     
    sondes_circle = HALO_sondes.where((HALO_sondes['lat']<lat_N) & (HALO_sondes['lat']>lat_S)\
                                      & (HALO_sondes['lon']<lon_E) &(HALO_sondes['lon']>lon_W), drop=True)
    nsondes_HALO_circle = len(sondes_circle.launch_time)
    print(nsondes_HALO_circle, "sondes launched in HALO circle")
    return sondes_circle

#%%
# =============================================================================
# Choose either ONE DAY/SUBSET OF DAYS, or ALL DAYS
# =============================================================================  
if True:
    # One day: 
    day_str = '2020-01-28' 
    sondes_oneday = all_sondes.sel(launch_time=day_str) 
    # Select multiple days:
    #sondes_multiple_days = all_sondes.sel(launch_time=slice('2020-01-19','2020-01-30')) 
    Sondes_Circle = HALO_circle(sondes_oneday)
    nsondes_HALO_circle = len(Sondes_Circle.launch_time)
    
    # remove sondes with large nunber of NaNs
    # threshold : require this many non-NA values
    sondes_circle = Sondes_Circle.dropna(dim="launch_time", \
                                         subset=['alt','pres','u_wind','v_wind', 'wspd','lat','lon','mr', 
                                         'theta', 'theta_e', 'theta_v', 'rh', 'T', 'dz', 'q', 'dp'], \
                                         how='any', thresh=8_000) 
    sondes_circle.rh.plot()
    
    nsondes_qc = len(sondes_circle.launch_time)
    print(nsondes_qc, "sondes after quality control on " + day_str)
          
# =============================================================================
# Or choose all days
# =============================================================================  

if False:    
    Sondes_Circle = HALO_circle(all_sondes)
    nsondes_HALO_circle = len(Sondes_Circle.launch_time)
    
    # remove sondes with large nunber of NaNs
    # threshold : require this many non-NA values
    sondes_circle = Sondes_Circle.dropna(dim="launch_time", \
                                         subset=['alt','pres','u_wind','v_wind', 'wspd','lat','lon','mr', 
                                         'theta', 'theta_e', 'theta_v', 'rh', 'T', 'dz', 'q', 'dp'], \
                                         how='any', thresh=10_000) 
    sondes_circle.rh.plot()
    
    nsondes_qc = len(sondes_circle.launch_time)
    print("# sondes after quality control : ", nsondes_qc)

#%% add density variable to all sondes

def calculateDensity(sondes):
    
    Rd = 287.058	 # gas constant for dry air J/kg/K
    eps = 0.62198
    
    # alternate equation: rho=pres/RaT*(1+x)/(1+x*Rw/Rd)
    """sondes = sondes.assign(rho=sondes["pres"]/(Rd*sondes["tdry"])*(1+sondes["mr"])/
                                                           (1+1.609*sondes["mr"]))
    """
    
    # equation: rho = P/(Rd * Tv), where Tv = T(1 + mr/eps)/(1+mr)
    sondes = sondes.assign(rho = sondes["pres"] / (Rd * sondes["T"] * \
                                   (1 + (sondes["mr"]/eps))/(1 + sondes["mr"]))) 
    
    # methods give answers within 0.05% of each other
    return sondes

sondes_circle = calculateDensity(sondes_circle)

#%%                
# =============================================================================
#             # divide into clear (0) and cloudy (1) sondes
# =============================================================================  

clearsky_sondes = sondes_circle.where(sondes_circle.cloud_flag==0, drop=True)
cloudy_sondes = sondes_circle.where(sondes_circle.cloud_flag==1, drop=True)
nsondes_clear = len(clearsky_sondes.launch_time)
nsondes_cloudy = len(cloudy_sondes.launch_time)
print(nsondes_clear, "clear-sky sondes,", nsondes_cloudy, "cloudy sondes, and", nsondes_qc , "total sondes after quality control")


#%% 
# =============================================================================
#             functions 
# =============================================================================  


# =============================================================================
#             rolling means
# =============================================================================  
def calc_rolling_mean(sondes, var_name, window):
    
    """ specify variable and window for centered rolling mean/moving average. 
    """
    
    var = sondes[var_name]
    var_rm = var.rolling(launch_time=window, min_periods = window, center=True).mean(skipna=True)
    # alternatively, require fewer samples per rolling mean, i.e. min_periods=int(window/2)
    #var_rm = var.rolling(launch_time=window, min_periods = int(window/2), center=True).mean(skipna=True)
    
    return var_rm

def plot_rolling_mean(var_rm, xlim_min, xlim_max, ylim_max, xlabel):

    plt.figure(figsize=(8,10))
    for i in range(0, len(var_rm.launch_time)):
        plt.plot(var_rm.isel(launch_time=i), var_rm.alt.values,color="lightgrey", linewidth=2,alpha=0.5)
    plt.plot(var_rm.mean(dim='launch_time'),var_rm.alt.values,linewidth=4, color='black')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel(xlabel)
    plt.ylabel('Altitude / m')
    plt.ylim([0, ylim_max])
    plt.xlim([xlim_min, xlim_max])
    if False:
        plt.grid()
        plt.axvline(x=0, color='black')

# =============================================================================
#            plotting all sondes
# =============================================================================        
def plot_all_sondes(var, xlabel, xlim1, xlim2):

    plt.figure(figsize=(8,10))
    for i in range(0, len(var.launch_time)):
        plt.plot(var.isel(launch_time=i), var.alt.values,color="lightgrey", linewidth=2,alpha=0.5)
    plt.plot(var.mean(dim='launch_time'),var.alt.values,linewidth=4, color='black')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel(xlabel)
    plt.ylabel('Altitude / m')
    plt.ylim([0, 4000])
    plt.xlim([xlim1, xlim2])
    if False:
        plt.axvline(x=0, color='black')
        
def plot_mean_sigma_all_days(var, nsigma, xlabel, xlim1, xlim2):

    mean = var.mean(dim='launch_time')
    sigma =  var.std(dim='launch_time')
    alt = var.alt.values    
    if True:
        plt.figure(figsize=(8,10))
        plt.plot(mean, alt, linewidth=4, color='black', label='mean') #label='$\mu$')
        plt.fill_betweenx(alt, mean, mean -(nsigma*sigma), color='lightgrey', alpha=0.6, label='1$\sigma$')
        plt.fill_betweenx(alt, mean, mean + (nsigma*sigma),  color='lightgrey', alpha=0.6)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel(xlabel)
        plt.ylabel('Altitude / m')
        plt.ylim([0, 4000])
        plt.xlim([xlim1, xlim2])
        #plt.legend(loc='best')
        # remove yticks and ylabel
        if False:
            plt.tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
            plt.ylabel('')
        if False:
            plt.axvline(x=0, color='black')

    #return mean, sigma
# =============================================================================
#             static stability and LCL from Bolton approximation
# =============================================================================  
def static_stability(T_vec_K, theta_vec, pres_vec_Pa):
    """ Equation source: https://www.ncl.ucar.edu/Document/Functions/Contributed/static_stability.shtml
      Inputs:
         -- Temp (K)
         -- Pressure (Pa)
    
    Static stability measures the gravitational resistance of an atmosphere to vertical displacements. 
    It results from fundamental buoyant adjustments, and so it is determined by the vertical stratification 
    of density or potential temperature.
    """
    
    static_stability = - (T_vec_K/theta_vec) * (theta_vec.diff(dim="alt")/pres_vec_Pa.diff(dim="alt"))
    return static_stability

def calc_LCL_Bolton(T, RH):
    '''
    Source: https://climlab.readthedocs.io/en/latest/_modules/climlab/utils/thermo.html
    
    Compute the Lifiting Condensation Level (LCL) for a given temperature and relative humidity
    Inputs:  T is temperature in Kelvin
            RH is relative humidity (dimensionless)
    Output: LCL in meters
    This is height (relative to parcel height)*** at which the parcel would become saturated during 
    adiabatic ascent. Based on approximate formula from Bolton (1980 MWR) as given by Romps (2017 JAS)
    
    *** Note that this is relative to parcel height, so initial parcel height needs to be 
    added if not using surface values
    '''
    g = 9.81       # gravity
    cpd = 1005.7   # specific heat at constant pressure of dry air, J/kg/K
    Tadj = T-55.   # in Kelvin
    LCL_Bolton = cpd/g*(Tadj - (1/Tadj - np.log(RH)/2840.)**(-1))
    return LCL_Bolton

def LCL_metpy(pres, temp, dp):
    """
    Inputs are pressure (hPa), temperature (degC), and dew point (degC)
    """
    # pre-allocate
    len_time = len(pres.launch_time)
    lcl_p = [None] * len_time
    lcl_t = [None] * len_time
    lcl_ht = [None] * len_time
    pres_avg = np.zeros(len_time)
    T_avg = np.zeros(len_time)
    dp_avg = np.zeros(len_time)
    
    for i in range(len_time):
        
        # input: sonde mean-value from 60-210 m for pressure, temp, and dew point temp
        pres_avg[i] = pres[i,5:20].mean(dim="alt", skipna=True)
        T_avg[i] = temp[i,5:20].mean(dim="alt", skipna=True)
        dp_avg[i] = dp[i,5:20].mean(dim="alt", skipna=True)
        
        # break loop and restart if NaN encountered
        if np.isnan(T_avg[i]) == True:
            continue
    
        # lcl_p and lcl_t are the values of pressure (hPa) and temperature (K) at LCL, respectively
        lcl_p[i], lcl_t[i] = (
                        mpcalc.lcl(pres_avg[i]*units.hPa,
                        T_avg[i]*units.degC,
                        dp_avg[i]*units.degC,
                        max_iters=200))
        
        lcl_p[i] = lcl_p[i].magnitude
        lcl_t[i] = lcl_t[i].magnitude
    
        # find height where PLCL- 10hPa < P < PLCL + 10hPa
        altitude = dp['alt']
        lcl_ht[i] = altitude.where((pres[i]<lcl_p[i]+10) & \
                      (pres[i]>lcl_p[i] - 10)).mean().values
    
        # convert None to NaN
        lcl_ht = np.array(lcl_ht,dtype=float)
        
    return lcl_ht
#%% 
# =============================================================================
# choose all sondes (sondes_circle), clear-sky (clearsky_sondes), or cloudy (cloudy_sondes)
# ============================================================================= 

sondes =  sondes_circle 

#%%
# =============================================================================
#          LOAD raw data
# ============================================================================= 
mixing_ratio = sondes['mr']
theta = sondes['theta']
theta_v = sondes['theta_v']
theta_e = sondes['theta_e']
temp = sondes['T']
RH = sondes['rh']
wspd = sondes['wspd']
pressure = sondes['pres']
u_wind = sondes['u_wind']
v_wind = sondes['v_wind']
wdir = sondes['wdir']

#%% unit conversion 

T_vec_K = temp + 273.15 # degC to K
pres_vec_Pa = pressure * 100 # convert hPa to Pa

#%% calculate static stability
static_stab = static_stability(T_vec_K, theta, pres_vec_Pa) * 100

#%% plot all sondes

plot_all_sondes(mixing_ratio, xlabel='Mixing ratio / g/kg ', xlim1 =-10, xlim2=20)
plot_all_sondes(theta, xlabel='theta / K', xlim1 =295, xlim2=320)
plot_all_sondes(theta_v, xlabel='theta_v / K', xlim1 =297, xlim2=320)
plot_all_sondes(theta_e, xlabel='theta_e / K', xlim1 =305, xlim2=355)
plot_all_sondes(RH, xlabel='RH / %', xlim1 =0, xlim2=100)
plot_all_sondes(wspd, xlabel='wind speed / m/s', xlim1 =0, xlim2=15)
plot_all_sondes(pressure, xlabel='pressure / hPa', xlim1 =1030, xlim2=600)
plot_all_sondes(pressure, xlabel='pressure / hPa', xlim1 =1030, xlim2=600)
plot_all_sondes(static_stab, xlabel='static stability / K/hPa', xlim1 =-0.3, xlim2=1)

#%% plot mean + n*sigma

# plot 1sigma
plt.rcParams.update({'font.size': 28})
plot_mean_sigma_all_days(mixing_ratio, nsigma = 1, xlabel='Mixing ratio / g/kg ', xlim1 =0, xlim2=20)
plot_mean_sigma_all_days(theta, nsigma = 1, xlabel='theta / K', xlim1 =295, xlim2=320)
plot_mean_sigma_all_days(theta_v,nsigma = 1, xlabel='theta_v / K', xlim1 =297, xlim2=320)
plot_mean_sigma_all_days(theta_e, nsigma = 1, xlabel='theta_e / K', xlim1 =305, xlim2=355)
plot_mean_sigma_all_days(RH, nsigma = 1, xlabel='RH / %', xlim1 =0, xlim2=100)
plot_mean_sigma_all_days(wspd, nsigma = 1, xlabel='wind speed / m/s', xlim1 =0, xlim2=15)
plot_mean_sigma_all_days(wdir, nsigma = 1, xlabel='wind direction / degrees ', xlim1 =0, xlim2=360)
plot_mean_sigma_all_days(u_wind, nsigma = 1, xlabel='zonal wind / m/s ', xlim1 =-15, xlim2=5)
plot_mean_sigma_all_days(v_wind, nsigma = 1, xlabel='meridional wind / m/s ', xlim1 =-15, xlim2=5)
plot_mean_sigma_all_days(static_stab, nsigma = 1, xlabel='static stability / K/hPa', xlim1 =-0.1, xlim2=0.2)


#%% 
# =============================================================================
#         calculate rolling means
# ============================================================================= 
# choose window
window=12

list_vars = ['mr', 'theta', 'theta_e', 'theta_v', 'rh', 'T', 'pres', 'wspd', 'rho', 'dp']
list_rm = []
for i, var in enumerate(list_vars):
    x = list_vars[i]+str('_rm')
    list_rm.append(x)
    vars()[x] = calc_rolling_mean(sondes, var, window=window)
    vars()[x] = vars()[x][6:-5,:]

#%% alternatively, rolling means by defining variables:

window=12

mr_rm = calc_rolling_mean(sondes, 'mr', window=window)[6:-5,:] # mixing ratio
theta_rm = calc_rolling_mean(sondes, 'theta', window=window)[6:-5,:]
theta_e_rm = calc_rolling_mean(sondes, 'theta_e', window=window)[6:-5,:]
theta_v_rm = calc_rolling_mean(sondes, 'theta_v', window=window)[6:-5,:]
rh_rm = calc_rolling_mean(sondes, 'rh', window=window)[6:-5,:]
T_rm = calc_rolling_mean(sondes, 'T', window=window)[6:-5,:]
pres_rm = calc_rolling_mean(sondes, 'pres', window=window)[6:-5,:]
wspd_rm = calc_rolling_mean(sondes, 'wspd', window=window)[6:-5,:]
rho_rm = calc_rolling_mean(sondes, 'rho', window=window)[6:-5,:]
dp_rm = calc_rolling_mean(sondes, 'dp', window=window)[6:-5,:] # dewpoint

# How many non-NaN values?
# print(sum(~np.isnan(mr_rm)))

#%% unit conversion for rolling mean values

T_rm_K = T_rm + 273.15 # degC to K
rh_rm_frac = (rh_rm / 100) # value (0-100) to percent
pres_rm_Pa = pres_rm * 100 # convert hPa to Pa

#%% plot rolling means

max_alt = 4000
#np.transpose(list_rm)
#list_rm = ['mr_rm', 'theta_rm', 'theta_e_rm', 'theta_v_rm', 'rh_rm', 'T_rm', 'pres_rm', 'wspd_rm', 'rho_rm']

plot_rolling_mean(mr_rm, xlim_min=0, xlim_max=20, ylim_max=max_alt, xlabel='Mixing ratio / g/kg ')
plot_rolling_mean(theta_rm, xlim_min=297, xlim_max=320, ylim_max=max_alt, xlabel='theta / K')
plot_rolling_mean(theta_v_rm, xlim_min=296, xlim_max=320, ylim_max=max_alt, xlabel='theta_v / K')
plot_rolling_mean(theta_e_rm, xlim_min=310, xlim_max=348, ylim_max=max_alt, xlabel='theta_e / K')
plot_rolling_mean(rh_rm, xlim_min=0, xlim_max=100, ylim_max=max_alt, xlabel= 'relative humidity / %')
plot_rolling_mean(T_rm, xlim_min=0, xlim_max=30, ylim_max=max_alt, xlabel='temperature / K')
plot_rolling_mean(pres_rm, xlim_min=1020, xlim_max=700, ylim_max=max_alt, xlabel='pressure / hPa')
plot_rolling_mean(wspd_rm, xlim_min=0, xlim_max=20, ylim_max=max_alt, xlabel='wind speed / m/s')

#%% calling and plotting LCL functions

def call_LCL_functions(T_rm_K, rh_rm_frac, pres_rm):
    
    """call multiple LCL functions:
    
        Bolton LCL height approximation
        - have to add initial parcel height
        
        Metpy LCL, iterative approach to finding P_LCL and T_LCL
    """
    
    len_time = len(pres_rm.launch_time)
    LCL_Bolton_vec_rm = np.zeros(len_time)
    for i in range(len_time):
        idx_z = 10
        # call function + add initial parcel height
        LCL_Bolton_vec_rm[i] = calc_LCL_Bolton(T_rm_K[i,idx_z] ,rh_rm_frac[i,idx_z]) \
                                + rh_rm_frac.alt[idx_z].values
    
    LCL_metpy_vec_rm = LCL_metpy(pres_rm, T_rm, dp_rm)

    # plot:
    if True:
        launch_time_vec = range(len_time)
        plt.figure(figsize=(12,8))
        plt.plot(launch_time_vec, LCL_Bolton_vec_rm, linewidth=4, color='blue', label="Bolton")
        plt.plot(launch_time_vec, LCL_metpy_vec_rm, linewidth=4, color='black', label="Metpy")
        plt.xlim([launch_time_vec[0], launch_time_vec[-1]])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        #plt.grid()
        plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
        plt.ylabel('LCL / m')
        plt.ylim([400,1100])
        plt.legend(loc='best')
        
        # difference looks like white noise, ~ 5m difference
#        plt.figure(figsize=(12,8))
#        plt.plot(LCL_Bolton_vec_rm - LCL_metpy_vec_rm)
        
    return LCL_Bolton_vec_rm, LCL_metpy_vec_rm


#%% call LCL function

LCL_Bolton_vec_rm, LCL_metpy_vec_rm = call_LCL_functions(T_rm_K, rh_rm_frac, pres_rm)


#plt.figure(figsize=(12,8))
#launch_time_vec = range(len(rh_rm_frac.launch_time))
#plt.plot(launch_time_vec, LCL_metpy_vec_rm, linewidth=4, color='blue', label="Metpy LCL")
#plt.plot(launch_time_vec, LCL_Bolton_vec_rm, linewidth=4, color='black', label="Bolton LCL")
#plt.plot(launch_time_vec, lcl_ht_Bjorn, linewidth=4, color='grey', label="Bjorn's LCL")
#plt.xlim([launch_time_vec[0], launch_time_vec[-1]])
#plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['top'].set_visible(False)
##plt.grid()
#plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
#plt.ylabel('LCL / m')
#plt.ylim([500,1200])
#plt.legend()

#%% Altitude of peak relative humidity 

def peak_RH(RH_rm):

    alt_peakRH = np.zeros(len(RH_rm))
    for i in range(len(RH_rm)):
        sonde_rh =  RH_rm.isel(launch_time=i)
        alt_peakRH[i] =   sonde_rh.where((sonde_rh == sonde_rh.max()),drop=True).alt.values
    
    alt_peakRH_trim = np.trim_zeros(alt_peakRH, trim='fb')
    alt_peakRH_trim[alt_peakRH_trim < 100]=np.nan
    
    # option to plot
    if 1:
        plt.figure(figsize=(12,8))
        plt.plot(range(len(RH_rm)), alt_peakRH_trim, linewidth=4, color='black')
        #plt.xlim([x1, x2])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
        plt.ylabel('Level of max. relative humidity / m')
        plt.ylim([300,800])
   
    return alt_peakRH
#%% call function
    
alt_peakRH = peak_RH(rh_rm_frac)

#%% hmix from rolling mean data
    
def calculateHmix_var_rm(rho_rm_sonde, var_rm_sonde, threshold):
    #def calc_plot_rolling_mean(var_name, window, xlim_min, xlim_max, ylim_max, xlabel):
    
    """ Code from Ludovic Touzé-Peiffer
    calculate height of well-mixed layer given a threshold of 
        for the veritcal difference from one spatial step to the next (dz) 
    """
    
    #Script to compute the height of the mixed layer in theta_v
    
    #size = len(sonde.alt)
    var = 0
    numer = 0
    denom = 0
    var_mix = 0

    k=8

# thetav - integral < threshold 
    while(abs(var) < threshold):

        #delta_z = sonde.alt[k+1]-sonde.alt[k]
        delta_z = 10

        # integral of rho*thetav dz/ int(rho)

        numer += 0.5*(rho_rm_sonde[k+1] * var_rm_sonde[k+1] + rho_rm_sonde[k] * var_rm_sonde[k])*delta_z
        denom += 0.5*(rho_rm_sonde[k+1] + rho_rm_sonde[k])*delta_z
        var_mix = numer/denom
        k += 1
        var = (var_rm_sonde[k] - var_mix).values

    hmix = var_rm_sonde.alt.values[k]
    #sonde["hmix"] = hmix
    # not adding this back
    
    return hmix
    #return sonde

def call_hmix_for_var(rho_rm, var_rm, thres_var):
    
    # theta
    hmix_var_rm = np.zeros(len(rho_rm.launch_time))
    
    for i in range(len(rho_rm.launch_time)): 
        rho_rm_sonde = rho_rm.isel(launch_time=i)
        var_rm_sonde = var_rm.isel(launch_time=i)
        hmix_var_rm[i] = calculateHmix_var_rm(rho_rm_sonde, var_rm_sonde, threshold=thres_var)
    
    plt.plot(hmix_var_rm)
    #print("range of estimated well-mixed layers for threshold=" + repr(thres_var) + " is", repr(np.ptp(hmix_var_rm)))
    
    return hmix_var_rm

#%% call vertical gradient function and plot results

# better to normalize profiles, dividing by std. dev. 
# and have one threshold for all profiles 
# but very high variability in hmix for normalized values (X - mean(X)) / std(X)
    
threshold_theta = 0.1
#theta_norm = (theta_rm - theta_rm.mean(dim="launch_time", skipna=True) )/ theta_rm.std(dim="launch_time", skipna=True)
hmix_theta_rm = call_hmix_for_var(rho_rm, theta_rm, threshold_theta)   

threshold_theta_v = 0.1
hmix_theta_v_rm = call_hmix_for_var(rho_rm, theta_v_rm, threshold_theta_v)   

threshold_theta_e = 1 #0.7, 1
hmix_theta_e_rm = call_hmix_for_var(rho_rm, theta_e_rm, threshold_theta_e)   

threshold_mr = 0.8
hmix_mr_rm = call_hmix_for_var(rho_rm, mr_rm, threshold_mr)   

# plot
if True:
    plt.figure(figsize=(18,8))
    sonde_no = range(len(rho_rm.launch_time))
    plt.plot(sonde_no, hmix_theta_rm, linewidth=4, color='black', linestyle='dashed', label = 'constant $\partial$$\Theta$/$\partial$z')
    plt.plot(sonde_no, hmix_theta_v_rm, linewidth=4, color='lightgrey', linestyle='dashed', label = 'constant $\partial$$\Theta$$_{v}$/$\partial$z')
    plt.plot(sonde_no, hmix_theta_e_rm, linewidth=4, color='lightblue', linestyle='dashed', label = 'constant $\partial$$\Theta$$_{e}$/$\partial$z')
    plt.plot(sonde_no, hmix_mr_rm, linewidth=4, color='cornflowerblue', linestyle='dashed', label = 'constant $\partial$q/$\partial$z')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(loc='upper left')
    plt.xlabel('12-sonde rolling mean')
    plt.ylabel('mixed layer height / m')
    plt.ylim([300,800])
    plt.xlim(sonde_no[1],sonde_no[-1])

#     sigma =  var.std(dim='launch_time')

#%% plot all rolling mean data together:
# RH peak, LCL estimates, well-mixed layer heights
    
soundings_rm = range(len(hmix_mr_rm))
plt.figure(figsize=(18,8))
soundings_rm = range(len(rho_rm.launch_time))

plt.plot(soundings_rm, alt_peakRH, linewidth=4, color='black', label='RH maximum')
plt.plot(soundings_rm, LCL_metpy_vec_rm, linewidth=4, color='grey', label='LCL metpy')
plt.plot(soundings_rm, LCL_Bolton_vec_rm, linewidth=4, color='lightgrey', label = 'LCL Bolton')
#plt.plot(soundings_rm, lcl_ht_Bjorn, linewidth=4, color='crimson', label = 'LCL Bjorn')
plt.plot(soundings_rm, hmix_theta_rm, linewidth=4, color='black', linestyle='dashed', label = 'constant $\partial$$\Theta$/$\partial$z')
plt.plot(soundings_rm, hmix_theta_v_rm, linewidth=4, color='lightgrey', linestyle='dashed', label = 'constant $\partial$$\Theta$$_{v}$/$\partial$z')
plt.plot(soundings_rm, hmix_theta_e_rm, linewidth=4, color='lightblue', linestyle='dashed', label = 'constant $\partial$$\Theta$$_{e}$/$\partial$z')
plt.plot(soundings_rm, hmix_mr_rm, linewidth=4, color='cornflowerblue', linestyle='dashed', label = 'constant $\partial$q/$\partial$z')
plt.xlim([soundings_rm[0], soundings_rm[-1]])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
#plt.legend(loc='best')
plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
plt.ylabel('mixed layer height / m')
plt.ylim([300,900])
plt.xlim(soundings_rm[1],soundings_rm[-1])


#%%
# look at correlations among these proxies

Data = {'RH': alt_peakRH,
        'LCL_Metpy': LCL_metpy_vec_rm,
        'LCL_Bolton': LCL_Bolton_vec_rm,
        'gradient_theta': hmix_theta_rm,
        'gradient_theta_v': hmix_theta_v_rm,
        'gradient_theta_e': hmix_theta_e_rm,
        'gradient_mr': hmix_mr_rm
        }

df = DataFrame(Data,columns=['RH','LCL_Metpy','LCL_Bolton', 'gradient_theta','gradient_theta_v', 'gradient_theta_e', 'gradient_mr'])

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 12))
plt.rcParams['font.size'] = 15
ax = sns.heatmap(corr, annot = True, mask=mask, cmap="coolwarm", vmax=1, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": .8},annot_kws={"size": 24})
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 24)
ax.set_yticklabels(ax.get_xmajorticklabels(), fontsize =24 )
plt.yticks(rotation = 0)
plt.xticks(rotation = 45)
ax.get_ylim()
ax.set_ylim(7, 0)

#%% go back to bigger font size

plt.rcParams.update({'font.size': 18})

#%% 
# =============================================================================
#         calculate inversion base height from static stability
# ============================================================================= 

#%% calculate static stability from rolling means

max_alt = 4000
static_stability_rm_vec = static_stability(T_rm_K, theta_rm, pres_rm) 
plot_rolling_mean(static_stability_rm_vec, xlim_min=0, xlim_max=0.25, ylim_max=max_alt, xlabel='static stability / K/hPa')

#%% find where stability first exceeds threshold = 0.1 K/hPa

def static_stability_thres(static_stability_rm_vec):

    alt_peak_stability = np.zeros(len(static_stability_rm_vec.launch_time))
    for i in range(len(static_stability_rm_vec.launch_time)):
        sonde_stability_da =  static_stability_rm_vec.isel(launch_time=i)
        sonde_stability =  static_stability_rm_vec.isel(launch_time=i).values
        idx_stability = np.argmax(sonde_stability>0.1)
        alt_peak_stability[i] =   sonde_stability_da[idx_stability].alt.values
    
    # option to plot
    if 1:
        plt.figure(figsize=(12,8))
        plt.plot(range(len(static_stability_rm_vec)), alt_peak_stability, linewidth=4, color='black')
        plt.xlim(range(len(static_stability_rm_vec))[0], range(len(static_stability_rm_vec))[-1])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
        plt.ylabel('Estimate for height of inversion base / m')
        plt.ylim([1800,2500])
   
    return alt_peak_stability

#%% call function 

alt_peak_stability = static_stability_thres(static_stability_rm_vec)

#%% 
# =============================================================================
#        RH and temperature vertical derivatives to calculate inversion height
# ============================================================================= 

"""
Method:
    Following Grindinger, 1992, found in paper Cao et al, Inversion Variability in the Hawaiian Trade Wind Regime
    calculate inversion neight as layer with positive temperature gradient and RH decrease with height
    inversion top is identified as level where temperature begins to decrease with height.
 
"""

#%%
def plot_gradient_vec(var_rm, xlim1, xlim2):
    gradient_vec = var_rm.differentiate(coord="alt")
    plt.figure(figsize=(6,8));
    for i in range(len(gradient_vec)):
            plt.plot(gradient_vec.isel(launch_time=i), gradient_vec.alt.values,color="lightgrey", linewidth=2,alpha=0.5)
    plt.plot(gradient_vec.mean(dim='launch_time'),gradient_vec.alt.values,linewidth=4, color='black')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel('vertical derivative (dz=10m)')
    plt.ylabel('Altitude / m')
    plt.ylim([0,4000])
    plt.axvline(x=0, color='black')
    plt.xlim([xlim1, xlim2])
    

def plot_gradient_vecs(var1, var2, xlabel1, xlabel2):
    fig, ax = plt.subplots(1,2,figsize=(20,8))
    
    gradient_vec1 = var1.differentiate(coord="alt")
    for i in range(len(gradient_vec1)):
        ax[0].plot(gradient_vec1.isel(launch_time=i), gradient_vec1.alt.values,color="lightgrey", linewidth=2,alpha=0.5)
    ax[0].plot(gradient_vec1.mean(dim='launch_time'),gradient_vec1.alt.values,linewidth=4, color='black')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel('Altitude / m')
    ax[0].set_ylim([0,4000])
    ax[0].axvline(x=0, color='black')
    ax[0].set_xlim([ -0.4, 0.4])
    
    gradient_vec2 = var2.differentiate(coord="alt")
    for i in range(len(gradient_vec2)):
        ax[1].plot(gradient_vec2.isel(launch_time=i), gradient_vec2.alt.values,color="lightgrey", linewidth=2,alpha=0.5)
    ax[1].plot(gradient_vec2.mean(dim='launch_time'),gradient_vec2.alt.values,linewidth=4, color='black')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_xlabel(xlabel2)
    #ax[1].set_ylabel('Altitude / m')
    ax[1].set_ylim([0,4000])
    ax[1].axvline(x=0, color='black')
    ax[1].set_xlim([-0.03, 0.03])
    
    return gradient_vec1, gradient_vec2

#%% call RH and temp gradient function

# individual plots
#plot_gradient_vec(rh_rm, xlim1=-1.5, xlim2=1)
#plot_gradient_vec(T_rm, xlim1= -0.1, xlim2=0.1)

# both gradients together
xlabel1 = '$\partial$RH/$\partial$z'
xlabel2 = '$\partial$T/$\partial$z (dz=10m)'  
gradient_rh, gradient_T = plot_gradient_vecs(rh_rm, T_rm, xlabel1, xlabel2)

#%% filter where dRH/dz <= 0 

gradient_rh_filt = gradient_rh.where(gradient_rh <= 0)
gradient_rh_filt.transpose().plot()
plt.ylim([0, 3000])

#%% filter where dT/dz >= 0 

gradient_T_filt = gradient_T.where(gradient_T >= 0, drop=True)
gradient_T_filt.transpose().plot()
# ds_mm.sst.mean(dim='lon').transpose().plot.contourf(levels=12, vmin=-2, vmax=30)

plt.ylim([0, 3000])

#%% inversion layer where dT/dz >= 0 and dRH/dz <= 0 

alt_inv = gradient_T.alt.where((gradient_rh <= 0) & (gradient_T >= 0))
alt_inv.plot()
plt.ylim([1500, 3000])

#%% find inversion base

def inversion_base_from_gradients(alt_inv):
    count=0
    inv_base_height = np.zeros(len(alt_inv.launch_time))
    for i in range(len(alt_inv.launch_time)):
        alt_inv_sonde_da =  alt_inv.isel(launch_time=i)
        alt_inv_sonde =  alt_inv.isel(launch_time=i).values
        idx_base = np.argmax(alt_inv_sonde>2000)
        inv_base_height[i] =   alt_inv_sonde_da[idx_base].alt.values
        count += 1
        print(count)
        print(inv_base_height[i])

# option to plot
    if True:
        plt.figure(figsize=(12,8))
        plt.plot(range(len(inv_base_height)), inv_base_height, linewidth=4, color='black')
        plt.xlim([range(len(inv_base_height))[0], range(len(inv_base_height))[-1]])
        #plt.xlim([x1, x2])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
        plt.ylabel('Estimate for height of inversion base / m')
        plt.ylim([1800,2500])
    
    return inv_base_height

inv_base_height_vec = inversion_base_from_gradients(alt_inv)
print(sum(~np.isnan(inv_base_height_vec)))
plt.plot(inv_base_height_vec)

#%% plot

plt.figure(figsize=(12,8))
plt.plot(range(len(static_stability_rm_vec)), alt_peak_stability, linewidth=4, color='grey', label ='static stability criterion')
plt.plot(range(len(inv_base_height_vec)), inv_base_height_vec, linewidth=4, color='midnightblue', label ='RH + temp criterion')
plt.xlim([range(len(inv_base_height_vec))[0], range(len(inv_base_height_vec))[-1]])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('Sounding circle / ' + str(window) + '-sonde rolling mean ')
plt.ylabel('Estimate for height of inversion base / m')
plt.ylim([1800,2500])
plt.legend(loc='best')

#%% 
# =============================================================================
#           MIXING LINE
# =============================================================================  

def plot_mixingLine_rm(mixing_ratio_vals, theta_e_vals):
    plt.figure(figsize=(7,7))
    for i in range(len(mr_rm)):
        # cut off at 3km = 700hPa
        max_alt = 3000
        sel_mr = mr_rm.where(mixing_ratio_vals["alt"]<max_alt, drop=True).isel(launch_time=i)
        sel_theta_e = theta_e_vals.where(theta_e_rm["alt"]<max_alt, drop=True).isel(launch_time=i)
        #sel_pres = pres_rm.where(pres_rm["alt"]<3000).isel(launch_time=i)
        plt.plot(sel_theta_e, sel_mr, color="lightgrey", linewidth=2,alpha=0.5)
        plt.gca().invert_yaxis()
    
    x = theta_e_rm.where(theta_e_rm["alt"]<max_alt, drop=True).mean(dim='launch_time')
    y = mr_rm.where(mr_rm["alt"]<max_alt, drop=True).mean(dim='launch_time')
    plt.plot(x, y, linewidth=4, color='black')
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylabel('Mixing ratio g/kg')
    plt.xlabel('$\Theta_e$ (K)')
    plt.xlim(310, 350)
    plt.ylim([0, 18])
    plt.gca().invert_yaxis()
    
    z = pres_rm[5,:].values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    z = z[~np.isnan(z)]
    idx1010 = (np.abs(z-1009)).argmin()
    idx900 = (np.abs(z-900)).argmin()
    idx750 = (np.abs(z-750)).argmin()
    xsub = [x[idx1010],x[idx900],x[idx750]]
    ysub = [y[idx1010],y[idx900],y[idx750]]
    zsub = [z[idx1010],z[idx900],z[idx750]]
    for a,b, c in zip(xsub, ysub, zsub): 
        plt.text(a, b, str(int(c)))


#%% 
# with rolling means  
plot_mixingLine_rm(mr_rm, theta_e_rm)

# without rolling means  
plot_mixingLine_rm(mixing_ratio, theta_e)



