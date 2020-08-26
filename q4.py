import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
#matplotlib.use('pdf')
from pylab import *
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
import subprocess
from matplotlib.ticker import StrMethodFormatter, NullFormatter
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png', 'pdf')
import glob
import os
import scipy.stats
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  	 # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

ds = "/glade/work/im2527/jpl_ds/"

def weight(lat, lon, mask):
	'''
	calculates weights for (lat,lon) data with cos(lat) 
	'''
	if lon is not None:
		w = np.tile(np.cos(np.deg2rad(lat)), (lon.size, 1)).transpose()	
		w = xr.DataArray(w, [('lat',lat), ('lon',lon)]) * mask
		w = w / w.mean() 
		#print(w)
	if lon is None:
		return("lon cannot be None")
	return w

########## loading datasets ##########

amsre = xr.open_dataset(ds + "tos_AMSRE_L3_v7_200206-201012.nc").tos.sel(time=slice('2002','2010'))#.groupby('time.year').mean('time') # SST
argo = xr.open_dataset(ds + 'ot_ARGO_200101-201305.nc').ot.sel(time=slice('2001','2012'))#.groupby('time.year').mean('time') # Ocean Temperature
lat = argo.lat
lon = argo.lon

########## loading masks ##########

arctic = xr.open_dataset(ds + "masks/arctic.nc").DATA01_A[0,0,:,:]*0 + 1
arctic = xr.DataArray(arctic, dims = ["lat","lon"], coords = [arctic.LAT, arctic.LON])

atlantic = xr.open_dataset(ds + "masks/atlantic.nc").DATA01_A[0,0,:,:]*0 + 1
atlantic = xr.DataArray(atlantic, dims = ["lat","lon"], coords = [atlantic.LAT, atlantic.LON])

indian = xr.open_dataset(ds + "masks/indian.nc").DATA01_A[0,0,:,:]*0 + 1
indian = xr.DataArray(indian, dims = ["lat","lon"], coords = [indian.LAT, indian.LON])

pacific = xr.open_dataset(ds + "masks/pacific.nc").DATA01_A[0,0,:,:]*0 + 1
pacific = xr.DataArray(pacific, dims = ["lat","lon"], coords = [pacific.LAT, pacific.LON])

southern_ocean = xr.open_dataset(ds + "masks/southern_ocean.nc").DATA01_A[0,0,:,:]*0 + 1
southern_ocean = xr.DataArray(southern_ocean, dims = ["lat","lon"], coords = [southern_ocean.LAT, southern_ocean.LON])

########## Basin Averages ##########

def atl(plev):
	y = np.array( (argo.sel(plev=plev, method = 'nearest').groupby('time.year').mean('time') * weight(lat, lon, atlantic)).mean(dim=['lat','lon']) )
	return y - y[0]
def arct(plev):
	y = np.array( (argo.sel(plev=plev, method = 'nearest').groupby('time.year').mean('time') * weight(lat, lon, arctic)).mean(dim=['lat','lon']) )
	return y - y[0]
def ind(plev):
	y = np.array( (argo.sel(plev=plev, method = 'nearest').groupby('time.year').mean('time') * weight(lat, lon, indian)).mean(dim=['lat','lon']) )
	return y - y[0]
def pac(plev):
	y = np.array( (argo.sel(plev=plev, method = 'nearest').groupby('time.year').mean('time') * weight(lat, lon, pacific)).mean(dim=['lat','lon']) )
	return y - y[0]
def s_ocn(plev):
	y = np.array( (argo.sel(plev=plev, method = 'nearest').groupby('time.year').mean('time') * weight(lat, lon, southern_ocean)).mean(dim=['lat','lon']) )
	return y - y[0]

########### SLOPE ############

def sl_atl(plev):
	y = scipy.stats.linregress(np.arange(2001,2013), atl(plev)).slope
	return np.format_float_scientific(y,precision=2)
def sl_arct(plev):
	y = scipy.stats.linregress(np.arange(2001,2013), arct(plev)).slope
	return np.format_float_scientific(y,precision=2)
def sl_ind(plev):
	y = scipy.stats.linregress(np.arange(2001,2013), ind(plev)).slope
	return np.format_float_scientific(y,precision=2)
def sl_pac(plev):
	y = scipy.stats.linregress(np.arange(2001,2013), pac(plev)).slope
	return np.format_float_scientific(y,precision=2)
def sl_s_ocn(plev):
	y = scipy.stats.linregress(np.arange(2001,2013), s_ocn(plev)).slope
	return np.format_float_scientific(y,precision=2)

########## Figure (4x4), for 10-2000m and all basins ##########

def q4_im_all():
 
	fig = plt.figure()
	fig.set_figwidth(fig.get_figwidth() * 2)
	fig.set_figheight(fig.get_figheight() * 2) 
		
	axes = fig.add_subplot(2,2,1)
	plot(np.arange(2001,2013), atl(10), color = 'green', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Atlantic,'+sl_atl(10)+'K/yr')
	plot(np.arange(2001,2013), arct(10), color = 'blue', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Arctic,'+sl_arct(10)+'K/yr')
	plot(np.arange(2001,2013), ind(10), color = 'red', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Indian,'+sl_ind(10)+'K/yr')
	plot(np.arange(2001,2013), pac(10), color = 'cyan', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Pacific,'+sl_pac(10)+'K/yr')
	plot(np.arange(2001,2013), s_ocn(10), color = 'black', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Southern Ocean,'+sl_s_ocn(10)+'K/yr')
	plt.title('10 m')
	plt.legend(loc=0)
	plt.xlim([2000.8,2012.2])
	plt.ylim([-0.1,0.67])
	plt.xlabel(['year'])
	plt.ylabel(r"$\Delta T$ [K]")
	
	axes = fig.add_subplot(2,2,2)
	plot(np.arange(2001,2013), atl(100), color = 'green', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Atlantic,'+sl_atl(100)+'K/yr')
	plot(np.arange(2001,2013), arct(100), color = 'blue', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Arctic,'+sl_arct(100)+'K/yr')
	plot(np.arange(2001,2013), ind(100), color = 'red', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Indian,'+sl_ind(100)+'K/yr')
	plot(np.arange(2001,2013), pac(100), color = 'cyan', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Pacific,'+sl_pac(100)+'K/yr')
	plot(np.arange(2001,2013), s_ocn(100), color = 'black', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Southern Ocean,'+sl_s_ocn(100)+'K/yr')
	plt.title('100 m')
	plt.legend(loc=0)
	plt.xlim([2000.8,2012.2])
	plt.ylim([-0.1,0.67])
	plt.xlabel(['year'])
	plt.ylabel(r"$\Delta T$ [K]")
	
	axes = fig.add_subplot(2,2,3)
	plot(np.arange(2001,2013), atl(700), color = 'green', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Atlantic,'+sl_atl(700)+'K/yr')
	plot(np.arange(2001,2013), arct(700), color = 'blue', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Arctic,'+sl_arct(700)+'K/yr')
	plot(np.arange(2001,2013), ind(700), color = 'red', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Indian,'+sl_ind(700)+'K/yr')
	plot(np.arange(2001,2013), pac(700), color = 'cyan', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Pacific,'+sl_pac(700)+'K/yr')
	plot(np.arange(2001,2013), s_ocn(700), color = 'black', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Southern Ocean,'+sl_s_ocn(700)+'K/yr')
	plt.title('700 m')
	plt.legend(loc=0)
	plt.xlim([2000.8,2012.2])
	plt.ylim([-0.1,0.67])
	plt.xlabel(['year'])
	plt.ylabel(r"$\Delta T$ [K]")
	
	axes = fig.add_subplot(2,2,4)
	plot(np.arange(2001,2013), atl(2000), color = 'green', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Atlantic,'+sl_atl(2000)+'K/yr')
	plot(np.arange(2001,2013), arct(2000), color = 'blue', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Arctic,'+sl_arct(2000)+'K/yr')
	plot(np.arange(2001,2013), ind(2000), color = 'red', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Indian,'+sl_ind(2000)+'K/yr')
	plot(np.arange(2001,2013), pac(2000), color = 'cyan', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Pacific,'+sl_pac(2000)+'K/yr')
	plot(np.arange(2001,2013), s_ocn(2000), color = 'black', linestyle ='-',markersize=10,linewidth=4,marker='o', label = 'Southern Ocean,'+sl_s_ocn(2000)+'K/yr')
	plt.title('2000 m')
	plt.legend(loc=0)
	plt.xlim([2000.8,2012.2])
	plt.ylim([-0.1,0.67])
	plt.xlabel(['year'])
	plt.ylabel(r"$\Delta T$ [K]")
	
	plt.tight_layout()
	plt.savefig('q4_im_all.pdf')
	plt.show()

q4_im_all()


