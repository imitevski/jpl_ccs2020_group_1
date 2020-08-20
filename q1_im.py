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
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  	 # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def weight(lat, lon):
	'''
	calculates weights for (lat,lon) data with cos(lat) 
	'''
	if lon is not None:
		weight = np.tile(np.cos(np.deg2rad(lat)), (lon.size, 1)).transpose()
		weight = xr.DataArray(weight / weight.mean(), [('lat',lat), ('lon',lon)])
	if lon is None:
		weight = np.cos(np.deg2rad(lat))
		weight = xr.DataArray(weight / weight.mean(), [('lat',lat)])
	return weight

rlut = xr.open_dataset('nasa_ceres_rlut_200003_201206_0f360f2.5_-90f90f2_-999999.nc').rlut
rsdt = xr.open_dataset('nasa_ceres_rsdt_200003_201206_0f360f2.5_-90f90f2_-999999.nc').rsdt
rsut = xr.open_dataset('nasa_ceres_rsut_200003_201206_0f360f2.5_-90f90f2_-999999.nc').rsut

def q1_im_all(): 
	fig = plt.figure()
	fig.set_figwidth(fig.get_figwidth() * 1)
	fig.set_figheight(fig.get_figheight() * 1) 
		
	axes = fig.add_subplot(1,1,1)
	plt.plot(rlut.time, (rlut*weight(rlut.lat, rlut.lon)).mean(dim=['lat','lon']), linewidth = 3, color = 'red', label='rlut (TOA out. LW)')
	plt.plot(rlut.time, (rsdt*weight(rlut.lat, rlut.lon)).mean(dim=['lat','lon']), linewidth = 3, color = 'blue', label='rsdt (TOA inc. SW)')
	plt.plot(rlut.time, (rsut*weight(rlut.lat, rlut.lon)).mean(dim=['lat','lon']), linewidth = 3, color = 'green', label='rsut (TOA out. SW)')
	plt.ylabel('W/m2')
	plt.xlabel('year')
	plt.title("NASA CERES")
	plt.legend(loc = 0)

	plt.tight_layout()
	plt.savefig('q1_im_all.pdf')
	plt.show()

def q1_im_net(): 
	fig = plt.figure()
	fig.set_figwidth(fig.get_figwidth() * 2)
	fig.set_figheight(fig.get_figheight() * 1) 
		
	r2 = (rlut*weight(rlut.lat, rlut.lon)).mean(dim=['lat','lon'])
	r1 = (rsdt*weight(rsdt.lat, rsdt.lon)).mean(dim=['lat','lon'])
	r3 = (rsut*weight(rsut.lat, rsut.lon)).mean(dim=['lat','lon'])
	
	r_year = (r1 - r2 - r3).sel(time=slice('2001','2011')).groupby('time.year').mean('time')
		
	axes = fig.add_subplot(1,2,1)
	plt.plot(rlut.time, r1 - r2 - r3, linewidth = 3, color = 'red', label='net radiation (+ downward)')
	plt.ylabel('W/m2')
	plt.xlabel('year')
	plt.title("a) monthly")
	plt.legend(loc = 0)
	
	axes = fig.add_subplot(1,2,2)
	plt.plot(r_year.time, r_year, linewidth = 3, color = 'red', label='net radiation (+ downward)')
	axes.axhline(y = r_year.mean(), color = 'red', linestyle='dotted', label=str(np.round(np.array(r_year.mean()), 2))+' W/m2, 2001-2011 mean')
	plt.ylabel('W/m2')
	plt.xlabel('year')
	plt.title("b) yearly")
	plt.legend(loc = 0)

	plt.tight_layout()
	plt.savefig('q1_im_net.pdf')
	plt.show()

#q1_im_all()
q1_im_net()
