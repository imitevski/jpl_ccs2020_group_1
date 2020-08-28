import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
#matplotlib.use('pdf')
from pylab import *
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import subprocess
from matplotlib.ticker import StrMethodFormatter, NullFormatter
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png', 'pdf')
import glob
import os
import scipy.stats
import pandas as pd
import seaborn as sns
SMALL_SIZE = 8
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  	 # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

ds = "/glade/work/im2527/jpl_ds/"

def weight(lat, lon, mask):
	'''
	calculates weights for (lat,lon) data with cos(lat) 
	'''
	if mask is not None:
		w = np.tile(np.cos(np.deg2rad(lat)), (lon.size, 1)).transpose()	
		w = xr.DataArray(w, [('lat',lat), ('lon',lon)]) * mask
		w = w / w.mean() 
		#print(w)
	if mask is None:
		w = np.tile(np.cos(np.deg2rad(lat)), (lon.size, 1)).transpose()	
		w = xr.DataArray(w / w.mean(), [('lat',lat), ('lon',lon)])
	return w

########## loading datasets ##########

ts = xr.open_dataset(ds + "ta_Amon_ECMWF_interim_197901-201512.nc").ta.sel(time=slice('1980','2013'),plev=1e5).groupby('time.year').mean('time')
sst = xr.open_dataset(ds + 'tos_Amon_ECMWF_interim_197901-201402.nc').tos.sel(time=slice('1980','2013')).groupby('time.year').mean('time') 
lat = sst.lat
lon = sst.lon

########## loading masks ##########

arctic = xr.open_dataset(ds + "masks_ecmwf/arctic.nc").DATA01_A[0,0,:,:]*0 + 1
#arctic.coords['LON1_481'] = arctic.LON1_481 - 180
arctic = arctic.interp(LON1_481 = lon, LAT = lat)
arctic = xr.DataArray(arctic, dims = ["lat","lon"], coords = [lat, lon])

atlantic = xr.open_dataset(ds + "masks_ecmwf/atlantic.nc").DATA01_A[0,0,:,:]*0 + 1
#atlantic.coords['LON1_481'] = atlantic.LON1_481 - 180
atlantic = atlantic.interp(LON1_481 = lon, LAT = lat)
atlantic = xr.DataArray(atlantic, dims = ["lat","lon"], coords = [lat, lon])

indian = xr.open_dataset(ds + "masks_ecmwf/indian.nc").DATA01_A[0,0,:,:]*0 + 1
#indian.coords['LON1_481'] = indian.LON1_481 - 180
indian = indian.interp(LON1_481 = lon, LAT = lat)
indian = xr.DataArray(indian, dims = ["lat","lon"], coords = [lat, lon])

pacific = xr.open_dataset(ds + "masks_ecmwf/pacific.nc").DATA01_A[0,0,:,:]*0 + 1
#pacific.coords['LON1_481'] = pacific.LON1_481 - 180
pacific = pacific.interp(LON1_481 = lon, LAT = lat)
pacific = xr.DataArray(pacific, dims = ["lat","lon"], coords = [lat, lon])

southern_ocean = xr.open_dataset(ds + "masks_ecmwf/southern_ocean.nc").DATA01_A[0,0,:,:]*0 + 1
#southern_ocean.coords['LON1_481'] = southern_ocean.LON1_481 - 180
southern_ocean = southern_ocean.interp(LON1_481 = lon, LAT = lat)
southern_ocean = xr.DataArray(southern_ocean, dims = ["lat","lon"], coords = [lat, lon])

########## SSTs per basin ##########

def atl(x):
	if x is 'sst':
		y = np.array( (sst * weight(lat, lon, atlantic)).mean(dim=['lat','lon']) )
	if x is 'ts':
		y = np.array( (ts * weight(lat, lon, atlantic)).mean(dim=['lat','lon']) )
	return y - y[0]
def arct(x):
	if x is 'sst':
		y = np.array( (sst * weight(lat, lon, arctic)).mean(dim=['lat','lon']) )
	if x is 'ts':
		y = np.array( (ts * weight(lat, lon, arctic)).mean(dim=['lat','lon']) )
	return y - y[0]
def ind(x):
	if x is 'sst':
		y = np.array( (sst * weight(lat, lon, indian)).mean(dim=['lat','lon']) )
	if x is 'ts':
		y = np.array( (ts * weight(lat, lon, indian)).mean(dim=['lat','lon']) )
	return y - y[0]
def pac(x):
	if x is 'sst':
		y = np.array( (sst * weight(lat, lon, pacific)).mean(dim=['lat','lon']) )
	if x is 'ts':
		y = np.array( (ts * weight(lat, lon, pacific)).mean(dim=['lat','lon']) )
	return y - y[0]
def s_ocn(x):
	if x is 'sst':
		y = np.array( (sst * weight(lat, lon, southern_ocean)).mean(dim=['lat','lon']) )
	if x is 'ts':
		y = np.array( (ts * weight(lat, lon, southern_ocean)).mean(dim=['lat','lon']) )
	return y - y[0]


########## Figure (4x4), for 10-2000m and all basins ##########

def q5_im_1():
 	
	SST = (sst*weight(lat,lon,None)).mean(dim=['lat','lon'])
	Ts = (ts*weight(lat,lon,None)).mean(dim=['lat','lon'])
		
	fig = plt.figure()
	fig.set_figwidth(fig.get_figwidth() * 2)
	fig.set_figheight(fig.get_figheight() * 1) 
		
	axes = fig.add_subplot(1,1,1)
	plot(np.arange(1980,2014), Ts - Ts[0], color = 'olive', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'GMSAT')
	plot(np.arange(1980,2014), SST - SST[0], color = 'purple', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'Global SST')
	plot(np.arange(1980,2014), atl('sst'), color = 'green', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'SST, Atlantic')#+sl_atl(10)+'K/yr')
	plot(np.arange(1980,2014), arct('sst'), color = 'blue', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'SST, Arctic')#+sl_arct(10)+'K/yr')
	plot(np.arange(1980,2014), ind('sst'), color = 'red', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'SST, Indian')#+sl_ind(10)+'K/yr')
	plot(np.arange(1980,2014), pac('sst'), color = 'cyan', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'SST, Pacific')#+sl_pac(10)+'K/yr')
	plot(np.arange(1980,2014), s_ocn('sst'), color = 'black', linestyle ='-',
		markersize=10,linewidth=4,marker='o', label = 'SST, Southern Ocean')#+sl_s_ocn(10)+'K/yr')
	plt.title('')
	plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
	#plt.xlim([2000.8,2012.2])
	#plt.ylim([-0.1,0.67])
	plt.xlabel('year')
	plt.ylabel(r"$\Delta T$ [K]")
	
	plt.tight_layout()
	plt.savefig('q5_im_1.pdf')
	plt.show()

def q5_im_2():
	
	SST = (sst*weight(lat,lon,None)).mean(dim=['lat','lon'])
	Ts = np.array((ts*weight(lat,lon,None)).mean(dim=['lat','lon']))
		
		
	corr_matrix = np.zeros((7,7))	
	########
	corr_matrix[0,0] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(Ts - Ts[0]))
	corr_matrix[0,1] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(SST - SST[0]))
	corr_matrix[0,2] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(atl('sst')))
	corr_matrix[0,3] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(arct('sst')))
	corr_matrix[0,4] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(ind('sst')))
	corr_matrix[0,5] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(pac('sst')))
	corr_matrix[0,6] = pd.Series.corr(pd.Series(Ts - Ts[0]),
					   pd.Series(s_ocn('sst')))
	########	
	corr_matrix[1,0] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(Ts - Ts[0]))	
	corr_matrix[1,1] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(SST - SST[0]))
	corr_matrix[1,2] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(atl('sst')))
	corr_matrix[1,3] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(arct('sst')))
	corr_matrix[1,4] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(ind('sst')))
	corr_matrix[1,5] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(pac('sst')))
	corr_matrix[1,6] = pd.Series.corr(pd.Series(SST - SST[0]),
					   pd.Series(s_ocn('sst')))
	########
	corr_matrix[2,0] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(Ts - Ts[0]))	
	corr_matrix[2,1] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(SST - SST[0]))
	corr_matrix[2,2] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(atl('sst')))
	corr_matrix[2,3] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(arct('sst')))
	corr_matrix[2,4] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(ind('sst')))
	corr_matrix[2,5] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(pac('sst')))
	corr_matrix[2,6] = pd.Series.corr(pd.Series(atl('sst')),
					   pd.Series(s_ocn('sst')))
	########
	corr_matrix[3,0] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(Ts - Ts[0]))	
	corr_matrix[3,1] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(SST - SST[0]))
	corr_matrix[3,2] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(atl('sst')))
	corr_matrix[3,3] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(arct('sst')))
	corr_matrix[3,4] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(ind('sst')))
	corr_matrix[3,5] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(pac('sst')))
	corr_matrix[3,6] = pd.Series.corr(pd.Series(arct('sst')),
					   pd.Series(s_ocn('sst')))
	########
	corr_matrix[4,0] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(Ts - Ts[0]))	
	corr_matrix[4,1] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(SST - SST[0]))
	corr_matrix[4,2] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(atl('sst')))
	corr_matrix[4,3] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(arct('sst')))
	corr_matrix[4,4] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(ind('sst')))
	corr_matrix[4,5] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(pac('sst')))
	corr_matrix[4,6] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(s_ocn('sst')))
	
	corr_matrix[5,0] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(Ts - Ts[0]))	
	corr_matrix[5,1] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(SST - SST[0]))
	corr_matrix[5,2] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(atl('sst')))
	corr_matrix[5,3] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(arct('sst')))
	corr_matrix[5,4] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(ind('sst')))
	corr_matrix[5,5] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(pac('sst')))
	corr_matrix[5,6] = pd.Series.corr(pd.Series(ind('sst')),
					   pd.Series(s_ocn('sst')))
	
	corr_matrix[5,0] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(Ts - Ts[0]))
	corr_matrix[5,1] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(SST - SST[0]))
	corr_matrix[5,2] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(atl('sst')))
	corr_matrix[5,3] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(arct('sst')))
	corr_matrix[5,4] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(ind('sst')))
	corr_matrix[5,5] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(pac('sst')))
	corr_matrix[5,6] = pd.Series.corr(pd.Series(pac('sst')),
					   pd.Series(s_ocn('sst')))

	corr_matrix[6,0] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(Ts - Ts[0]))	
	corr_matrix[6,1] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(SST - SST[0]))
	corr_matrix[6,2] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(atl('sst')))
	corr_matrix[6,3] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(arct('sst')))
	corr_matrix[6,4] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(ind('sst')))
	corr_matrix[6,5] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(pac('sst')))
	corr_matrix[6,6] = pd.Series.corr(pd.Series(s_ocn('sst')),
					   pd.Series(s_ocn('sst')))
	
	names = ['GMSAT','SST','Atl.','Arctic','Indian','Pacific','S. Ocean']
	df = pd.DataFrame(corr_matrix, columns=names)
	
	#print(corr_matrix)

	fig = plt.figure(figsize=(6,5))
	ax = fig.add_subplot(1,1,1)
	#ax = plt.axes()
	sns.heatmap(df, annot=True, yticklabels = names, vmin = -1, vmax = 1, \
                annot_kws={"size": 10}, cmap = "RdYlBu_r", ax = ax) # "PiYG", "RdBu", "RdY1Bu", "Spectral" _r is for reverse 
	title = 'Annual correlation, 1980-2013'
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)
	ax.set_title(title,fontsize = 12)
	
	fig.tight_layout()	
	plt.savefig('q5_im_2.pdf')
	plt.show()


q5_im_1()
#q5_im_2()


#print(pd.Series(pac('sst')))
#Ts = np.array( (ts*weight(lat,lon,None)).mean(dim=['lat','lon']) )
#print(Ts - Ts[0])
#print(pac('sst'))
#print( np.corrcoef(Ts - Ts[0], pac('sst'))[1][0] )

#print(  pd.Series.corr( pd.Series(pac('sst')), pd.Series(Ts - Ts[0]))  )

