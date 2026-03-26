#;+
#; :Author: Dr. Jing Wei (Email: weijing_rs@163.com)
#;-
import os
import netCDF4 as nc
import numpy as np  
from glob import glob
from osgeo import osr
from osgeo import gdal

#Define work and output paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WorkPath = os.path.join(BASE_DIR, "Data", "Code functioning example")
OutPath = os.path.join(BASE_DIR, "Output")


#Define air pollutant type 
#e.g., PM1, PM2.5, PM10, O3, NO2, SO2, and CO, et al.
AP = 'PM2.5'

#Define spatial resolution 
#e.g., 1 km ≈ 0.01 Degree
SP = 0.01 #Degrees

if not os.path.exists(OutPath):
    os.makedirs(OutPath)
path = glob(os.path.join(WorkPath, '*.nc'))

for file in path:
    f = nc.Dataset(file)   
    #Read SDS data
    data = np.array(f[AP][:]) 
    #Define missing value: NaN or -999
    data[data==65535] = np.nan #-999 
    #Read longitude and latitude information
    lon = np.array(f['lon'][:])
    lat = np.array(f['lat'][:])        
    LonMin,LatMax,LonMax,LatMin = lon.min(),lat.max(),lon.max(),lat.min()    
    N_Lat = len(lat) 
    N_Lon = len(lon)
    Lon_Res = SP #round((LonMax-LonMin)/(float(N_Lon)-1),2)
    Lat_Res = SP #round((LatMax-LatMin)/(float(N_Lat)-1),2)
    #Define Define output file
    fname = os.path.basename(file).split('.nc')[0]
    outfile = OutPath + '/{}.tif' .format(fname)        
    #Write GeoTIFF
    driver = gdal.GetDriverByName('GTiff')    
    outRaster = driver.Create(outfile,N_Lon,N_Lat,1,gdal.GDT_Float32)
    outRaster.SetGeoTransform([LonMin-Lon_Res/2,Lon_Res,0,LatMax+Lat_Res/2,0,-Lat_Res])
    sr = osr.SpatialReference()
    sr.SetWellKnownGeogCS('WGS84')
    outRaster.SetProjection(sr.ExportToWkt())
    outRaster.GetRasterBand(1).WriteArray(data)
    print(fname+'.tif',' Finished')     
    #release memory
    del outRaster
    f.close()