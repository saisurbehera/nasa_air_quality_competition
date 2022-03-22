import pandas as pd
from datetime import datetime
from osgeo import gdal
import numpy as np
import subprocess
import glob
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from dateutil.parser import parse
from multiprocessing import Pool
import tqdm
import glob

'''
Clean Data covert the data from raw HDF file to writeable format
'''





class Clean_Data():
    
    '''
    We declare all the global variables here
    '''
    def __init__(self,loc="train", num_processes=16):
        print("total files found " + len(glob.glob(loc+"/maiac/*/*.hdf")))
        all_files= glob.glob("test/maiac/*/*.hdf")
        self.loc = loc
        self.all_data = all_files
        self.train_labels = pd.read_csv("train_labels.csv")
        self.grid_metadata = pd.read_csv("grid_metadata.csv")
        self.satellite_metadata = pd.read_csv("pm25_satellite_metadata.csv")
        self.satellite_metadata['Date'] =  pd.to_datetime(self.satellite_metadata['time_end'], format='%Y-%m-%d')
        self.test_labels = pd.read_csv("submission_format.csv")
        self.train_labels["datetime_dt"]= pd.to_datetime(self.train_labels["datetime"])
    
    '''
    This function fetches the satellite metadata for a particular file. It gets all the columns
    '''
    def get_all_data_for_loci(self,ds,granule_id):
        each = {}
        metadata = ds.GetMetadata()
        for i in range(len(ds.GetSubDatasets())):
            raster = gdal.Open(ds.GetSubDatasets()[i][0]) #grid5km:cosSZA features only
            each_raster = raster.GetMetadata()

            long_name = each_raster["long_name"]


            all_rasters = []
        
            for j in range(int(raster.GetMetadata()["ADDITIONALLAYERS"])):
                try:
                    band = raster.GetRasterBand(j+1)
                    band_arr = band.ReadAsArray()
                    all_rasters.append(band_arr.tolist())
                except:
                    pass
            each[long_name] = all_rasters

        each_data_f = {'file':granule_id,'data':each}
        return each_data_f

    '''
    This function fetches the satellite metadata for a particular raster. 
    '''
    def get_all_data_for_loci_specific(self, ds,granule_id,parellel = False,i=0):
        each = {}
        s = ds.GetSubDatasets()
        raster = gdal.Open(s[i][0]) #grid5km:cosSZA features only
        each_raster = raster.GetMetadata()
        long_name = each_raster["long_name"]

        all_rasters = []
        
        for j in range(int(raster.GetMetadata()["ADDITIONALLAYERS"])):
            try:
                band = raster.GetRasterBand(j+1)
                band_arr = band.ReadAsArray()
                all_rasters.append(band_arr)
            except:
                pass
        each[long_name] = all_rasters



        each_data_f = {'file':granule_id,'data':each}
        del ds
        return each_data_f

    # Opens the HDF file
    def load_data(self,FILEPATH):
        ds = gdal.Open(FILEPATH)
        return ds

    # gets the formatted file path
    def format_file_path(self,granule_id):
        year = granule_id[:4]
        res = self.loc + '/maiac/'+year+'/'+granule_id
        return res

    # gets the satellite data for a particular granule
    def fetch_subset(self,granule_id,j=0):
        formatted = self.format_file_path(granule_id)
        ds = self.load_data( formatted)
        return self.get_all_data_for_loci_specific(ds,granule_id,i=j)

    # gets the satellite metadata for a particular granule
    def get_grid_data(self,metadata, grid_id):
        return metadata[metadata["grid_id"] == grid_id]

    # gets all the the training features for a particular grid
    def fetch_training_features(self,grid_id, datetime, split):
        temp = self.get_grid_data(self.grid_metadata, grid_id)
        sat_met = self.fetch_satellite_meta(self.satellite_metadata, 
                                   datetime, 
                                   temp.iloc[0]['location'], 
                                   "maiac", 
                                   split)
        counter = 0
        features = None
        for i in range(len(sat_met)):
            counter+=1
            granule_id = sat_met.iloc[i]['granule_id']

            subset = fetch_subset(granule_id)
            if features is None:
                features = subset
            else:
                features+=subset
        return features/counter
    
    '''
    This function gets us all the list of files which train labels mentionses
    '''
    def get_files(self):
        tasks = list(zip(train_labels["grid_id"],train_labels["datetime"]))

    # def get_models(self):
        


    def get_ds_loc(i):
        feature = fetch_training_features(i[0], i[1], split)
        return feature
    
    def get_data_all(self):
       from multiprocessing import get_context
       pool = Pool(processes=self.num_processes)
       one_important = list(dict_model_track.keys())
       args = [all_keys[i] for i in one_important]
       all_files_with_index = all_files_with_index = [ (i,j) for i in all_files for j in args]
       all_files_d_100 = []
       for x in tqdm.tqdm(pool.starmap(self.fetch_subset, all_files_with_index), total=len(all_files_with_index)):
           all_files_d_100.append(x)

    


if __name__ == '__main__':
    main()