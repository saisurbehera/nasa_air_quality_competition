
import pandas as pd
from datetime import datetime
from osgeo import gdal
import numpy as np
import subprocess
import glob
from dateutil.parser import parse
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

## Get The labels

# def get_data(path=""):
path = "../starter_code/"
train_labels = pd.read_csv(path+"train_labels.csv")
grid_metadata = pd.read_csv(path+"grid_metadata.csv")
satellite_metadata = pd.read_csv(path+"satellite_metadata.csv")
satellite_metadata['Date'] =  pd.to_datetime(satellite_metadata['time_end'], format='%Y-%m-%d')

def get_proper_label(text):
    val = "s3://drivendata-competition-airathon-public-us/pm25/train/maiac/"
    return val + text[:4] + "/" +text

def download_loc(text):
    partial = get_proper_label(text)
    return "aws s3 cp " + partial + " ./dataset/" + " --no-sign-request"

def get_grid_data(metadata, grid_id):
    return metadata[metadata["grid_id"] == grid_id]

def fetch_satellite_meta(metadata, datetime, location, datatype, split):
    if location == "Delhi":
        location = "dl"
    elif location == "Taipei":
        location = "tpe"
    else:
        location = "la"
    metadata = metadata[metadata['location'] == location]
    metadata = metadata[metadata['product'] == datatype]
    metadata = metadata[metadata['split'] == split]
    dateobject = parse(datetime)
    return metadata.loc[(metadata['Date'].dt.month == dateobject.month) & 
                        (metadata['Date'].dt.day == dateobject.day) &
                        (metadata['Date'].dt.year <= dateobject.year)]

# Opens the HDF file
def load_data(FILEPATH):
    ds = gdal.Open(FILEPATH)
    return ds

download_files = []

def fetch_subset(granule_id):
    
    result = get_proper_label(granule_id)
    already_files = [ i.split("/")[1] for i in glob.glob("dataset/*")]
    
    if (granule_id not in already_files):
        download_files.append(granule_id)
        print("Need to download: "+granule_id)
        already_files+= granule_id
        # subprocess.run(["aws", "s3", "cp", result, "./dataset", "--no-sign-request"])
        return None
    ds = load_data("dataset/" + granule_id)
    ds.GetSubDatasets()[0]
    raster = gdal.Open(ds.GetSubDatasets()[8][0]) #grid5km:cosSZA features only
    band = raster.GetRasterBand(1)
    band_arr = band.ReadAsArray()
    return band_arr

def fetch_training_features(grid_id, datetime, split):
    temp = get_grid_data(grid_metadata, grid_id)
    sat_met = fetch_satellite_meta(satellite_metadata, 
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
        if subset!=None:
            print(features)
            if features is None:
                features = subset
            else:
                features+=subset
    if(features!=None):
        return features/counter
    return None

def generate_features(train_labels, split):
    labels = []
    features = []
    for i in range(len(train_labels)):
        feature = fetch_training_features(train_labels.iloc[i]['grid_id'], train_labels.iloc[i]['datetime'], split)
        features.append(np.array(feature).reshape(-1))
        if split == "train":
            labels.append(train_labels.iloc[i]['value'])
    return np.array(features), np.array(labels)

features, labels = generate_features(train_labels, "train")
# print(download_files)
all_files_to_download = [download_loc(i) for i in download_files]
with open('download_files.txt', 'w') as f:
    for line in all_files_to_download:
        f.write(line)
        f.write('\n')