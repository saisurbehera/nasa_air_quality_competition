import os
import re
import random
import pickle

from pathlib import Path
os.chdir('/content/drive/MyDrive/datadriven/airathon')
DATA_PATH = Path.cwd() / 'data'
RAW = DATA_PATH / 'raw'
PROCESSED = DATA_PATH / 'processed'

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from osgeo import gdal
import geopandas as gpd
from pyhdf.SD import SD, SDC, SDS
from typing import Dict, List, Union

import pyproj
from pyproj import CRS, Proj
from pqdm.processes import pqdm

import multiprocessing
n_cpus = multiprocessing.cpu_count()

mpl.rcParams['figure.dpi'] = 100

# setup colab before using utils.py
# !pip install -q condacolab -q
# import condacolab
# condacolab.install()
# !conda install geopandas
# !pip install awscli
# !pip install cloudpathlib
# !pip install geopandas
# !pip install rasterio
# !pip install pyhdf
# !pip install cloudpathlib[s3]
# !pip install rtree
# !pip install pqdm



def get_hdf(url):
    fpth = str(RAW / '/'.join(url.split('/')[-4:]))
    hdf = SD(fpth, SDC.READ)
    return hdf


def get_align_dict(hdf):
    raw_attributes = hdf.attributes()["StructMetadata.0"]

    g = raw_attributes.split("END_GROUP=GRID_1")[0]
    meta = dict([x.split("=") for x in g.split() if "=" in x])
    for key, val in meta.items():
        try:
            meta[key] = eval(val)
        except:
            pass

    alignment_dict = {
        "upper_left": meta["UpperLeftPointMtrs"],
        "lower_right": meta["LowerRightMtrs"],
        "crs": meta["Projection"],
        "crs_params": meta["ProjParams"]}

    return alignment_dict


def calibrate(dataset, shape, calibration_dict):
    corrected = np.ma.empty(shape, dtype=np.double)

    for orbit in range(shape[0]):
        data = dataset[orbit, :, :].astype(np.double)
        invalid_condition = (
            (data < calibration_dict["valid_range"][0]) |
            (data > calibration_dict["valid_range"][1]) |
            (data == calibration_dict["_FillValue"])
        )
        data[invalid_condition] = np.nan
        data = (
            (data - calibration_dict["add_offset"]) *
            calibration_dict["scale_factor"]
        )
        data = np.ma.masked_array(data, np.isnan(data))
        corrected[orbit, : :] = data

    corrected.fill_value = np.nan
    return corrected


def meshgrid(alignment_dict, shape):
    x0, y0 = alignment_dict["upper_left"]
    x1, y1 = alignment_dict["lower_right"]
    
    x = np.linspace(x0, x1, shape[2], endpoint=True)
    y = np.linspace(y0, y1, shape[1], endpoint=True)
    
    xv, yv = np.meshgrid(x, y)
    return xv, yv


def transformArrays(xv, yv, crs_from, crs_to):
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    lon, lat = transformer.transform(xv, yv)
    return lon, lat


def arraysToDF(dataDict, lat, lon, id, crs, n_orbits, total_bounds=None):
    lats = lat.ravel()
    lons = lon.ravel()
    size = lats.size

    values = {"lat": np.tile(lats, n_orbits),
              "lon": np.tile(lons, n_orbits),
              "orbit": np.arange(n_orbits).repeat(size),
              "granule_id": [id] * size * n_orbits}

    datasets = list(dataDict.keys())
    for dataset_name in datasets:
        values[dataset_name] = np.concatenate([d.data.ravel() for d in dataDict[dataset_name]])

    
    df = pd.DataFrame(values).dropna()
    if total_bounds is not None:
        x_min, y_min, x_max, y_max = total_bounds
        df = df[df.lon.between(x_min, x_max) & df.lat.between(y_min, y_max)]
    
    gDF = gpd.GeoDataFrame(df)
    gDF["geometry"] = gpd.points_from_xy(gDF.lon, gDF.lat)
    gDF.crs = crs

    cols = ['granule_id', 'orbit', 'geometry'] + datasets
    return gDF[cols].reset_index(drop=True)


def preprocess(url, gridCellDF, datasets, total_bounds=None):
    fpth = str(RAW / '/'.join(url.split('/')[-4:]))
    hdf = SD(fpth, SDC.READ)
    alignment_dict = get_align_dict(hdf)
    td = hdf.select('Optical_Depth_047')
    shape = td.info()[2]

    dataDict = {}
    
    for dataset_name in datasets:
        data = hdf.select(dataset_name)
        calibration_dict = data.attributes()
        dataDict[dataset_name] = calibrate(dataset=data, shape=shape, calibration_dict=calibration_dict)
        
    xv, yv = meshgrid(alignment_dict=alignment_dict, shape=shape)
    lon, lat = transformArrays(xv, yv, sinu_crs, wgs84_crs)
    
    gDF = arraysToDF(
        dataDict=dataDict,
        lat=lat, lon=lon, id=url.split('/')[-1],
        crs=wgs84_crs,
        n_orbits = shape[0],
        total_bounds=gridCellDF.total_bounds)
    
    innerDF = gpd.sjoin(gridCellDF, gDF, how="inner")
    hdf.end()  
    return innerDF.drop(columns="index_right").reset_index()


def parallel_process(granulePaths, gridCellDF, datasets, n_jobs=n_cpus//2):
    args = ((gp, gridCellDF, datasets) for gp in granulePaths)
    output = pqdm(args, preprocess, n_jobs=n_jobs, argument_type="args")
    return pd.concat(output)


def calculate_features(feature_df, labels_df, datasets, type_=None):
    feature_df["datetime"] = pd.to_datetime(feature_df.granule_id.str.split("_", expand=True)[0], format="%Y%m%dT%H:%M:%S", utc=True)
    feature_df["day"] = feature_df.datetime.dt.date
    labels_df["day"] = labels_df.datetime.dt.date

    stats_list = ['mean', 'min', 'max']
    statsDF = feature_df.groupby(["day", "grid_id"]).agg({dataset_name:stats_list for dataset_name in datasets})

    cols = [dataset_name+'_'+stat_name for dataset_name in datasets for stat_name in stats_list]
    statsDF.columns = cols

    statsDF = statsDF.reset_index()

    how = "inner" if type_ == "train" else "left"
    # merged_data = pd.merge(labels_df, statsDF, how=how, left_on=["day", "grid_id"], right_on=["day", "grid_id"])
    merged_data = pd.merge(labels_df, statsDF, how=how, left_on=["day", "grid_id"], right_on=["day", "grid_id"])
    return merged_data



# define global sinu_crs and wgs84_crs
satellite_metadata = pd.read_csv(RAW / "pm25_satellite_metadata.csv", parse_dates=["time_start", "time_end"], index_col=0)
temp_url = satellite_metadata[(satellite_metadata["product"] == "maiac") & (satellite_metadata["split"] == "train")].iloc[0].us_url
temp_hdf = get_hdf(temp_url)
temp_align_dict = get_align_dict(hdf=temp_hdf)
sinu_crs = Proj(f"+proj=sinu +R={temp_align_dict['crs_params'][0]} +nadgrids=@null +wktext").crs
wgs84_crs = CRS.from_epsg("4326")
