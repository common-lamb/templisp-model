#start emacs from model env in shell
#spacemacs/force-init-spacemacs-env
# , n a activate model env
# open inferior repl

################
# Contents
################
# * Setup
# ** Import
# ** Path and file selection
# ** Global variables
# ** Input validation
# * AOI
# ** Define
# ** Create Grid
# * Load EOpatch
# ** Load rasters
# ** Load timestamps etc
# ** Load reference polygons
# *** Bind identities and observations
# *** Import vectors
# *** Rasterize observations
# ** Visualize layers
# *** Object contents
# *** RGB per time
# *** Reference identities map
# *** Rasterized observations
# * Prepare eopatch
# ** Concatenate
# ** Erosion
# ** Sampling
# * Create training data
# ** Extract eopatch
# ** Reshape data
# *** Split patches into train test for GBM
# *** Shape for GBM
# from t,w,h,f to n, m
# where n is pixels ie. w*h&&& and m is features x timesteps
# *** Split samples into train test for TST &&&
# *** Shape for TSP
# from t,w,h,f to s,v,t
# where s is w*h, v is features, t is time
# * GBM experiment
# ** Train
# ** Validate
# *** F1 etc table
# *** Confusion matrices
# *** Class balance
# *** ROC and AUC
# *** Feature importance
# ** Predict
# *** visualize prediction
# ** Quantify prediction
# *** Visualize predicted trait
# *** Visualize trait diff
# *** Quantify agreement
# * TST experiment
# ** Train
# *** Unsupervised training
# *** Transfer learning
# ** Validate
# *** F1 etc table
# *** Confusion matrices
# *** Class balance
# *** ROC and AUC
# *** Feature importance
# ** Predict
# *** visualize prediction
# ** Quantify prediction
# *** Visualize predicted trait
# *** Visualize trait diff
# *** Quantify agreement



################
# * Setup
################
######### ** Import
import os
import pathlib
import glob
import re
import warnings
from osgeo import gdal
from collections import defaultdict
import sys
import itertools
import joblib
from pandas._config import dates
from tqdm.auto import tqdm
import datetime
import numpy as np
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from aenum import MultiValueEnum
from shapely.geometry import Polygon
import lightgbm as lgb
import geopandas as gpd
from sentinelhub import DataCollection, UtmZoneSplitter

from eolearn.core import (EOExecutor,
                          EOPatch,
                          EOTask,
                          EOWorkflow,
                          FeatureType,
                          LoadTask,
                          MergeFeatureTask,
                          OverwritePermission,
                          SaveTask,
                          linearly_connect_tasks,)
from eolearn.io import ImportFromTiffTask, ExportToTiffTask, VectorImportTask
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.features.extra.interpolation import LinearInterpolationTask
from eolearn.geometry import ErosionTask, VectorToRasterTask
from eolearn.ml_tools import FractionSamplingTask
######### ** Path and file selection
# input data
DATA_ROOT= pathlib.Path("/bulk-2/2023-package")
# set expected dirs
DATA_AREAS = os.path.join(DATA_ROOT, "area_poly")
DATA_IDS = os.path.join(DATA_ROOT, "id_poly")
DATA_RASTERS = os.path.join(DATA_ROOT, "rasters")
DATA_TABLE= os.path.join(DATA_ROOT, "tabular")
# set expected files
DATA_train = os.path.join(DATA_AREAS, "test-AOI-north.gpkg")
DATA_test = os.path.join(DATA_AREAS, "test-AOI-south.gpkg")
DATA_all =  os.path.join(DATA_AREAS, "test-AOI.gpkg")
DATA_ids = os.path.join(DATA_IDS, "identities.gpkg")
DATA_table = os.path.join(DATA_TABLE, "field-data.csv")

# intermediate and output data
DATA_OP_ROOT = os.path.join(DATA_ROOT, "..", "model_output")
EOPATCH_DIR= os.path.join(DATA_OP_ROOT, "eopatches")
EOPATCH_SAMPLES_DIR= os.path.join(DATA_OP_ROOT, "eopatches_sampled")
RESULTS_DIR= os.path.join(DATA_OP_ROOT, "results")

######### ** Global variables
np.random.seed(42)
rndm_seed=42
CRS = "EPSG:32614"
expected_n_tifs = 630
expected_indices = ['nir', 'red_edge', 'red', 'green', 'blue', 'ndvi', 'sentera_ndre'] # check for expected unique indices, order irrelevant
used_indices = ['nir', 'red_edge', 'red', 'green', 'blue'] # set order and spectra to use, must be subset of expected

######### ** Input validation
def dir_file_enforce():
    # check exists
    for d in (DATA_ROOT, DATA_AREAS, DATA_IDS, DATA_RASTERS, DATA_TABLE):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Input directory not found: {d}")
    for f in (DATA_train, DATA_test, DATA_all, DATA_ids, DATA_table):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file not found: {f}")
    # make exist
    for d in (DATA_OP_ROOT, EOPATCH_DIR, EOPATCH_SAMPLES_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

dir_file_enforce()

def validate_input_files():
    "Validate expected qualities of input tifs."

    # check n of tifs as expected
    tifs = glob.glob(os.path.join(DATA_RASTERS, "*.tif"))
    if not len(tifs) == expected_n_tifs:
        raise Error(f"The number of tifs is not the expected {expected_n_tifs}. Found: {len(tifs)}")

    # check indices are as expected
    indices = set()
    index_pat = re.compile(r'index_(.+?)_sigma')
    for tif in tifs:
        fileName = os.path.basename(tif)
        index_match = index_pat.search(fileName)
        if index_match:
            indices.add(index_match.group(1))
    if not indices ==  set(expected_indices):
        raise ValueError(f"The indices are not those expected. Found: {indices}")

    # all same projection all same extent
    ref_ds = gdal.Open(tifs[0])
    ref_proj = ref_ds.GetProjection()
    ref_geotransform = ref_ds.GetGeoTransform()
    ref_shape =(ref_ds.RasterYSize, ref_ds.RasterXSize)
    for tif in tifs[1:]:
        ds = gdal.Open(tif)
        if ds.GetProjection() != ref_proj:
            print(ds.GetProjection())
            raise ValueError(f"Projection mismatch in {tif}")
        if ds.GetGeoTransform() != ref_geotransform:
            print("WARNING, potential geotransform mismatch")
            print(f"expected: {ref_geotransform}")
            print(f"found   : {ds.GetGeoTransform()}")
            print(os.path.basename(tif))
        if (ds.RasterYSize, ds.RasterXSize) != ref_shape:
            raise ValueError (f"Shape mismatch in {tif}")
    # crs checks
    for f in [DATA_train, DATA_test, DATA_all, DATA_ids]:
        extent = gpd.read_file(f)
        crs = extent.crs
        if not crs == CRS:
            print(f)
            raise ValueError(f"Unexpected crs: {crs}")

validate_input_files()


################
# * AOI
################
######### ** Define
# Load geopackage file
extent_train = gpd.read_file(DATA_train)

# Get the shape in polygon format
extent_train_shape = extent_train.geometry.values[0]
# Print size
width = extent_train_shape.bounds[2] - extent_train_shape.bounds[0]
height = extent_train_shape.bounds[3] - extent_train_shape.bounds[1]
print(f"Dimension of the area is: {width:.0f} x {height:.0f} m")

# Plot
plt.ion()
extent_train.plot()
plt.axis("off")
plt.close()

######### ** Create Grid
# &&& split into parts to fit RAM

################
# * Load EOpatch
################
######### ** Load rasters
# identify layers
def unique_tif_indicators():
    " Return lists of unique dates, globally selected used_indices, and unique sigma values "

    tifs = glob.glob(os.path.join(DATA_RASTERS, "*.tif"))

    dates = set()
    indices = set()
    sigmas = set()

    date_pat = re.compile(r'date_(.+?)_index')
    index_pat = re.compile(r'index_(.+?)_sigma')
    sigma_pat = re.compile(r'sigma-(.+?)\.tif')

    for file in tifs:
        fileName = os.path.basename(file)
        date_match = date_pat.search(fileName)
        if date_match:
            dates.add(date_match.group(1))
        index_match = index_pat.search(fileName)
        if index_match:
            indices.add(index_match.group(1))
        sigma_match = sigma_pat.search(fileName)
        if sigma_match:
            sigmas.add(sigma_match.group(1))

    # check globally set using is subset of found
    if not(set(used_indices) <= indices):
        raise ValueError("Expected found unique indices to be a superset of used_indices")

    dates = sorted(list(dates)) # alphanumeric order is time order
    indices = used_indices  # use to impose sort by freq, known correct due to validation and subset test
    sigmas = sorted(list(map(float, sigmas)))
    print(dates)
    print(indices)
    print(sigmas)

    return dates, indices, sigmas

dates, indices, sigmas = unique_tif_indicators()

#### tif file set handling tools

def select_tif_path(select_date, select_index, select_sigma):
    "Using dates indices and sigmas, return a single matching path"
    a = glob.glob(os.path.join(DATA_RASTERS, f"date_{select_date}*.tif"))
    b = glob.glob(os.path.join(DATA_RASTERS, f"*index_{select_index}_sigma*.tif"))
    a = set(a)
    b = set(b)

    c = set()
    tifs = glob.glob(os.path.join(DATA_RASTERS, "*.tif"))
    sigma_pat = re.compile(r'sigma-(.+?)\.tif')
    for tif in tifs:
        fileName = os.path.basename(tif)
        sigma_match = sigma_pat.search(fileName)
        if float(sigma_match.group(1)) == float(select_sigma):
            c.add(tif)

    selected = a & b & c
    if len(selected) > 1:
        raise ValueError(f"Selected is longer than one. Selected: {selected}")
    if len(selected) == 0:
        raise ValueError(f"No match found")
    return list(selected)[0]

# select_tif_path(dates[1], indices[2], sigmas[2])

def select_tif_set(date_list, index_list, sigma_list):
    "sets of tif files ordered as imposed in lists returned from unique_tif_indicators"
    selects = list()
    for i in date_list:
        for j in index_list:
            for k in sigma_list:
                sel = select_tif_path(i, j, k)
                selects.append(sel)
    return selects

# select_tif_set(dates[0:], indices[0:1], sigmas[0:1])
test = select_tif_set(dates[0:], indices[0:1], sigmas[0:1])
for t in test:
    print(t)
######### ** Load timestamps etc
######### ** Load reference polygons
#####
# *** Bind identities and observations
#####
# *** Import vectors
#####
# *** Rasterize observations
######### ** Visualize layers
#####
# *** Object contents
#####
# *** RGB per time
#####
# *** Reference identities map
#####
# *** Rasterized observations
################
# * Prepare eopatch
################
######### ** Concatenate
######### ** Erosion
######### ** Sampling
################
# * Create training data
################
######### ** Extract eopatch
######### ** Reshape data
#####
# *** Split patches into train test for GBM
#####
# *** Shape for GBM
# from t,w,h,f to n, m
# where n is pixels ie. w*h&&& and m is features x timesteps
#####
# *** Split samples into train test for TST &&&
#####
# *** Shape for TSP
# from t,w,h,f to s,v,t
# where s is w*h, v is features, t is time
################
# * GBM experiment
################
######### ** Train
######### ** Validate
#####
# *** F1 etc table
#####
# *** Confusion matrices
#####
# *** Class balance
#####
# *** ROC and AUC
#####
# *** Feature importance
######### ** Predict
#####
# *** visualize prediction
######### ** Quantify prediction
#####
# *** Visualize predicted trait
#####
# *** Visualize trait diff
#####
# *** Quantify agreement
################
# * TST experiment
################
######### ** Train
#####
# *** Unsupervised training
#####
# *** Transfer learning
######### ** Validate
#####
# *** F1 etc table
#####
# *** Confusion matrices
#####
# *** Class balance
#####
# *** ROC and AUC
#####
# *** Feature importance
######### ** Predict
#####
# *** visualize prediction
######### ** Quantify prediction
#####
# *** Visualize predicted trait
#####
# *** Visualize trait diff
#####
# *** Quantify agreement
