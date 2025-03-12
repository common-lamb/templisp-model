
#start emacs from model env in shell
#spacemacs/force-init-spacemacs-env
# , n a activate model env
# open inferior repl

import os
import pathlib
import sys
import itertools
import joblib
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

from eolearn.core import (EOExecutor, EOPatch, EOTask, EOWorkflow, FeatureType, LoadTask, MergeFeatureTask, OverwritePermission, SaveTask, linearly_connect_tasks,)
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.features.extra.interpolation import LinearInterpolationTask
from eolearn.geometry import ErosionTask, VectorToRasterTask
from eolearn.io import ExportToTiffTask, SentinelHubInputTask, VectorImportTask # &&& ImportFromTiffTask
from eolearn.ml_tools import FractionSamplingTask

np.random.seed(42)
#### # globals
#### # data load
#### ## parse input data dict
#### ### establish dir structure
# input data
DATA_ROOT= pathlib.Path("/bulk-2/2023-package")

DATA_AREAS = os.path.join(DATA_ROOT, "area_poly")
DATA_IDS = os.path.join(DATA_ROOT, "id_poly")
DATA_RASTERS = os.path.join(DATA_ROOT, "rasters")
DATA_TABLE= os.path.join(DATA_ROOT, "tabular")

DATA_train = os.path.join(DATA_AREAS, "test-AOI-north.gpkg")
DATA_test = os.path.join(DATA_AREAS, "test-AOI-south.gpkg")
DATA_ids = os.path.join(DATA_IDS, "identities.gpkg")
DATA_table = os.path.join(DATA_TABLE, "field-data.csv")

for d in (DATA_ROOT, DATA_AREAS, DATA_IDS, DATA_RASTERS, DATA_TABLE):
    if not os.path.exists(d):
        raise FileNotFoundError(f"Input directory not found: {d}")

for f in (DATA_train, DATA_test, DATA_ids, DATA_table):
    if not os.path.exists(f):
        raise FileNotFoundError(f"Input file not found: {f}")

# intermediate and output data
DATA_OP_ROOT = os.path.join(DATA_ROOT, "..", "model_output")
EOPATCH_DIR= os.path.join(DATA_OP_ROOT, "eopatches")
EOPATCH_SAMPLES_DIR= os.path.join(DATA_OP_ROOT, "eopatches_sampled")
RESULTS_DIR= os.path.join(DATA_OP_ROOT, "results")
for d in (DATA_OP_ROOT, EOPATCH_DIR, EOPATCH_SAMPLES_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

#### ### define aoi

# Load geopackage file
extent_train = gpd.read_file(DATA_train)
# Get the shape in polygon format
extent_train_shape = extent_train.geometry.values[0]

# Plot
plt.ion()
extent_train.plot()
plt.axis("off")
plt.close()

# Print size
width = extent_train_shape.bounds[2] - extent_train_shape.bounds[0]
height = extent_train_shape.bounds[3] - extent_train_shape.bounds[1]
print(f"Dimension of the area is: {width:.0f} x {height:.0f} m")

#### ### identify layers
#&&& sort rasters by date, index in spectral order, sigma in numeric order
