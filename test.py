
#start emacs from model env in shell
#spacemacs/force-init-spacemacs-env
# , n a activate model env
# open inferior repl

import os
import pathlib
import glob
import re
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

#### # globals
np.random.seed(42)
seed=42
crs&&&

#### # data load
#### ## parse input data dict
#### ### establish dir structure
# input data
DATA_ROOT= pathlib.Path("/bulk-2/2023-package")

DATA_AREAS = os.path.join(DATA_ROOT, "area_poly")
DATA_IDS = os.path.join(DATA_ROOT, "id_poly")
DATA_RASTERS = os.path.join(DATA_ROOT, "rasters")
DATA_TABLE= os.path.join(DATA_ROOT, "tabular")

for d in (DATA_ROOT, DATA_AREAS, DATA_IDS, DATA_RASTERS, DATA_TABLE):
    if not os.path.exists(d):
        raise FileNotFoundError(f"Input directory not found: {d}")

DATA_train = os.path.join(DATA_AREAS, "test-AOI-north.gpkg")
DATA_test = os.path.join(DATA_AREAS, "test-AOI-south.gpkg")
DATA_ids = os.path.join(DATA_IDS, "identities.gpkg")
DATA_table = os.path.join(DATA_TABLE, "field-data.csv")

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

# Print size
width = extent_train_shape.bounds[2] - extent_train_shape.bounds[0]
height = extent_train_shape.bounds[3] - extent_train_shape.bounds[1]
print(f"Dimension of the area is: {width:.0f} x {height:.0f} m")

# Plot
plt.ion()
extent_train.plot()
plt.axis("off")
plt.close()

#### ### identify layers
def unique_tif_indicators():
    "Validate expected qualities of input tifs. Return unique dates, indices, and sigma values "

    expected_n_tifs = 630
    expected_indices = ['nir', 'red_edge', 'red', 'green', 'blue', 'ndvi', 'sentera_ndre']

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

    if not len(tifs) == expected_n_tifs:
        raise Error(f"The number of tifs is not the expected {expected_n_tifs}. Found: {len(tifs)}")
    if not indices ==  set(expected_indices):
        raise ValueError(f"The indices are not those expected. Found: {indices}")

    dates = sorted(list(dates)) # alphanumeric order is time order
    indices = expected_indices  # use this sort, known correct
    sigmas = sorted(list(map(float, sigmas)))

    return dates, indices, sigmas

dates, indices, sigmas = unique_tif_indicators()


'''
python multi value return from function



In Python, functions can return multiple values using tuples, lists, or dictionaries. Here's a concise example using a tuple:

```python
def get_name_and_age():
    return "Alice", 30

name, age = get_name_and_age()
print(name, age)  # Output: Alice 30
```

You can also use unpacking to assign the returned values to variables directly.'''

#### ## eo-learn input task
#### ### load layers to patch

class SentinelHubValidDataTask(EOTask):
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __init__(self, output_feature):
        self.output_feature = output_feature

    def execute(self, eopatch):
        eopatch[self.output_feature] = eopatch.mask["IS_DATA"].astype(bool) & (~eopatch.mask["CLM"].astype(bool))
        return eopatch


class AddValidCountTask(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch[FeatureType.MASK_TIMELESS, self.name] = np.count_nonzero(eopatch.mask[self.what], axis=0)
        return eopatch

#### Define the workflow tasks
# BAND DATA
# Add a request for S2 bands.
# Here we also do a simple filter of cloudy scenes (on tile level).
# The s2cloudless masks and probabilities are requested via additional data.
band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
add_data = SentinelHubInputTask(
    bands_feature=(FeatureType.DATA, "BANDS"),
    bands=band_names,
    resolution=10,
    maxcc=0.8,
    time_difference=datetime.timedelta(minutes=120),
    data_collection=DataCollection.SENTINEL2_L1C,
    additional_data=[(FeatureType.MASK, "dataMask", "IS_DATA"), (FeatureType.MASK, "CLM"), (FeatureType.DATA, "CLP")],
    max_threads=5,
)


# CALCULATING NEW FEATURES
# NDVI: (B08 - B04)/(B08 + B04)
# NDWI: (B03 - B08)/(B03 + B08)
# NDBI: (B11 - B08)/(B11 + B08)

ndvi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDVI"), [band_names.index("B08"), band_names.index("B04")]
)
ndwi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDWI"), [band_names.index("B03"), band_names.index("B08")]
)
ndbi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDBI"), [band_names.index("B11"), band_names.index("B08")]
)

# VALIDITY MASK
# Validate pixels using SentinelHub's cloud detection mask and region of acquisition
add_sh_validmask = SentinelHubValidDataTask((FeatureType.MASK, "IS_VALID"))

# COUNTING VALID PIXELS
# Count the number of valid observations per pixel using valid data mask
add_valid_count = AddValidCountTask("IS_VALID", "VALID_COUNT")

# SAVING TO OUTPUT (if needed)
save = SaveTask(EOPATCH_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)+begin_src python
