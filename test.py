
#start emacs from model env in shell
#spacemacs/force-init-spacemacs-env
# , n a activate model env
# open inferior repl


import os
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
