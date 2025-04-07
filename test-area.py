#start emacs from model env in shell
# M-x spacemacs/force-init-spacemacs-env
# , n a activate model env
# , ' open inferior repl

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
# *** Split samples into train test for TST &&&
# *** Shape for TSP
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

import shapely.geometry
import shapely.validation
from copy import deepcopy
from osgeo import gdal
from fiona.collection import BytesCollection
from collections import defaultdict
import sys
import itertools
import joblib
import pandas as pd
from pandas._config import dates
import datetime
import numpy as np
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import lightgbm as lgb
from tsai.all import *
import rasterio
import geopandas as gpd
from sentinelhub import DataCollection, UtmZoneSplitter

from eolearn.core.core_tasks import CreateEOPatchTask, InitializeFeatureTask, RemoveFeatureTask
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
from eolearn.core.eonode import EONode
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
DATA_RASTERS = os.path.join(DATA_ROOT, "test-rasters") #dir rasters or test-rasters
DATA_TABLE= os.path.join(DATA_ROOT, "tabular")
# set expected files
DATA_train = os.path.join(DATA_AREAS, "test-AOI-north.gpkg")
DATA_test = os.path.join(DATA_AREAS, "test-AOI-south.gpkg")
DATA_all =  os.path.join(DATA_AREAS, "test-AOI.gpkg")
DATA_ids = os.path.join(DATA_IDS, "identities.gpkg")
DATA_table = os.path.join(DATA_TABLE, "field-data.csv")

# intermediate and output data
DATA_OP_ROOT = os.path.join(DATA_ROOT, "..", "model_output")
EOPATCH_TRAIN_DIR= os.path.join(DATA_OP_ROOT, "eopatches_train")
EOPATCH_VALIDATE_DIR= os.path.join(DATA_OP_ROOT, "eopatches_validation")
DATA_DIR= os.path.join(DATA_OP_ROOT, "data")
EOPATCH_SAMPLES_DIR= os.path.join(DATA_OP_ROOT, "eopatches_sampled")
RESULTS_DIR= os.path.join(DATA_OP_ROOT, "results")

######### ** Global variables
RNDM = 42
np.random.seed(RNDM)
CRS = "EPSG:32614"
GRID_SIZE = 500 # pixel count of patch edge, reduce to relieve RAM
RESOLUTION = 0.003 # gsd in meters
EXPECTED_N_TIFS = 630
EXPECTED_INDICES = ['nir', 'red_edge', 'red', 'green', 'blue', 'ndvi', 'sentera_ndre'] # check for expected unique indices, order irrelevant
USED_INDICES = ['nir', 'red_edge', 'red', 'green', 'blue'] # set order and spectra to use, must be subset of expected

SAMPLE_RATE = 0.01 # percentage of eopatches sampled for training. in (0.0-1.0)
TEST_PERCENTAGE = 0.20 # perventage of test-train set to use for testing. in (0.0-1.0)

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
    for d in (DATA_OP_ROOT, EOPATCH_TRAIN_DIR, EOPATCH_VALIDATE_DIR, DATA_DIR, EOPATCH_SAMPLES_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

def parse_identifiers(path):
    "extract date index and sigma from a path"
    date_pat = re.compile(r'date_(.+?)_index')
    index_pat = re.compile(r'index_(.+?)_sigma')
    sigma_pat = re.compile(r'sigma-(.+?)\.tif')

    fileName = os.path.basename(path)

    date_found = str()
    index_found = str()
    sigma_found = str()

    date_match = date_pat.search(fileName)
    if date_match:
        date_found = (date_match.group(1))
    else:
        print(f"WARNING: No date match found in the path: {path}")

    index_match = index_pat.search(fileName)
    if index_match:
        index_found  = (index_match.group(1))
    else:
        print(f"WARNING: No index match found in the path: {path}")

    sigma_match = sigma_pat.search(fileName)
    if sigma_match:
        sigma_found = (sigma_match.group(1))
    else:
        print(f"WARNING: No sigma match found in the path: {path}")

    return {'date':date_found,
            'index':index_found,
            'sigma':sigma_found}

def input_tifs():
    tifs = glob.glob(os.path.join(DATA_RASTERS, "*.tif"))
    return tifs

def validate_input_files(tifs=input_tifs(), expected_n_tifs=EXPECTED_N_TIFS, expected_indices=EXPECTED_INDICES, used_indices=USED_INDICES):
    "Validate expected qualities of input tifs."

    # check n of tifs as expected
    if not len(tifs) ==  expected_n_tifs:
        raise Error(f"The number of tifs is not the expected {expected_n_tifs}. Found: {len(tifs)}")

    # check indices are as expected
    indices = set()
    for tif in tifs:
        index = parse_identifiers(tif)['index']
        indices.add(index)
        # print nodata value if weird
        with rasterio.open(tif) as src:
            nd_value = src.nodata
            if nd_value != -10000:
                print(f"gtiff: {tif}")
                print(f"Nodata value: {nd_value}")
    if not indices ==  set(expected_indices):
        raise ValueError(f"The indices are not those expected. Found: {indices}")
    # check globally set using is subset of found
    if not(set(used_indices) <= indices):
        raise ValueError("Expected found unique indices to be a superset of used_indices")

    # all share projection, shape and extent
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
            raise ValueError(f"Unexpected crs for file: {f} crs: {crs}")

dir_file_enforce()
validate_input_files()
# for test are and for full area
# &&& 2023-09-05 potential geotransform mismatch
# &&& 2023-07-24 Nodata value is None


################
# * AOI
################
######### ** Define

def area_grid(area, grid_size=GRID_SIZE, resolution=RESOLUTION, show=False):
    """
   read in gpkg file compute patches by pixels

    Args:
        area: the AOI geopackage with area shape
        grid_size: patch size to split the AOI into
        resolution: gsd of data in meters
    Return:
        bbox-list: an array of bounding boxes
    """

    # Load geopackage file
    extent = gpd.read_file(area)
    # Get the shape in polygon format
    extent_shape = extent.geometry.values[0]
    # get the width and height of the area in meters
    width = extent_shape.bounds[2] - extent_shape.bounds[0]
    height = extent_shape.bounds[3] - extent_shape.bounds[1]
    print(f"Dimension of the area is: {width:.0f} x {height:.0f} m")
    # get the width and height of the area in pixels
    width_pix = int(width / resolution)
    height_pix = int(height / resolution)
    print(f"Dimension of the area is {width_pix} x {height_pix} pixels")
    # &&& get the number of grid cells wide and tall our area is
    # get the width and height of the area in patches
    width_patch = int(round(width_pix / grid_size))
    height_patch =  int(round(height_pix / grid_size))
    print(f"Dimension of the area is {width_patch} x {height_patch} patches (rounded to nearest full patch)")
    # length of patch edge m=(m/px)*px
    edge_m =resolution*grid_size

    # if show:
    #     plt.ion()
    #     extent.plot()
    #     plt.axis("off")
    #     plt.close()

    # Create a splitter to obtain a list of bboxes
    bbox_splitter = UtmZoneSplitter([extent_shape], extent.crs, edge_m)
    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    # Prepare info of selected EOPatches
    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs = [info["index"] for info in info_list]
    idxs_x = [info["index_x"] for info in info_list]
    idxs_y = [info["index_y"] for info in info_list]
    bbox_gdf = gpd.GeoDataFrame({"index": idxs, "index_x": idxs_x, "index_y": idxs_y}, crs=extent.crs, geometry=geometry)

    #collect patches
    patch_ids = []
    for idx, info in enumerate(info_list):
        patch_ids.append(idx)

    if show:
        # Plot the polygon and its partitions

        plt.ion()
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_title(f"Area: {area}, partitioned to patches", fontsize=25)
        extent.plot(ax=ax, facecolor="w", edgecolor="b", alpha=0.5) # area shape
        bbox_gdf.plot(ax=ax, facecolor="w", edgecolor="r", alpha=0.5) # patch grid
        for bbox, info in zip(bbox_list, info_list):
            geo = bbox.geometry
            ax.text(geo.centroid.x, geo.centroid.y, info["index"], ha="center", va="center")
        plt.axis("off");

    print(f"Total patches is: {len(bbox_list)}")
    return bbox_list

######### ** Create Grid
# show grid that will be used later
test = area_grid(DATA_train, show=True)

################
# * Load EOpatch
################

######### ** Load rasters

# tif file set handling tools
def unique_tif_indicators(tifs=input_tifs(), used_indices=USED_INDICES):
    " Return correctly ordered lists of: unique dates, globally selected used_indices, and unique sigma values "

    rawdates = set()
    rawindices = set()
    rawsigmas = set()

    for file in tifs:
        parsed = parse_identifiers(file)
        # no expected match may be empty
        if parsed['date'] == '' or parsed['index'] == '':
            raise RuntimeError(f"A file has no unique match: {file}")
        rawdates.add(parsed['date'])
        rawindices.add(parsed['index'])
        rawsigmas.add(parsed['sigma'])

    # rawindices must be a superset of USED_INDICES
    if not set(used_indices) <= rawindices:
        raise AssertionError(f"The raw indices are not a super set of USED_INDICES. raw: {rawindices}")

    dates = sorted(list(rawdates)) # alphanumeric order is time order
    indices = used_indices  # use to impose sort by freq, known correct due to validation and subset test
    sigmas = sorted(list(map(float, rawsigmas))) # sorted list of numbers

    return {'dates':dates,
            'indices':indices,
            'sigmas':sigmas}

def select_tif_path(select_date, select_index, select_sigma, tifs=input_tifs() ):
    "Using intersect of dates indices and sigmas, return a single matching path"
    a = set() # dates
    b = set() # indices
    c = set() # sigmas

    for file in tifs:
        parsed = parse_identifiers(file)
        if parsed['date'] == select_date:
            a.add(file)
        if parsed['index'] == select_index:
            b.add(file)
        if float(parsed['sigma']) == float(select_sigma):
            c.add(file)

    selected = a & b & c
    if len(selected) > 1:
        raise ValueError(f"Selected is longer than one. Selected: {selected}")
    if len(selected) == 0:
        raise ValueError(f"No match found")
    return list(selected)[0]
# select_tif_path(unique_tif_indicators()['dates'][1], unique_tif_indicators()['indices'][1], unique_tif_indicators()['sigmas'][1])

def select_tif_set(date_list, index_list, sigma_list):
    "sets of tif files ordered as imposed in lists returned from unique_tif_indicators"
    selects = list()
    for i in date_list:
        for j in index_list:
            for k in sigma_list:
                sel = select_tif_path(i, j, k)
                selects.append(sel)
    return selects
# select_tif_set(unique_tif_indicators()['dates'][1:2], unique_tif_indicators()['indices'][1:2], unique_tif_indicators()['sigmas'][1:2])
# len( select_tif_set(unique_tif_indicators()['dates'],['red'],[2.0]))

# define tasks for load workflow
def make_workflow_name(n, i, s):
    "assemble name for the date-set of tiffs, like 0000_index_<>_sigma_<>-<> "
    # pad the counter to maintain ordering
    zeroPaddedN = f"{n:0{4}d}"
    unsafe = f"{zeroPaddedN}_index_{i}_sigma_{s}"
    # need to drop decimals,invalid in feature names
    safe = unsafe.replace('.', '-')
    return safe

def taskList_to_nodeList(taskList):
    "connects a list of tasks as EONodes, returning a list of nodes"
    nodes = []
    node_previous = None

    for task in taskList:
        node_current = None
        if node_previous is None:
            # the first node has no previous dependency
            node_current = EONode(task)
        else:
            # all other nodes depend on the previous
            node_current = EONode(task, inputs=[node_previous])
        # add the Node and assign previous
        nodes.append(node_current)
        node_previous = node_current
    return nodes

def stack_timeless_features(*features):
    "Merge multiple DATA_TIMELESS features into a time-series DATA feature"
    print(f"Merging {len(features)} features")

    for i, feat in enumerate(features):
        print(f"feature: {i}")
        print(f"  Shape: {feat.shape}")
        print(f"  Dtype: {feat.dtype}")

    stacked= np.stack(features, axis=0)
    print(f"Stacked shape: {stacked.shape}")
    return stacked

def CreateStackLoaderWorkflow(indicators, areas, eopatch_dir):
    """
    Creates a workflow to import tiffs individually, concatenate sets on the time axis, and delete the individuals.
    Args:
    indicators=unique_tif_indicators() order of dates,indices,sigmas determines final order in eopatch
    areas=area_grid(DATA_train) or area_grid(DATA_test) grid determines parallell processing and patch size
    Rets:
    workflow
    execution_args for workflow
    """

    dates = indicators['dates']
    indices = indicators['indices']
    sigmas = indicators['sigmas']

    n = 0 # counter padded and prepended to name to force ordering of sets
    tasklist = [] # aggregate the set of tasks
    create_task = CreateEOPatchTask() # initialize, will accept bbox for the entire chain
    tasklist.append(create_task)
    #SET actions, all dates are used
    for i in indices:
        for s in sigmas:
            name_set = make_workflow_name(n, i, s)
            tiffs = select_tif_set(dates, [i], [s])
            #SINGLE actions
            m = 0 # identifiers for singles of a set, pad and prepend
            single_list = []
            for tif in tiffs:
                name_single = f"{m:0{3}d}_{name_set}"
                single_data_feature = (FeatureType.DATA_TIMELESS, name_single)
                load_task = ImportFromTiffTask(feature=single_data_feature, path=tif)
                tasklist.append(load_task)
                single_list.append(single_data_feature)
                m = m + 1
            # concatenate all single layers to a set
            set_data_feature = (FeatureType.DATA, name_set)
            # zip_function:  expand dims to 4d (1t*h*w*1c), then concat along time to 3d (t*h*w*1c)
            merge_task = MergeFeatureTask(single_list, set_data_feature, zip_function=stack_timeless_features)
            tasklist.append(merge_task)
            # delete task
            remove_task = RemoveFeatureTask(single_list)
            tasklist.append(remove_task)
            n = n + 1
    save_task = SaveTask(eopatch_dir, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    tasklist.append(save_task)

    workflow_nodes = taskList_to_nodeList(tasklist)
    workflow = EOWorkflow(workflow_nodes)
    # additional arguments
    execution_args = []
    for idx, bbox in enumerate(areas):
        args_dict = {workflow_nodes[0]: {"bbox": bbox}, # create task is first
                     workflow_nodes[-1]: {"eopatch_folder": f"eopatch_{idx}"} # save task is last
                     }
        execution_args.append(args_dict)
    return workflow, execution_args

def execute_prepared_workflow(workflow_and_args):
    # create workflow
    wf, args = workflow_and_args
    # Execute the workflow
    executor = EOExecutor(wf, args, save_logs=True)
    executor.run(workers=1) # workers must be 1 to avoid pickling error
    executor.make_report()

    failed_ids = executor.get_failed_executions()
    if failed_ids:
        raise RuntimeError(
            f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
            f"For more info check report at {executor.get_report_path()}"
        )

def ask_loadgeotiffs():
    print("TIME INTENSIVE JOB load all geotiffs into training eopatches stacked")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreateStackLoaderWorkflow(indicators=unique_tif_indicators(),
                                                            areas=area_grid(DATA_train),
                                                            eopatch_dir=EOPATCH_TRAIN_DIR ))

ask_loadgeotiffs()

######### ** Load timestamps etc

class AddTimestamps(EOTask):
    "Adds timestamps to an eopatch. uses the function unique_tif_indicators to identify times"
    def __init__(self, unique_tif_indicators=unique_tif_indicators()):
        self.indicators = unique_tif_indicators
        self.dates = self.indicators['dates']
    def _toDatetime(self, stringlist):
        fmt = '%Y-%m-%d'
        return [datetime.datetime.strptime(s, fmt) for s in stringlist]
    def execute(self, eopatch):
        eopatch.timestamps = self._toDatetime(self.dates)
        return eopatch

class MakeAreaMask(EOTask):
    "Uses polygon to make a valid area mask. Arg: path to geopackage containing geometry"
    def __init__(self, gpkg_path):
        self.gpkg_path = gpkg_path
    def _load_polygon(self):
        gdf = gpd.read_file(self.gpkg_path)
        unified = shapely.unary_union(gdf.geometry)
        return unified
    def _get_dimensions(self, eopatch):
        for feature_type, feature_name in eopatch.get_features():
            if feature_type == FeatureType.DATA:
                height, width = eopatch.get_spatial_dimension(feature_type, feature_name)
                break
        else:
            raise ValueError("No DATA feature found to infer spatial dimensions")
        return height, width
    def _get_transform(self, eopatch, height, width):
        bbox = eopatch.bbox
        tnsf = rasterio.transform.from_bounds(
            west=bbox.min_x,
            south=bbox.min_y,
            east=bbox.max_x,
            north=bbox.max_y,
            width=width,
            height=height
        )
        return tnsf
    def _get_mask(self, polygon, transform, height, width):
        mask = rasterio.features.geometry_mask(
            [polygon],
            out_shape =(height, width),
            transform = transform,
            invert = True # Pixel is inside geometry
        )
        return mask
    def execute(self, eopatch):
        height, width = self._get_dimensions(eopatch)
        transform = self._get_transform(eopatch, height, width)
        polygon = self._load_polygon()
        mask = self._get_mask(polygon, transform, height, width) # shape (n*m)
        mask3d = mask[..., np.newaxis] # add d for (n*m*d)
        # if FeatureType.MASK_TIMELESS not in eopatch:
        #     eopatch[FeatureType.MASK_TIMELESS] = {}
        eopatch[FeatureType.MASK_TIMELESS, "IN_POLYGON"] = mask3d.astype(bool)
        print(eopatch)
        return eopatch

######### ** Load reference polygons
#####
# *** Bind identities and observations

def report_repair_invalid_geometry(invalid_index, gdf, show):
    invalid_geom = gdf.loc[invalid_index, 'geometry']
    explanation = shapely.validation.explain_validity(invalid_geom)
    repaired_geom = shapely.validation.make_valid(invalid_geom)

    if show:
        print("\nInvalid geometry\n",invalid_geom)
        print("\nInvalid explanation\n", explanation)
        print("\nRepaired geometry\n", repaired_geom)

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("Original Geometry")
        ax2.set_title("Repaired Geometry")
        if hasattr(invalid_geom, 'exterior'):
            ax1.plot(*invalid_geom.exterior.xy)
        else:
            ax1.plot(invalid_geom.xy)
        if isinstance(repaired_geom, shapely.geometry.GeometryCollection):
            for geom in repaired_geom.geoms:
                if hasattr(geom, 'exterior'):
                    ax2.plot(*geom.exterior.xy)
                else:
                    ax2.plot(*geom.xy)
        elif hasattr(repaired_geom, 'exterior'):
            ax2.plot(*repaired_geom.exterior.xy)
        else:
            ax2.plot(*repaired_geom.xy)
        ax1.axis('equal')
        ax2.axis('equal')
        plt.show()

    return repaired_geom

def check_valid_geometry(gdf , show=False):
    gdf['is_valid'] = gdf.geometry.is_valid
    invalid_geoms = gdf[~gdf['is_valid']]

    if show:
        print(f"Total geometries: {len(gdf)}")
        print(f"Valid geometries: {gdf['is_valid'].sum}")
        print(f"Invalid geometries: {len(invalid_geoms)}")
        print(f"Invalid indices: {invalid_geoms.index.tolist()}")

    repaired_gdf = deepcopy(gdf)
    for idx in invalid_geoms.index:
        repaired_gdf.loc[idx, 'geometry'] = report_repair_invalid_geometry(idx, gdf, show)

    return repaired_gdf

# show the repairs that will be used later
test = check_valid_geometry(gpd.read_file(DATA_ids), show=True)

def bind_observations(polygons=DATA_ids, observations=DATA_table, ddir=DATA_DIR):
    "Append table of observation data to polygons, ensure common column, then row bind on samples. Returns: fiona readable object"
    gdf = gpd.read_file(polygons)
    gdf = check_valid_geometry(gdf)
    df = pd.read_csv(observations)
    # ensure correct attributes
    target_shared = 'sample'
    change_from = 'SAMPLE'
    cols_gdf = gdf.columns
    # must have target_shared
    if not target_shared in cols_gdf:
        raise ValueError(f"Expected column not found in {cols_gdf} Expected: {target_shared}")
    cols_df = df.columns
    # must have change_from
    if not change_from in cols_df:
        raise ValueError(f"Expected column not found in {cols_df} Expected: {change_from}")
    # make rename for merge
    df.rename(columns={change_from: target_shared}, inplace=True)
    # merge
    merged_gdf = gdf.merge(df, on=target_shared, how='left')
    # merged_gdf.columns
    # write to disk
    path = os.path.join(ddir, 'bound_observations.gpkg')
    abs_path = os.path.abspath(path)
    merged_gdf.to_file(abs_path, driver='GPKG', layer='name')
    return abs_path

def CreateDetailsLoaderWorkflow(areas, observations, eopatch_dir):
    """
    Creates a workflow to add dates, masks, rasterized observation data to eopatches.
    Args:
    areas=area_grid(DATA_train) or area_grid(DATA_test), grid determines patch count and size
    observations=bind_observations(), a path to a fiona readable object with identity geometries and observation data per identity
    eopatch_dir path to the loaded tif stack eopatches which this workflow will augment
    Rets:
    workflow
    execution_args for workflow
    """

    #####
    # *** Import vectors
    vector_feature = FeatureType.VECTOR_TIMELESS, "IDENTITIES"
    vector_import_task = VectorImportTask(vector_feature, observations)

    #####
    # *** Rasterize observations
    '''
    merged_gdf.columns for selection by values_column

    Index(['sample', 'geometry', 'NAME', 'HEIGHT-CM', 'ROW-TYPE',
           'HULLESS-CONDITION', 'SBLOTCH-RATING', 'WEIGHT', 'DIAMETER', 'AREA',
           'STEM-WEIGHT', 'DENSITY', 'ROWS', 'BARLEY-WHEAT', 'HULLED',
           'SBLOTCH-LMH'],
          dtype='object')

    which are valid for test area &&&
    '''

    rasterize_height_task = VectorToRasterTask(
        vector_feature, # as used in vector import task
        (FeatureType.DATA_TIMELESS, "HEIGHT"), #name of rasterized new layer, DATA floats
        values_column="HEIGHT-CM", # col of merged_gdf.columns to rasterize
        raster_shape=(FeatureType.MASK_TIMELESS, "IN_POLYGON"),
        raster_dtype=np.float32, # &&& float
    )

    # initialize tasks or copy
    load_task = LoadTask(eopatch_dir)
    # clean_task = RemoveFeatureTask(features=[(FeatureType.DATA_TIMELESS, 'HEIGHT')])
    add_timestamps_task = AddTimestamps()
    make_areamask_task = MakeAreaMask(DATA_train)
    vector_task = vector_import_task
    rasterize_task = rasterize_height_task
    save_task = SaveTask(eopatch_dir, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    # node list
    workflow_nodes = linearly_connect_tasks(load_task, add_timestamps_task, make_areamask_task, vector_task, rasterize_task, save_task)

    # workflow
    workflow = EOWorkflow(workflow_nodes)
    # additional arguments
    execution_args = []
    for idx, bbox in enumerate(areas):
        execution_args.append(
            {
                workflow_nodes[0]: {"eopatch_folder": f"eopatch_{idx}"}, # load task is first
                workflow_nodes[-1]: {"eopatch_folder": f"eopatch_{idx}"} # save task is last
            }
        )

    return workflow, execution_args

def ask_loadDetails():
    print("Load masks and timestamps to the patches after the gtiff stacks")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreateDetailsLoaderWorkflow(areas=area_grid(DATA_train),
                                                          observations=bind_observations(),
                                                          eopatch_dir=EOPATCH_TRAIN_DIR))
ask_loadDetails()

######### ** Visualize layers
#####
# *** Object contents
eopatch = EOPatch.load(os.path.join(EOPATCH_TRAIN_DIR, 'eopatch_0'))
eopatch
eopatch.timestamps
data_keys = sorted(list(eopatch.data.keys()))
data_keys

#####
# *** RGB per time
eopatch.plot((FeatureType.DATA, data_keys[-1]))
#&&& make vis on load to check

#####
# *** Reference identities map
eopatch.plot((FeatureType.MASK_TIMELESS, 'IN_POLYGON'))

# *** Rasterized observations
eopatch.plot((FeatureType.DATA_TIMELESS, 'HEIGHT'))

################
# * Prepare eopatch
################

def CreatePatchPrepWorkflow(areas, eopatch_dir, eopatch_out_dir, trait, sample_rate=SAMPLE_RATE):
    "Creates a workflow to finalize and sample eopatches. "

    eopatch = EOPatch.load(os.path.join(eopatch_dir, 'eopatch_0'))
    data_keys = sorted(list(eopatch.data.keys()))

    load_task = LoadTask(eopatch_dir)

    ######### ** Concatenate
    concatenate_task = MergeFeatureTask({FeatureType.DATA: data_keys}, (FeatureType.DATA, "FEATURES_TRAINING"))

    ######### ** Erosion
    erosion_task = ErosionTask(mask_feature=(FeatureType.DATA_TIMELESS, trait, f"{trait}_ERODED"), disk_radius=1)

    ######### ** Sampling
    sampling_task = FractionSamplingTask(features_to_sample=[(FeatureType.DATA, 'FEATURES_TRAINING', 'FEATURES_SAMPLED'),
                                                             (FeatureType.DATA_TIMELESS, f"{trait}_ERODED", f"{trait}_SAMPLED")],
                                         sampling_feature=(FeatureType.MASK_TIMELESS, 'IN_POLYGON'),
                                         fraction=sample_rate,
                                         exclude_values=[0])

    save_task = SaveTask(eopatch_out_dir, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # node list
    workflow_nodes = linearly_connect_tasks(load_task, concatenate_task, erosion_task, sampling_task, save_task)

    # workflow
    workflow = EOWorkflow(workflow_nodes)

    # additional arguments
    execution_args = []
    for idx, bbox in enumerate(areas):
        execution_args.append(
            {
                workflow_nodes[0]: {"eopatch_folder": f"eopatch_{idx}"}, # load task is first
                workflow_nodes[-2]: {"seed": RNDM}, # sample task is second last
                workflow_nodes[-1]: {"eopatch_folder": f"eopatch_{idx}"} # save task is last
            }
        )

    return workflow, execution_args

def ask_preparePatches():
    print("Finalize and sample EOPatches?")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreatePatchPrepWorkflow(areas=area_grid(DATA_train),
                                                          eopatch_dir=EOPATCH_TRAIN_DIR,
                                                          eopatch_out_dir=EOPATCH_SAMPLES_DIR,
                                                          trait='HEIGHT'))

ask_preparePatches()

################
# * Create training data
################

######### ** Extract eopatches
def sampledData(areas, eopatch_samples_dir, trait):
    """
    Takes grid of areas, a source of eopatches, and a single trait.
    Concatenates all then Returns sampled_features and trait_eroded
    """

    sampled_eopatches = []
    for i in range(len(areas)):
        sample_path = os.path.join(eopatch_samples_dir, f"eopatch_{i}")
        sampled_eopatches.append(EOPatch.load(sample_path, lazy_loading=True))

    features = np.concatenate([eopatch.data["FEATURES_SAMPLED"] for eopatch in sampled_eopatches], axis=1)
    labels = np.concatenate([eopatch.data_timeless[f"{trait}_SAMPLED"] for eopatch in sampled_eopatches], axis=0)

    return features, labels

f, l = sampledData(areas=area_grid(DATA_train), eopatch_samples_dir=EOPATCH_SAMPLES_DIR, trait = 'HEIGHT')
f.shape # (10, 1686, 1, 45)
l.shape #     (1686, 1, 1)

test = f, l
######### ** Reshape data

# *** Shape for TSP
def reshape_eopatch_to_TSAI(data):
    """
    from eopatch as t,w,h,f
    to
    TSAI requires data as s,v,t
    where s is w*h, v is features, t is time
    """
    features, labels = data

    # t w h f 0123 -4-3-2-1
    ft, fw, fh, ff = features.shape
    lw, lh, lf = labels.shape
    features_reshaped = np.moveaxis(features, 0, -1).reshape(fw * fh, ff, ft)
    labels_reshaped = labels.reshape(lw * lh)

    return features_reshaped, labels_reshaped

fT, lT = reshape_eopatch_to_TSAI(data=test)
fT.shape # (1686, 45, 10)
lT.shape # (1686,)
testT = fT, lT

# *** Shape for GBM
def reshape_to_GBM(data, TSAI_shape=True):
    """
    from eopatch as t,w,h,f
    or from TSAI as s,v,t
    to
    GBM requires data as n, m
    where n is pixels ie. s or w*h and m is features x timesteps
    """
    features, labels = data

    if TSAI_shape:
        # s v t 012 -3-2-1
        fs, fv, ft = features.shape
        features_reshaped = np.moveaxis(features, -1, -2).reshape(fs, ft * fv)
        labels_reshaped = labels
    else:
        # direct from eolearn
        # t w h f 0123 -4-3-2-1
        ft, fw, fh, ff = features.shape
        features_reshaped = np.moveaxis(features, 0, -2).reshape(fw * fh, ft * ff)
        lw, lh, lf = labels.shape
        labels_reshaped = labels.reshape(lw * lh)

    return features_reshaped, labels_reshaped

fG, lG = reshape_to_GBM(data=testT)
fG.shape # (1686, 450)
lG.shape # (1686,)

#####
# *** Split samples into train test sets
#####

def split_for_TSAI(data, test_percentage=TEST_PERCENTAGE):
    "Takes eolearn  shaped features and labels, returns X,Y,splits shaped for TSAI"
    features, labels = reshape_eopatch_to_TSAI(data)
    # &&& make show false optional
    splits = get_splits(labels, valid_size=test_percentage, stratify=True, random_state=RNDM, shuffle=True)
    return features, labels, splits

fS, lS, sS = split_for_TSAI(test)
fS.shape # (1686, 45, 10)
lS.shape # (1686,)
len(sS[0]) # 1349
len(sS[1]) # 337
testS = fS, lS, sS



def split_reconfigure_for_GBM(split_data):
    "Takes TSAI shaped X,Y,splits, and returns x_train, y_train, x_test, y_test shaped for GBM"
    features, labels, splits = split_data

    split_train = splits[0]
    mask_train = np.zeros(len(features), dtype=bool)
    mask_train[split_train] = True
    x_train = features[mask_train]
    print(x_train.shape)
    y_train = labels[mask_train]
    print(y_train.shape)
    data_train = x_train, y_train

    split_test = splits[1]
    mask_test = np.zeros(len(features), dtype=bool)
    mask_test[split_test] = True
    x_test =features[mask_test]
    print(x_test.shape)
    y_test =labels[mask_test]
    print(y_test.shape)
    data_test = x_test, y_test

    x_train_GMB, y_train_GBM = reshape_to_GBM(data=data_train)
    x_test_GBM, y_test_GBM = reshape_to_GBM(data=data_test)
    return x_train_GMB, y_train_GBM, x_test_GBM, y_test_GBM

x_train_GMB, y_train_GBM, x_test_GBM, y_test_GBM = split_reconfigure_for_GBM(testS)
x_train_GMB.shape # (1349, 450)
y_train_GBM.shape # (1349,)
x_test_GBM.shape # (337, 450)
y_test_GBM.shape # (337,)

################
# * GBM experiment
################
def create_GBM_training_data():
    a = sampledData(areas=area_grid(DATA_train),
                    eopatch_samples_dir=EOPATCH_SAMPLES_DIR,
                    trait = 'HEIGHT')
    b = split_for_TSAI(a)
    return split_reconfigure_for_GBM(b)

x_train_GMB, y_train_GBM, x_test_GBM, y_test_GBM = create_GBM_training_data()
x_train_GMB.shape # (1349, 450)
y_train_GBM.shape # (1349,)
x_test_GBM.shape # (337, 450)
y_test_GBM.shape # (337,)

def create_TSAI_training_data():
    a = sampledData(areas=area_grid(DATA_train),
                    eopatch_samples_dir=EOPATCH_SAMPLES_DIR,
                    trait = 'HEIGHT')
    return split_for_TSAI(a)

x_TSAI, y_TSAI, splits = create_TSAI_training_data()
x_TSAI.shape # (1686, 45, 10)
y_TSAI.shape # (1686,)
len(splits[0]) # 1349
len(splits[1]) # 337

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




############################################

class MultiLoader(EOTask):
    "Not used but kept because it was working although the shape was wrong"
    def __init__(self, dates, indices, sigmas):
        self.dates = dates
        self.indices = indices
        self.sigmas = sigmas
    def _toDatetime(self, stringlist):
        fmt = '%Y-%m-%d'
        return [datetime.datetime.strptime(s, fmt) for s in stringlist]
    def _toNParray(self, tiffs):
        # Open and stack the tiffs
        tiff_stack = []
        for path in tiffs:
            with rasterio.open(path) as src:
                tiff_stack.append(src.read(1))  # Read the first band
        # Convert to numpy array and reshape
        tiff_array = np.array(tiff_stack)
        # &&& modify channel dimension targeting t,h,w,c where c is 1
        reshaped_array = np.expand_dims(tiff_array, axis=-1)
        # - Before: `tiff_array.shape = (t, height, width)`
        # - After: `tiff_array.shape = (t, height, width, 1)`
        # &&& confirm this
        return reshaped_array
    def _make_safe_name(self, n, i, s):
        # pad the counter to maintain ordering
        zeroPaddedN = f"{n:0{4}d}"
        unsafe = f"{zeroPaddedN}_index_{i}_sigma_{s}"
        # need to drop decimals,invalid in feature names
        safe = unsafe.replace('.', '-')
        return safe
    def execute(self, bbox):
        eopatch = EOPatch(bbox=bbox)
        eopatch.timestamps = self._toDatetime(self.dates)
        eopatch.bbox = bbox
        n = 0 # make counter and add to name to force ordering
        for i in self.indices:
            for s in self.sigmas:
                name = self._make_safe_name(n, i, s)
                tiffs = select_tif_set(self.dates, [i], [s])
                array = self._toNParray(tiffs)
                eopatch[FeatureType.DATA, name] = array
                n = n + 1
        return eopatch
