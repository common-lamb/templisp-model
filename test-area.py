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
# *** Split samples into train test for TST
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
# *** Supervised training
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
from copy import deepcopy
from geopandas.plotting import plot_dataframe
import joblib
import itertools
import datetime

import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge as riomerge
import geopandas as gpd
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import shapely.validation
import shapely.geometry
from shapely.geometry import Polygon
from sentinelhub import DataCollection, UtmZoneSplitter

from fastai.torch_basics import set_seed
import lightgbm as lgb
from sklearn import metrics
from sklearn import preprocessing
from tsai.all import *

from eolearn.core.core_tasks import CreateEOPatchTask, RemoveFeatureTask
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
from eolearn.core.eonode import EONode # good
from eolearn.ml_tools import FractionSamplingTask
from eolearn.io import ImportFromTiffTask, ExportToTiffTask, VectorImportTask
from eolearn.features.extra.interpolation import LinearInterpolationTask
from eolearn.geometry import ErosionTask, VectorToRasterTask

######### ** Path and file selection
# input data
DATA_ROOT= pathlib.Path("/bulk-2/2023-package") # contains all input data dirs
# set expected dirs
DATA_AREAS = os.path.join(DATA_ROOT, "area_poly") # contains gpkg polygons that define study area
DATA_IDS = os.path.join(DATA_ROOT, "id_poly") # contains a gpkg with polygons whose attribute table defines their identity by integer
DATA_RASTERS = os.path.join(DATA_ROOT, "test-rasters") # contains the tif files named like: date_2023-07-31_index_blue_sigma-0.tif
DATA_TABLE= os.path.join(DATA_ROOT, "tabular") # contains a csv where columns are trait data, with a column matches the id_poly attribute table
# set expected files
DATA_train = os.path.join(DATA_AREAS, "test-AOI-north.gpkg")
DATA_validate = os.path.join(DATA_AREAS, "test-AOI-south.gpkg")
DATA_all =  os.path.join(DATA_AREAS, "test-AOI.gpkg")
DATA_ids = os.path.join(DATA_IDS, "identities.gpkg")
DATA_table = os.path.join(DATA_TABLE, "field-data.csv")

# intermediate and output data
DATA_OP_ROOT = os.path.join(DATA_ROOT, "..", "model_output")
EOPATCH_TRAIN_DIR= os.path.join(DATA_OP_ROOT, "eopatches_train")
EOPATCH_VALIDATE_DIR= os.path.join(DATA_OP_ROOT, "eopatches_validation")
DATA_DIR= os.path.join(DATA_OP_ROOT, "data")
EOPATCH_SAMPLES_DIR= os.path.join(DATA_OP_ROOT, "eopatches_sampled")
MODELS_DIR= os.path.join(DATA_OP_ROOT, "models")
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

SAMPLE_RATE = 0.50 # percentage of eopatches sampled for training. in (0.0-1.0)
TEST_PERCENTAGE = 0.20 # perventage of test-train set to use for testing. in (0.0-1.0)

NO_DATA_VALUE = -9999 #used for export, ensure no collision with expected data

######### ** Input validation
def dir_file_enforce():
    # check exists
    for d in (DATA_ROOT, DATA_AREAS, DATA_IDS, DATA_RASTERS, DATA_TABLE):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Input directory not found: {d}")
    for f in (DATA_train, DATA_validate, DATA_all, DATA_ids, DATA_table):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file not found: {f}")
    # make exist
    for d in (DATA_OP_ROOT, EOPATCH_TRAIN_DIR, EOPATCH_VALIDATE_DIR, DATA_DIR, EOPATCH_SAMPLES_DIR, RESULTS_DIR, MODELS_DIR):
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
        raise RuntimeError(f"The number of tifs is not the expected {expected_n_tifs}. Found: {len(tifs)}")

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
    for f in [DATA_train, DATA_validate, DATA_all, DATA_ids]:
        extent = gpd.read_file(f)
        crs = extent.crs
        if not crs == CRS:
            raise ValueError(f"Unexpected crs for file: {f} crs: {crs}")

# USER
dir_file_enforce()
validate_input_files()

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
    # get the width and height of the area in pixels
    width_pix = int(width / resolution)
    height_pix = int(height / resolution)
    # get the width and height of the area in patches
    width_patch = int(round(width_pix / grid_size))
    height_patch =  int(round(height_pix / grid_size))
    # length of patch edge m=(m/px)*px
    edge_m =resolution*grid_size

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
        # report on dimensions
        print(f"Dimension of the area is: {width:.0f} x {height:.0f} m")
        print(f"Dimension of the area is {width_pix} x {height_pix} pixels")
        print(f"Dimension of the area is {width_patch} x {height_patch} patches (rounded to nearest full patch)")
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

# USER
# show grid that will be used later
# test = area_grid(DATA_train, show=True)

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

def ask_loadgeotiffs(areas, eopatch_dir):
    print("TIME INTENSIVE JOB load all geotiffs into training eopatches stacked")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreateStackLoaderWorkflow(indicators=unique_tif_indicators(),
                                                            areas=areas,
                                                            eopatch_dir=eopatch_dir ))

# USER
ask_loadgeotiffs(areas=area_grid(DATA_train), eopatch_dir=EOPATCH_TRAIN_DIR)

######### ** Load timestamps etc

class AddTimestamps(EOTask):
    "Adds timestamps to an eopatch. uses the function unique_tif_indicators to identify times"
    def __init__(self, unique_tif_indicators=unique_tif_indicators()):
        self.indicators = unique_tif_indicators
        self.dates = self.indicators['dates']
    def _toDatetime(self, stringlist):
        fmt = '%Y-%m-%d'
        return [datetime.strptime(s, fmt) for s in stringlist]
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

# USER
# show the repairs that will be used later
# test = check_valid_geometry(gpd.read_file(DATA_ids), show=True)

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

def CreateDetailsLoaderWorkflow(areas, mask_file, observations, eopatch_dir):
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

    we know height is valid for test area
    '''

    # USER
    # add more rasterization tasks as needed
    rasterize_height_task = VectorToRasterTask(
        vector_feature, # as used in vector import task
        (FeatureType.DATA_TIMELESS, "HEIGHT"), #name of rasterized new layer, DATA floats
        values_column="HEIGHT-CM", # col of merged_gdf.columns to rasterize
        raster_shape=(FeatureType.MASK_TIMELESS, "IN_POLYGON"),
        raster_dtype=np.float32, # float
    )

    # initialize tasks or copy
    load_task = LoadTask(eopatch_dir)
    add_timestamps_task = AddTimestamps()
    make_areamask_task = MakeAreaMask(mask_file)
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

def ask_loadDetails(areas, mask_file, eopatch_dir):
    print("Load masks and timestamps to the patches after the gtiff stacks")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreateDetailsLoaderWorkflow(areas=areas,
                                                              observations=bind_observations(),
                                                              mask_file=mask_file,
                                                              eopatch_dir=eopatch_dir))

# USER
ask_loadDetails(areas=area_grid(DATA_train), mask_file=DATA_train, eopatch_dir=EOPATCH_TRAIN_DIR)

######### ** Visualize layers
#####
# *** Object contents

def verify_eopatch_loaded():

    eopatch = EOPatch.load(os.path.join(EOPATCH_TRAIN_DIR, 'eopatch_0'))
    eopatch

    print(f"timestamps: {eopatch.timestamps}")
    data_keys = sorted(list(eopatch.data.keys()))
    print(f"data_keys: {data_keys}")

    #####
    # *** RGB per time
    eopatch.plot((FeatureType.DATA, data_keys[-1])) # all across time axis of one feature, high range can make features look "flattened"

    #####
    # *** Reference identities map
    eopatch.plot((FeatureType.MASK_TIMELESS, 'IN_POLYGON')) # the aoi masking polygon

    # *** Rasterized observations
    eopatch.plot((FeatureType.DATA_TIMELESS, 'HEIGHT')) # trait raster

# USER
verify_eopatch_loaded()

################
# * Prepare eopatch
################

def CreatePatchPrepWorkflow(areas, eopatch_dir, eopatch_out_dir, trait, sample_rate=None):
    "Creates a workflow to finalize and sample eopatches. "

    #use current global
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

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

    save_task = SaveTask(eopatch_out_dir, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

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
                workflow_nodes[-2]: {"seed": RNDM}, # sample task
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

# USER
ask_preparePatches()

################
# * Create training data
################

######### ** Extract eopatches
def sampledData(areas, eopatch_samples_dir, trait, show=False):
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

    if show:
        print("sampledData")
        print(f"features.shape: {features.shape}")
        print(f"labels.shape: {labels.shape}")

    return features, labels

######### ** Reshape data

# *** Shape for TSP
def reshape_eopatch_to_TSAI(data, show=False):
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

    if show:
        print("reshape to TSAI")
        print(f"features.shape: {features_reshaped.shape}")
        print(f"labels.shape: {labels_reshaped.shape}")

    return features_reshaped, labels_reshaped

# *** Shape for GBM
def reshape_to_GBM(data, TSAI_shape=True, show=False):
    """
    from TSAI as s,v,t
    or from eopatch as t,w,h,f
    to
    GBM requires data as n, m
    where n is or w*h (ie. s) and m is t*f
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

    if show:
        print("reshape to GBM")
        print(f"features.shape: {features_reshaped.shape}")
        print(f"labels.shape: {labels_reshaped.shape}")

    return features_reshaped, labels_reshaped

#####
# *** Split samples into train test sets
#####

def split_for_TSAI(data, test_percentage=TEST_PERCENTAGE, show=False):
    "Takes eolearn  shaped features and labels, returns X,Y,splits shaped for TSAI"
    features, labels = reshape_eopatch_to_TSAI(data)
    splits = get_splits(labels, valid_size=test_percentage, stratify=True, random_state=RNDM, shuffle=True, show_plot=show)
    if show:
        print("split for TSAI")
        print(f"features.shape: {features.shape}")
        print(f"labels.shape: {labels.shape}")
        print(f"len split 0 : {len(sS[0])}")
        print(f" len split 1: {len(sS[1])}")

    return features, labels, splits

def split_reconfigure_for_GBM(split_data, show=False):
    "Takes TSAI shaped X,Y,splits, and returns x_train, y_train, x_test, y_test shaped for GBM"
    features, labels, splits = split_data

    split_train = splits[0]
    mask_train = np.zeros(len(features), dtype=bool)
    mask_train[split_train] = True
    x_train = features[mask_train]
    y_train = labels[mask_train]
    data_train = x_train, y_train

    split_test = splits[1]
    mask_test = np.zeros(len(features), dtype=bool)
    mask_test[split_test] = True
    x_test =features[mask_test]
    y_test =labels[mask_test]
    data_test = x_test, y_test

    x_train_GBM, y_train_GBM = reshape_to_GBM(data=data_train)
    x_test_GBM, y_test_GBM = reshape_to_GBM(data=data_test)
    if show:
        print("splits reconfigure for GBM")
        print(f"x_train: {x_train_GBM.shape}")
        print(f"y_train: {y_train_GBM.shape}")
        print(f"x_test: {x_test_GBM.shape}")
        print(f"y_test: {y_test_GBM.shape}")

    return x_train_GBM, y_train_GBM, x_test_GBM, y_test_GBM

# * GBM experiment
################

def create_GBM_training_data(trait_name, show=False):
    a = sampledData(areas=area_grid(DATA_train),
                    eopatch_samples_dir=EOPATCH_SAMPLES_DIR,
                    trait = trait_name)
    b = split_for_TSAI(a)
    c = split_reconfigure_for_GBM(b)
    if show:
        x_train, y_train, x_test, y_test = c
        print("GBM training data")
        print(f"x_train: {x_train.shape}") # (1349, 450)
        print(f"y_train: {y_train.shape}") # (1349,)
        print(f"x_test: {x_test.shape}") # (337, 450)
        print(f"y_test: {y_test.shape}") # (337,)
    return c

def create_TSAI_training_data(trait_name, show=False):
    a = sampledData(areas=area_grid(DATA_train),
                    eopatch_samples_dir=EOPATCH_SAMPLES_DIR,
                    trait = trait_name)
    b = split_for_TSAI(a)
    if show:
        x, y, splits = b
        print("TSAI training data")
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")
        print(f"splits train: {len(splits[0])}")
        print(f"splits test: {len(splits[1])}")
    return b

######### ** Train

def trainGBM(objective,
             area_name,
             trait_name,
             model_type,
             x_train_GBM,
             y_train_GBM,):

    learning_rate=0.1
    # count training classes for classification arg
    n_labels_unique = len(np.unique(y_train_GBM))
    # count predictions for ranking arg
    group_all = [len(x_train_GBM)]

    # Set up the model
    # metric options: https://lightgbm.readthedocs.io/en/stable/Parameters.html#metric
    if objective == 'multiclass':
        model = lgb.LGBMClassifier(objective=objective, num_class=n_labels_unique, metric="multi_logloss",learning_rate=learning_rate, random_state=RNDM)
        model.fit(x_train_GBM, y_train_GBM)
    elif objective == 'regression':
        model = lgb.LGBMRegressor(objective=objective, metric="mean_absolute_error",learning_rate=learning_rate, random_state=RNDM)
        model.fit(x_train_GBM, y_train_GBM)
    # elif objective == 'ranking':
        # # not tested on ranked data
        # model = lgb.LGBMRanker(objective=objective, metric="ndcg",learning_rate=learning_rate, random_state=RNDM)
        # # must set the group(s) https://github.com/microsoft/LightGBM/issues/4808#issuecomment-1219044835
        # model.fit(x_train_GBM, y_train_GBM, group=group_all)
    else:
        raise ValueError("objective not recognized")

    # Train the model
    # Save the model
    joblib.dump(model, os.path.join(MODELS_DIR, f"{area_name}-{trait_name}-{objective}-{model_type}.pkl"))

def ask_trainGBM():
    print("train GBM model?")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        x_train_GBM, y_train_GBM, x_test_GBM, y_test_GBM = create_GBM_training_data(trait_name='HEIGHT')
        trainGBM(objective='multiclass',
                 area_name='test-area',
                 trait_name='HEIGHT',
                 model_type='GBM',
                 x_train_GBM=x_train_GBM,
                 y_train_GBM=y_train_GBM,)
        trainGBM(objective='regression',
                 area_name='test-area',
                 trait_name='HEIGHT',
                 model_type='GBM',
                 x_train_GBM=x_train_GBM,
                 y_train_GBM=y_train_GBM,)

# USER
ask_trainGBM()


######### ** Validate

def loadModel(area_name, trait_name, objective, model_type):
    "Loads trained GBM and TSAI models from disk."
    identifier = f"{area_name}-{trait_name}-{objective}-{model_type}.pkl"
    model_path = os.path.join(MODELS_DIR, identifier)

    if model_type == "GBM":
        # Load the model
        model = joblib.load(model_path)
    elif model_type == "TSAI":
        model = load_learner(model_path, cpu=False)
    else:
        raise ValueError(f'Model type ({model_type}) not recognized')
    return model

def predict_testSet(x_testSet, area_name, trait_name, objective, model_type, show=False):
    "loads model and predicts y_predicted"
    model = loadModel(area_name=area_name, trait_name=trait_name, objective=objective, model_type=model_type)
    # Predict the test labels
    if model_type == "GBM":
        predicted_labels_test = model.predict(x_testSet)
        return predicted_labels_test, model
    elif model_type == "TSAI":
        # Labelled data
        # learn = model
        # dls = learn.dls
        # valid_dl = dls.valid
        # test_ds = valid_dl.dataset.add_test(x_testSet, y_testSet)
        # test_dl = valid_dl.new(test_ds)
        # test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True)
        #
        # Unlabelled data
        learn = model
        dls = learn.dls
        valid_dl = dls.valid
        test_ds = dls.dataset.add_test(x_testSet)
        test_dl = valid_dl.new(test_ds)
        print(x_testSet.shape)

        test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True)
        #
        # get_preds returns a tuple of three elements: (predictions, targets, decoded)
        #   - `predictions` are the raw outputs from the model
        #     - For binary classification, it returns the probability of the positive class.
        #     - For multi-class classification, it returns probabilities for each class.
        #   - `targets` are the actual labels (if available in the dataset)
        #   - `decoded` contains the decoded predictions (e.g., class labels for classification tasks)
        if show:
            test_probas.numpy().shape
            test_targets.numpy().shape
            test_preds.numpy().shape
        return test_preds.numpy(), learn
    else:
        raise ValueError(f'Model type ({model_type}) not recognized')

#####
# *** F1 etc table

def report_Metrics_Classification(y_test, predicted_labels_test, class_names, model_type, testset_name, trait_name, pred_type):

    class_labels_drop = y_test[~np.isnan(y_test)]
    class_labels_float = np.unique(class_labels_drop)
    class_labels = [int(x) for x in class_labels_float]

    #handle unexpected class names
    if len(class_names) != len(class_labels):
        print("Alert: unexpected length of class labels. setting class names to be found values")
        print(f"expected: n: {len(class_names)} class names: {class_names} ")
        print(f"found: n: {len(class_labels)} class labels: {class_labels} ")
        class_names = class_labels


    mask = np.in1d(predicted_labels_test, y_test)
    predictions = predicted_labels_test[mask]
    true_labels = y_test[mask]

    # Extract and display metrics
    accuracy = metrics.accuracy_score(true_labels, predictions)

    avg_f1_score = metrics.f1_score(true_labels, predictions, average="weighted")

    f1_scores = metrics.f1_score(true_labels, predictions, labels=class_labels, average=None)
    recall = metrics.recall_score(true_labels, predictions, labels=class_labels, average=None)
    precision = metrics.precision_score(true_labels, predictions, labels=class_labels, average=None)

    print("")
    print (f"Classification Metrics\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}")
    print("---------------------------------")
    print("Classification accuracy {:.1f}%".format(100 * accuracy))
    print("Classification F1-score {:.1f}%".format(100 * avg_f1_score))
    print("---------------------------------")
    print("")
    print("             Class              =  F1  | Recall | Precision")
    print("         --------------------------------------------------")
    for idx in range(len(class_labels)):
        name = str(class_names[idx])
        line_data = (name, f1_scores[idx] * 100, recall[idx] * 100, precision[idx] * 100)
        print("         * {0:20s} = {1:2.1f} |  {2:2.1f}  | {3:2.1f}".format(*line_data))
    print("         --------------------------------------------------")
    print("")

def report_Metrics_Regression(y_test, predicted_values_test, model_type, testset_name, trait_name, pred_type):

    # drop values from both where nan
    mask = ~np.isnan(y_test)
    y_test = y_test[mask]
    predicted_values_test = predicted_values_test[mask]

    # Calculate metrics
    mse = metrics.mean_squared_error(y_test, predicted_values_test)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, predicted_values_test)
    r2 = metrics.r2_score(y_test, predicted_values_test)

    print("")
    print(f"Regression Metrics\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}")
    print("---------------------------------")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("---------------------------------")

#####
# *** Confusion matrices

def plot_confusion_matrix(confusion_matrix, classes, title, normalize=False, ylabel="True label", xlabel="Predicted label"):
    """
    prints and plots one confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        normalisation_factor = confusion_matrix.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps
        confusion_matrix = confusion_matrix.astype("float") / normalisation_factor

    np.set_printoptions(precision=2, suppress=True)

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(title, fontsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    threshold = confusion_matrix.max() / 2.0
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i,
            format(confusion_matrix[i, j], ".2f" if normalize else "d"),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > threshold else "black",
            fontsize=12,
        )
    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)

def show_std_T_confusionMatrix(predicted_labels_test,
                               y_test,
                               trait_name,
                               model_type,
                               testset_name,
                               pred_type,
                               class_names):
    "plots standard and transposed confusion matrix"

    # keep the array locations only where prediction set values are in the test set values
    mask = np.in1d(predicted_labels_test, y_test)
    predictions = predicted_labels_test[mask]
    true_labels = y_test[mask]

    # get unique labels from the test set
    class_labels_drop = y_test[~np.isnan(y_test)]
    class_labels_float = np.unique(class_labels_drop)
    class_labels = [int(x) for x in class_labels_float]

    if len(class_names) != len(class_labels):
        #handle unexpected num of class names
        print("Alert: unexpected length of class labels. setting class names to be found values")
        print(f"expected: n: {len(class_names)} class names: {class_names} ")
        print(f"found: n: {len(class_labels)} class labels: {class_labels} ")
        class_names = class_labels


    fig = plt.figure(figsize=(20, 20))

    plt.subplot(1, 2, 1)
    plot_confusion_matrix(
        metrics.confusion_matrix(true_labels, predictions),
        classes=class_names,
        normalize=True,
        ylabel="Ground Truth",
        xlabel="Predicted",
        title= f"Confusion Matrix\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}")

    plt.subplot(1, 2, 2)
    plot_confusion_matrix(
        metrics.confusion_matrix(predictions, true_labels),
        classes=class_names,
        normalize=True,
        xlabel="Ground Truth",
        ylabel="Predicted",
        title=f"Transposed Confusion Matrix\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}")

    plt.tight_layout()
    plt.show()

def plot_regression_results(
    y_true,
    y_pred,
    title,
    ylabel="Predicted Values",
    xlabel="True Values"):
    """
    Plots regression results including a scatter plot and error histogram.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.set_title(f"Predictions\n{title}", fontsize=20)

    # Error histogram
    errors = y_pred - y_true
    ax2.hist(errors, bins=30, edgecolor='black')
    ax2.set_xlabel("Prediction Error", fontsize=15)
    ax2.set_ylabel("Frequency", fontsize=15)
    ax2.set_title(f"Error Distribution\n{title}", fontsize=20)

    plt.tight_layout()
    plt.show()

def show_regression_results(predicted_values_test,
                            y_test,
                            trait_name,
                            model_type,
                            testset_name,
                            pred_type):
    """Plots regression results"""

    # drop values from both where nan
    mask = ~np.isnan(y_test)
    y_test = y_test[mask]
    predicted_values_test = predicted_values_test[mask]

    n_samples = len(y_test)
    if n_samples > 500:
        indices = np.random.choice(n_samples, 500, replace=False)
        y_test_sampled = y_test[indices]
        predicted_values_test_sampled = predicted_values_test[indices]
        print(f"Generating plot with {len(y_test_sampled)} selected points.")

    else:
        y_test_sampled = y_test
        predicted_values_test_sampled = predicted_values_test
        print(f"Generating plot with all {len(y_test_sampled)} points.")

    plot_regression_results(
        y_true=y_test_sampled,
        y_pred=predicted_values_test_sampled,
        ylabel="Predicted Values",
        xlabel="True Values",
        title=f"model: {model_type} test_set: {testset_name} trait: {trait_name} prediction: {pred_type}")

#####
# *** Class balance

def show_ClassBalance(y_train, class_names, model_type, testset_name, trait_name, pred_type):
    "Plot class balance for training data"

    # get unique labels from the train set
    class_labels_drop = y_train[~np.isnan(y_train)]
    class_labels_float = np.unique(class_labels_drop)
    class_labels = [int(x) for x in class_labels_float]

    # replace class_names if unexpected count
    if len(class_names) != len(class_labels):
        #handle unexpected num of class names
        print("Alert: unexpected length of class labels. setting class names to be found values")
        print(f"expected: n: {len(class_names)} class names: {class_names} ")
        print(f"found: n: {len(class_labels)} class labels: {class_labels} ")
        class_names = class_labels

    fig = plt.figure(figsize=(20, 5))
    label_ids, label_counts = np.unique(y_train, return_counts=True)
    label_ids = [int(x) for x in label_ids]
    plt.bar(range(len(label_ids)), label_counts)
    plt.xticks(range(len(class_names)),
               class_names,
               rotation=0,
               fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"Training Data Class Balance\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}", fontsize=20)
    plt.show()

def show_ValueBalance(y_train, class_names, model_type, testset_name, trait_name, pred_type):
    ""
    # get unique labels from the test set
    class_labels_drop = y_train[~np.isnan(y_train)]
    class_labels_float = np.unique(class_labels_drop)
    class_labels = [int(x) for x in class_labels_float]

    # replace class_names if unexpected count
    if len(class_names) != len(class_labels):
        #handle unexpected num of class names
        print("Alert: unexpected length of class labels. setting class names to be found values")
        print(f"expected: n: {len(class_names)} class names: {class_names} ")
        print(f"found: n: {len(class_labels)} class labels: {class_labels} ")
        class_names = class_labels

    # histogram
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.hist(y_train, bins=30, edgecolor='black')
    # ax.set_xticks(fontsize=15)
    # ax.set_yticks(fontsize=15)
    ax.set_xlabel("Trait Values", fontsize=15)
    ax.set_ylabel("Frequency", fontsize=15)
    ax.set_title(f"Training Data Value Distribution\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}", fontsize=20)

    plt.tight_layout()
    plt.show()



#####
# *** ROC and AUC

def show_ROCAUC(model_GBM, class_names, y_test_GBM, y_train_GBM, x_test_GBM, model_type, testset_name, trait_name, pred_type):

    class_labels = np.unique(np.hstack([y_test_GBM, y_train_GBM]))
    labels_binarized = preprocessing.label_binarize(y_test_GBM, classes=class_labels)
    scores_test = model_GBM.predict_proba(x_test_GBM)
    colors = plt.cm.Set1.colors

    # replace class_names if unexpected count
    if len(class_names) != len(class_labels):
        #handle unexpected num of class names
        print("Alert: unexpected length of class labels. setting class names to be found values")
        print(f"expected: n: {len(class_names)} class names: {class_names} ")
        print(f"found: n: {len(class_labels)} class labels: {class_labels} ")
        class_names = class_labels

    fpr, tpr, roc_auc = {}, {}, {}
    for idx, _ in enumerate(class_labels):
        fpr[idx], tpr[idx], _ = metrics.roc_curve(labels_binarized[:, idx], scores_test[:, idx])
        roc_auc[idx] = metrics.auc(fpr[idx], tpr[idx])

    plt.figure(figsize=(20, 10))
    for idx, lbl in enumerate(class_labels):
        if np.isnan(roc_auc[idx]):
            continue
        plt.plot(
            fpr[idx],
            tpr[idx],
            color=colors[idx],
            lw=2,
            label= f"{class_names[idx]}" + " ({:0.5f})".format(roc_auc[idx]),
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 0.99])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"ROC Curve\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}", fontsize=20)
    plt.legend(loc="center right", prop={"size": 15})
    plt.show()

#####
# *** Feature importance

def show_featureImportance(model_GBM, feature_names, t_dim, f_dim, model_type, testset_name, trait_name, pred_type):
    "plots a heatmap of time and feature showing contribution of each to predictions "

    # Get feature importances and reshape them to dates and features
    feature_importances = model_GBM.feature_importances_.reshape((t_dim, f_dim))

    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()

    # Plot the importances
    im = ax.imshow(feature_importances, aspect=1.0)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=15)
    plt.yticks(range(t_dim), [f"t {i}" for i in range(t_dim)], fontsize=15)
    plt.xlabel(f"\nSpectral Bands and Sigma of Gaussian Space", fontsize=15)
    plt.ylabel("Time", fontsize=15)
    plt.title(f"Feature Importance\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}", fontsize=20)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    fig.subplots_adjust(wspace=0, hspace=0)
    cb = fig.colorbar(im, ax=[ax], orientation="horizontal", pad=0.01, aspect=100)
    cb.ax.tick_params(labelsize=20)
    plt.show()

def testset_predict_validate(trait_name, area_name, objective, model_type, testset_name, class_names):
    """
    collects sampled data, predicts and then reports metrics

    trait_name: str identifies trait in patches eg 'HEIGHT'
    area_name: str descibes area being studied eg 'test-area'
    objective: training objective, one of 'multiclass'  'regression', 'ranking'
    model_type: model type one of 'GBM', 'TSAI'
    class_names: list of str names for classes which were predicted
    """
    # get dims
    f, l = sampledData(areas=area_grid(DATA_train), eopatch_samples_dir=EOPATCH_SAMPLES_DIR, trait = trait_name)
    t_dim = f.shape[0] #time_dimension
    f_dim = f.shape[-1] #features_dimension
    # get feature names
    feature_names = [f"{i} s:{s}" for i in unique_tif_indicators()['indices'] for s in unique_tif_indicators()['sigmas']]

    GBM_flag = model_type == 'GBM'
    TSAI_flag = model_type == 'TSAI'
    regression_flag = objective == 'regression'
    multiclass_flag = objective == 'multiclass'

    # get prediction data
    if GBM_flag:
        x_train_GBM, y_train_GBM, x_test_GBM, y_test_GBM = create_GBM_training_data(trait_name=trait_name)
        predicted_test, model = predict_testSet(x_testSet=x_test_GBM, area_name=area_name, trait_name=trait_name, objective=objective, model_type=model_type)
        # connect
        x_train = x_train_GBM
        y_train = y_train_GBM
        x_test = x_test_GBM
        y_test = y_test_GBM
    elif TSAI_flag:
        x_all_TSAI, y_all_TSAI, splits = create_TSAI_training_data(trait_name=trait_name)
        # splits to x/y test
        x_train_TSAI = x_all_TSAI[splits[0]]
        y_train_TSAI= y_all_TSAI[splits[0]]
        x_test_TSAI= x_all_TSAI[splits[1]]
        y_test_TSAI= y_all_TSAI[splits[1]]
        predicted_test, model=predict_testSet(x_testSet=x_test_TSAI, area_name=area_name, trait_name=trait_name, objective=objective, model_type=model_type)
        # connect
        x_train = x_train_TSAI
        y_train = y_train_TSAI
        x_test = x_test_TSAI
        y_test = y_test_TSAI
    else:
        raise ValueError('Model type not recognized')

    if (GBM_flag or TSAI_flag ) and (multiclass_flag):
        show_ClassBalance(y_train=y_train,
                          class_names=class_names,
                          model_type=model_type,
                          testset_name=testset_name,
                          trait_name=trait_name,
                          pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (regression_flag):
        show_ValueBalance(y_train=y_train,
                          class_names=class_names,
                          model_type=model_type,
                          testset_name=testset_name,
                          trait_name=trait_name,
                          pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (multiclass_flag):
        report_Metrics_Classification(
            y_test=y_test,
            predicted_labels_test=predicted_test,
            class_names=class_names,
            model_type=model_type,
            testset_name=testset_name,
            trait_name=trait_name,
            pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (regression_flag):
        report_Metrics_Regression(
            y_test=y_test,
            predicted_values_test=predicted_test,
            model_type=model_type,
            testset_name=testset_name,
            trait_name=trait_name,
            pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (multiclass_flag):
        show_std_T_confusionMatrix(
            predicted_labels_test=predicted_test,
            y_test=y_test,
            trait_name=trait_name,
            class_names=class_names,
            model_type=model_type,
            testset_name=testset_name,
            pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (regression_flag):
        show_regression_results(
            predicted_values_test=predicted_test,
            y_test=y_test,
            trait_name=trait_name,
            model_type=model_type,
            testset_name=testset_name,
            pred_type=objective)

    if (GBM_flag) and (multiclass_flag):
        show_ROCAUC(
            model_GBM=model,
            class_names=class_names,
            y_test_GBM=y_test,
            y_train_GBM=y_train,
            x_test_GBM=x_test,
            model_type=model_type,
            testset_name=testset_name,
            trait_name=trait_name,
            pred_type=objective)

    if (GBM_flag) and (multiclass_flag):
        show_featureImportance(
            model_GBM=model,
            feature_names=feature_names,
            t_dim=t_dim,
            f_dim=f_dim,
            model_type=model_type,
            testset_name=testset_name,
            trait_name=trait_name,
            pred_type=objective)

# USER
testset_predict_validate(trait_name='HEIGHT', area_name='test-area', objective='multiclass', model_type='GBM', testset_name='holdout', class_names=['black','white'])
testset_predict_validate(trait_name='HEIGHT', area_name='test-area', objective='regression', model_type='GBM', testset_name='holdout', class_names=['black','white'])

######### ** Predict

# USER
# test = area_grid(DATA_validate, show=True)
# prepare eopatches for the validation area
ask_loadgeotiffs(areas=area_grid(DATA_validate), eopatch_dir=EOPATCH_VALIDATE_DIR)
ask_loadDetails(areas=area_grid(DATA_validate), mask_file=DATA_validate, eopatch_dir=EOPATCH_VALIDATE_DIR)

def verify_validation_eopatch():
    ""
    eopatch = EOPatch.load(os.path.join(EOPATCH_VALIDATE_DIR, 'eopatch_0'))
    eopatch

    eopatch.plot((FeatureType.MASK_TIMELESS, 'IN_POLYGON'))

verify_validation_eopatch()

class PredictPatchTask(EOTask):
    """
    Make model predictions on a patch. Optionally include probas
    """

    def __init__(self, model, model_type, feature, predicted_trait_name, predicted_probas_name=None):
        self.model = model
        self.model_type = model_type
        self.feature = feature
        self.predicted_trait_name = predicted_trait_name
        self.predicted_probas_name = predicted_probas_name

    def execute(self, eopatch):
        features = eopatch[self.feature]
        t, w, h, f = features.shape
        #make fake labels array of w,h,1
        fake_labels = np.zeros((w,h,1))
        # make tsai data
        datatoTSAI = features, fake_labels
        data_TSAI = reshape_eopatch_to_TSAI(data=datatoTSAI, show=False)
        features_TSAI, fake_labels_back = data_TSAI
        # make gbm data
        data_GBM = reshape_to_GBM(data=data_TSAI)
        features_GBM, fake_labels_back = data_GBM
        del fake_labels_back

        if self.model_type == 'GBM':
            #get GBM prediction
            predicted_trait= self.model.predict(features_GBM)
            # reshape
            predicted_trait= predicted_trait.reshape(w, h)
            predicted_trait= predicted_trait[..., np.newaxis]
            eopatch[(FeatureType.DATA_TIMELESS, self.predicted_trait_name)] = predicted_trait
            # get probas
            if self.predicted_probas_name:
                predicted_scores = self.model.predict_proba(features_GBM)
                # reshape probas
                _, d = predicted_scores.shape
                predicted_scores = predicted_scores.reshape(w, h, d)
                eopatch[(FeatureType.DATA_TIMELESS, self.predicted_probas_name)] = predicted_scores
            # result
            return eopatch
        elif self.model_type == 'TSAI':
            #get TSAI prediction
            learn = self.model
            dls = learn.dls
            valid_dl = dls.valid
            test_ds = dls.dataset.add_test(features_TSAI)
            test_dl = valid_dl.new(test_ds)
            probas, targets, preds = learn.get_preds(dl=test_dl, with_decoded=True)
            # reshape
            preds_array = preds.numpy()
            predicted_trait= preds_array.reshape(w, h)
            predicted_trait= predicted_trait[..., np.newaxis]
            eopatch[(FeatureType.DATA_TIMELESS, self.predicted_trait_name)] = predicted_trait
            # reshape probas
            if self.predicted_probas_name:
                predicted_scores = probas.numpy()
                _, d = predicted_scores.shape
                predicted_scores = predicted_scores.reshape(w, h, d)
                eopatch[(FeatureType.DATA_TIMELESS, self.predicted_probas_name)] = predicted_scores
            # result
            return eopatch
        else:
            raise ValueError('Model type not recognized')

def CreatePredictionWorkflow(areas, eopatch_dir, area_name, trait_name, objective, model_type):
    "Creates a workflow to predict trait on validation eopatches. "
    model = loadModel(area_name=area_name, trait_name=trait_name, objective=objective, model_type=model_type)
    eopatch = EOPatch.load(os.path.join(eopatch_dir, 'eopatch_0'))
    # drop FEATURES_TRAINING if it exists
    data_keys = sorted(list(eopatch.data.keys()))
    data_keys = sorted(set(data_keys) - set(['FEATURES_TRAINING']))

    load_task = LoadTask(eopatch_dir)

    ######### ** Concatenate
    concatenate_task = MergeFeatureTask({FeatureType.DATA: data_keys}, (FeatureType.DATA, "FEATURES_TRAINING"))

    # predict
    if model_type == 'GBM' and objective == 'regression': #GBM regression has no probas
        predict_task = PredictPatchTask(model=model,
                                        model_type=model_type,
                                        feature=(FeatureType.DATA, "FEATURES_TRAINING"),
                                        predicted_trait_name=f"PREDICTED_{trait_name}_{objective}_{model_type}")
    else:
        predict_task = PredictPatchTask(model=model,
                                        model_type=model_type,
                                        feature=(FeatureType.DATA, "FEATURES_TRAINING"),
                                        predicted_trait_name=f"PREDICTED_{trait_name}_{objective}_{model_type}",
                                        predicted_probas_name=f"PREDICTED_{trait_name}_{objective}_{model_type}_PROBA")

    save_task = SaveTask(eopatch_dir, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    # node list
    workflow_nodes = linearly_connect_tasks(load_task, concatenate_task, predict_task, save_task)

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

def ask_PredictPatches_GBM():
    print("predict validation area EOPatches?")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreatePredictionWorkflow(areas=area_grid(DATA_validate),
                                                           eopatch_dir=EOPATCH_VALIDATE_DIR,
                                                           area_name='test-area',
                                                           trait_name='HEIGHT',
                                                           objective='multiclass',
                                                           model_type = 'GBM'))
        execute_prepared_workflow(CreatePredictionWorkflow(areas=area_grid(DATA_validate),
                                                           eopatch_dir=EOPATCH_VALIDATE_DIR,
                                                           area_name='test-area',
                                                           trait_name='HEIGHT',
                                                           objective='regression',
                                                           model_type = 'GBM'))

# USER
ask_PredictPatches_GBM()

#####
# *** visualize prediction

def verify_predictions_GBM():
    eopatch = EOPatch.load(os.path.join(EOPATCH_VALIDATE_DIR, 'eopatch_0'))
    eopatch
    eopatch.plot((FeatureType.DATA_TIMELESS, 'PREDICTED_HEIGHT_multiclass_GBM'))
    eopatch.plot((FeatureType.DATA_TIMELESS, 'PREDICTED_HEIGHT_multiclass_GBM_PROBA'))
    eopatch.plot((FeatureType.DATA_TIMELESS, 'PREDICTED_HEIGHT_regression_GBM'))

# USER
verify_predictions_GBM()

######### ** Quantify prediction
#####
# *** Visualize predicted trait

def cartesian_from_position(position, grid_h, grid_w):
    "convert between the grid ordering of eopatches and matplotlib axes coordinates"
    # area is segmented into n=row*col sections, with origin lower left, iterating rows fast ie: for cols(for rows), numbered 0...n-1
    # plt grids are cartesian (row, col) with origin upper left
    def invertPosition(axis_pos, axis_len):
        lastPos = axis_len -1
        invertedPos = lastPos - axis_pos
        return invertedPos
    rowInverted = position % grid_h # modulo of height gives row
    row = invertPosition(rowInverted, grid_h) # flip to account for opposite cutting order
    col = position // grid_h # int div of height gives col
    return row, col

def plot_prediction(grid_h, grid_w, trait_name, areas, model_type, testset_name, pred_type):
    ""
    fig, axs = plt.subplots(nrows=grid_h+1, ncols=grid_w+1, figsize=(20, 25))
    for i in tqdm(range(len(areas))):
        eopatch_path = os.path.join(EOPATCH_VALIDATE_DIR, f"eopatch_{i}")
        eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
        row, col = cartesian_from_position(i, grid_h, grid_w)
        ax = axs[row][col]
        im = ax.imshow(eopatch.data_timeless[f"PREDICTED_{trait_name}_{pred_type}_{model_type}"].squeeze())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        del eopatch

    fig.subplots_adjust(wspace=0, hspace=0)

    cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
    cb.ax.tick_params(labelsize=20)
    plt.title(f"Prediction: model: {model_type} test-set: {testset_name} trait {trait_name} prediction {pred_type}", fontsize=20)
    plt.show()

# USER
plot_prediction(grid_h = 1, grid_w = 2, trait_name = 'HEIGHT', model_type='GBM', testset_name='transfer', pred_type='multiclass', areas=area_grid(DATA_validate))
plot_prediction(grid_h = 1, grid_w = 2, trait_name = 'HEIGHT', model_type='GBM', testset_name='transfer', pred_type='regression', areas=area_grid(DATA_validate))

#####
# *** Visualize trait diff

def plot_disagreement(areas, trait_name, inspect_ratio, model_type, testset_name, pred_type):
    "plot ground truth, prediction, categorical and continuous agreement"

    #pick rndm patch
    idx = np.random.choice(range(len(areas)))
    eopatch = EOPatch.load(os.path.join(EOPATCH_VALIDATE_DIR, f"eopatch_{idx}"), lazy_loading=True)
    #set size of inspection window
    w, h = eopatch.data_timeless[trait_name].squeeze().shape
    inspect_ratio = min (inspect_ratio, 1)
    smallest_side = min(w, h)
    inspect_size = math.floor(smallest_side * inspect_ratio)
    w_min = np.random.choice(range(w - inspect_size))
    w_max = w_min + inspect_size
    h_min = np.random.choice(range(h - inspect_size))
    h_max = h_min + inspect_size
    # compose identifier
    identifier = f"PREDICTED_{trait_name}_{pred_type}_{model_type}"
    # aoi logical mask
    in_poly = eopatch.mask_timeless["IN_POLYGON"].squeeze()
    mask = ~in_poly

    # Draw the Reference map
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Spatial Prediction\nmodel: {model_type} test-set: {testset_name} trait: {trait_name} prediction: {pred_type}", fontsize=20)

    # trait
    ax = plt.subplot(2, 2, 1)
    data = eopatch.data_timeless[trait_name].squeeze()
    masked = np.ma.masked_where(mask, data)
    plt.imshow(masked[w_min:w_max, h_min:h_max])
    plt.colorbar(label=f"{trait_name}")
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect("auto")
    plt.title(f"Ground Truth: {trait_name}", fontsize=15)

    #prediction
    ax = plt.subplot(2, 2, 2)
    data = eopatch.data_timeless[identifier].squeeze()
    masked = np.ma.masked_where(mask, data)
    plt.imshow(masked[w_min:w_max, h_min:h_max])
    plt.colorbar(label=f"{trait_name}")
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect("auto")
    plt.title("Prediction", fontsize=15)

    #disagreement logical
    ax = plt.subplot(2, 2, 3)
    pred =  eopatch.data_timeless[identifier].squeeze()
    true = eopatch.data_timeless[trait_name].squeeze()
    if pred_type == 'multiclass':
        dis_type = "logical"
        dis_amnt = "difference"
        data = pred != true
    elif pred_type == 'regression':
        threshold = 50
        dis_type = "percentile"
        dis_amnt = threshold
        difference = np.abs(pred - true)
        percentile = np.percentile(difference, threshold)
        data = difference > percentile
    else:
        raise ValueError(f"pred_type not recognized ({pred_type})")

    masked = np.ma.masked_where(mask, data)
    cmap = plt.cm.colors.ListedColormap(['green', 'red'])
    plt.imshow(masked[w_min:w_max, h_min:h_max], cmap=cmap)
    plt.legend([plt.Rectangle((0,0),1,1,fc='red'), plt.Rectangle((0,0),1,1,fc='green')],
              ['Disagree', 'Agree'], loc='lower right')
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect("auto")
    plt.title(f"Disagreement ({dis_type} {dis_amnt})", fontsize=15)

    # disagreement quantity
    ax = plt.subplot(2, 2, 4)
    data = eopatch.data_timeless[identifier].squeeze() - eopatch.data_timeless[trait_name].squeeze()
    masked = np.ma.masked_where(mask, data)
    vmax = max(abs(masked.min()), abs(masked.max()))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    plt.imshow(masked[w_min:w_max, h_min:h_max], cmap="seismic", norm=norm)
    plt.colorbar(label="Difference")
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect("auto")
    plt.title("Difference", fontsize=15)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# USER
plot_disagreement(trait_name = 'HEIGHT', areas = area_grid(DATA_validate), inspect_ratio=0.99, model_type='GBM', testset_name='transfer', pred_type="multiclass")
plot_disagreement(trait_name = 'HEIGHT', areas = area_grid(DATA_validate), inspect_ratio=0.99, model_type='GBM', testset_name='transfer', pred_type="regression")

#####
# *** Quantify agreement
def predictedData(areas, eopatch_samples_dir, trait_name, objective, model_type, show=False):
    """
    Takes grid of areas, a source of eopatches, and a single trait.
    Concatenates all then Returns features and trait and prediction
    """

    sampled_eopatches = []
    for i in range(len(areas)):
        sample_path = os.path.join(eopatch_samples_dir, f"eopatch_{i}")
        sampled_eopatches.append(EOPatch.load(sample_path, lazy_loading=True))

    features = np.concatenate([eopatch.data["FEATURES_TRAINING"] for eopatch in sampled_eopatches], axis=1)
    labels = np.concatenate([eopatch.data_timeless[f"{trait_name}"] for eopatch in sampled_eopatches], axis=0)
    predicted_labels = np.concatenate([eopatch.data_timeless[f"PREDICTED_{trait_name}_{objective}_{model_type}"] for eopatch in sampled_eopatches], axis=0)

    in_poly = np.concatenate([eopatch.mask_timeless["IN_POLYGON"] for eopatch in sampled_eopatches], axis=0)
    mask = ~in_poly
    masked_labels = np.ma.masked_where(mask, labels)
    masked_predicted_labels = np.ma.masked_where(mask, predicted_labels)

    expanded_mask = np.expand_dims(mask, axis=0)
    expanded_mask = np.repeat(expanded_mask, features.shape[0], axis=0)
    expanded_mask = np.repeat(expanded_mask, features.shape[-1], axis=-1)
    masked_features = np.ma.masked_where(expanded_mask, features)

    if show:
        print("predicted data:")
        print(f"features.shape: {masked_features.shape}")
        print(f"labels.shape: {masked_labels.shape}")
        print(f"predicted labels.shape: {masked_predicted_labels.shape}")

    # return features, labels, predicted_labels
    return masked_features, masked_labels, masked_predicted_labels

def create_validation_data(trait_name, objective, model_type, show=False):
    """
    extract a trait and its prediction from eopatches for metrics
    """
    features, labels, predicted_labels = predictedData(
        areas=area_grid(DATA_validate),
        eopatch_samples_dir=EOPATCH_VALIDATE_DIR,
        model_type=model_type,
        objective=objective,
        trait_name=trait_name)
    #make and reshape two sets so labels and predictions get equivalent treatment
    fl = features, labels
    fp = features, predicted_labels
    fl_reshaped = reshape_to_GBM(data=fl, TSAI_shape=False)
    fp_reshaped = reshape_to_GBM(data=fp, TSAI_shape=False)
    f_reshaped, l_reshaped = fl_reshaped
    f_reshaped, p_reshaped = fp_reshaped
    #there is no test,train split in validation data so repeat f,l in both x,y positions
    data = f_reshaped, l_reshaped, f_reshaped, l_reshaped, p_reshaped

    if show:
        x_train, y_train, x_test, y_test, y_pred = data
        print("GBM validation data")
        print(f"x_train: {x_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_test: {y_test.shape}")
        print(f"y_pred: {y_pred.shape}")
    return data

def validationset_metrics(trait_name, area_name, objective, model_type, testset_name, class_names):
    """
    collects predicted data,  reports metrics

trait_name: str identifies trait in patches eg 'HEIGHT'
area_name: str descibes area being studied eg 'test-area'
objective: training objective, one of 'multiclass'  'regression', 'ranking'
model_type: model type one of 'GBM', 'TSAI'
class_names: list of str names for classes which were predicted
    """

    # get prediction data
    x_train, y_train, x_test, y_test, predicted_test  = create_validation_data(trait_name=trait_name, objective=objective, model_type=model_type)

    # deal with masked values in validation data: y_test, predicted_labels_test

    predicted_test = predicted_test.astype(float) # convert to guarantee int array is float
    predicted_test = predicted_test.filled(np.nan) # fill masked positions
    y_test = y_test.astype(float)
    y_test = y_test.filled(np.nan)

    # quantify prediction
    GBM_flag = model_type == 'GBM'
    TSAI_flag = model_type == 'TSAI'
    regression_flag = objective == 'regression'
    multiclass_flag = objective == 'multiclass'


    if (GBM_flag or TSAI_flag ) and (multiclass_flag):
        report_Metrics_Classification(
            y_test=y_test,
            predicted_labels_test=predicted_test,
            class_names=class_names,
            model_type=model_type,
            testset_name=testset_name,
            trait_name=trait_name,
            pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (regression_flag):
        report_Metrics_Regression(
            y_test=y_test,
            predicted_values_test=predicted_test,
            model_type=model_type,
            testset_name=testset_name,
            trait_name=trait_name,
            pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (multiclass_flag):
        show_std_T_confusionMatrix(
            predicted_labels_test=predicted_test,
            y_test=y_test,
            trait_name=trait_name,
            class_names=class_names,
            model_type=model_type,
            testset_name=testset_name,
            pred_type=objective)

    if (GBM_flag or TSAI_flag ) and (regression_flag):
        show_regression_results(
            predicted_values_test=predicted_test,
            y_test=y_test,
            trait_name=trait_name,
            model_type=model_type,
            testset_name=testset_name,
            pred_type=objective)

# USER
validationset_metrics(trait_name='HEIGHT', area_name='test-area', objective='multiclass', model_type='GBM', testset_name='transfer', class_names=['black','white', 'secret third thing'])
validationset_metrics(trait_name='HEIGHT', area_name='test-area', objective='regression', model_type='GBM', testset_name='transfer', class_names=['black','white', 'secret third thing'])

################
# * TST experiment
################

# USER
# test = create_TSAI_training_data(trait_name='HEIGHT', show=True)

######### ** Train

#####
# *** Supervised training
def trainTSAI(objective,
              area_name,
              trait_name,
              model_type,
              x_train_TSAI,
              y_train_TSAI,
              splits,
              show = False
              ):
    "Trains TSAI models"

    # shared setup
    batch_size = 8192 # print(math.pow(2,13))
    n_epochs = 400
    batch_tfms = TSStandardize(by_var=True) # TST model requires normalization by var
    inplace = False #true, transformation of training data, faster if it fits in mem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: device is cpu!")
    seed = RNDM

    # Set up the model and learner
    if objective == 'multiclass':
        tfms  = [None, [Categorize()]]
        metrics = [accuracy]
        loss_func = LabelSmoothingCrossEntropyFlat()
        dropout=0.3, # &&& hard coded
        fc_dropout=0.5

        # build unsupervised learner
        set_seed(seed, reproducible=True)
        dsets = TSDatasets(x_train_TSAI, y_train_TSAI, splits=splits, tfms=tfms, inplace=inplace)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, device=device, bs=batch_size, batch_tfms=batch_tfms, num_workers=0)
        model = TST(c_in=dls.vars, c_out=dls.c, seq_len=dls.len, dropout=.3, fc_dropout=.5)
        learn = Learner(dls, model, loss_func=loss_func, metrics=metrics)

    elif objective == 'regression':
        tfms  = [None, [TSRegression()]]
        metrics = [mae, rmse]
        loss_func = MSELossFlat()
        dropout=0.3, # &&& hard coded
        fc_dropout=0.5

        set_seed(seed, reproducible=True)
        dsets = TSDatasets(x_train_TSAI, y_train_TSAI, splits=splits, tfms=tfms, inplace=inplace)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, device=device, bs=batch_size, batch_tfms=batch_tfms, num_workers=0)
        model = TST(c_in=dls.vars, c_out=dls.c, seq_len=dls.len, dropout=.3, fc_dropout=.5)
        learn = Learner(dls, model, loss_func=loss_func, metrics=metrics)

    else:
        raise ValueError(f"The provided objective ({objective}) is not recognized")

    # Train the model
    plt.ioff() # turn off the plot of learning rate
    lr = learn.lr_find()
    plt.ion()
    learning_rate = lr[0]
    learn.fit_one_cycle(n_epochs, lr_max=learning_rate)
    # post run check
    print(f"Optimal Learning Rate: {learning_rate}")
    learn.plot_metrics()

    if show:
        # visualize results
        learn.show_results()
        learn.show_probas()
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()

    # Save learner
    identifier = f"{area_name}-{trait_name}-{objective}-{model_type}.pkl"
    model_path = os.path.join(MODELS_DIR, identifier)
    learn.export(model_path)

def ask_trainTSAI():
    print("train TSAI model?")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        x_train_TSAI, y_train_TSAI, splits = create_TSAI_training_data(trait_name='HEIGHT')
        trainTSAI(objective='multiclass',
                 area_name='test-area',
                 trait_name='HEIGHT',
                 model_type='TSAI',
                 x_train_TSAI=x_train_TSAI,
                 y_train_TSAI=y_train_TSAI,
                 splits = splits)
        trainTSAI(objective='regression',
                 area_name='test-area',
                 trait_name='HEIGHT',
                 model_type='TSAI',
                 x_train_TSAI=x_train_TSAI,
                 y_train_TSAI=y_train_TSAI,
                 splits = splits)

# USER
ask_trainTSAI()

######### ** Validate
# quantify prediction

# USER
testset_predict_validate(trait_name='HEIGHT', area_name='test-area', objective='regression', model_type='TSAI', testset_name='holdout', class_names=['black','white'])
testset_predict_validate(trait_name='HEIGHT', area_name='test-area', objective='multiclass', model_type='TSAI', testset_name='holdout', class_names=['black','white'])

######### ** Predict

# USER
# show validation area segmentation
# test = area_grid(DATA_validate, show=True)

# Prepare eopatches for the TSAI validation area
# TSAI overwrites GBM, may be needed, currently proceeding without this
# ask_loadgeotiffs(areas=area_grid(DATA_validate), eopatch_dir=EOPATCH_VALIDATE_DIR)
# ask_loadDetails(areas=area_grid(DATA_validate), mask_file=DATA_validate, eopatch_dir=EOPATCH_VALIDATE_DIR)
# eopatch = EOPatch.load(os.path.join(EOPATCH_VALIDATE_DIR, 'eopatch_0'))
# eopatch

def ask_PredictPatches_TSAI():
    print("predict validation area EOPatches?")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        execute_prepared_workflow(CreatePredictionWorkflow(areas=area_grid(DATA_validate),
                                                           eopatch_dir=EOPATCH_VALIDATE_DIR,
                                                           area_name='test-area',
                                                           trait_name='HEIGHT',
                                                           objective='regression',
                                                           model_type = 'TSAI'))
        execute_prepared_workflow(CreatePredictionWorkflow(areas=area_grid(DATA_validate),
                                                           eopatch_dir=EOPATCH_VALIDATE_DIR,
                                                           area_name='test-area',
                                                           trait_name='HEIGHT',
                                                           objective='multiclass',
                                                           model_type = 'TSAI'))

# USER
ask_PredictPatches_TSAI()

#####
# *** visualize prediction

def verify_predictions_TSAI():
    ""
    eopatch = EOPatch.load(os.path.join(EOPATCH_VALIDATE_DIR, 'eopatch_0'))
    eopatch
    eopatch.plot((FeatureType.DATA_TIMELESS, 'PREDICTED_HEIGHT_regression_TSAI'))
    eopatch.plot((FeatureType.DATA_TIMELESS, 'PREDICTED_HEIGHT_regression_TSAI_PROBA'))

# USER
verify_predictions_TSAI()

######### ** Quantify prediction
#####
# *** Visualize predicted trait

# USER
plot_prediction(grid_h = 1, grid_w = 2, trait_name = 'HEIGHT', model_type='TSAI', testset_name='transfer', pred_type='regression', areas=area_grid(DATA_validate))
plot_prediction(grid_h = 1, grid_w = 2, trait_name = 'HEIGHT', model_type='TSAI', testset_name='transfer', pred_type='multiclass', areas=area_grid(DATA_validate))

#####
# *** Visualize trait diff

# USER
plot_disagreement(trait_name = 'HEIGHT', areas = area_grid(DATA_validate), inspect_ratio=0.99, model_type='TSAI', testset_name='transfer', pred_type="regression")
plot_disagreement(trait_name = 'HEIGHT', areas = area_grid(DATA_validate), inspect_ratio=0.99, model_type='TSAI', testset_name='transfer', pred_type="multiclass")

#####
# *** Quantify agreement
# USER

validationset_metrics(trait_name='HEIGHT', area_name='test-area', objective='regression', model_type='TSAI', testset_name='transfer', class_names=['black','white', 'secret third thing'])
validationset_metrics(trait_name='HEIGHT', area_name='test-area', objective='multiclass', model_type='TSAI', testset_name='transfer', class_names=['black','white', 'secret third thing'])

######### ** Export to geotiff all for model comparison

#####
# *** export

class MaskTask(EOTask):
    "Uses in_polygon layer to mask a target prediction. Arg: ident, the layer to mask"
    def __init__(self, ident):
        self.ident = ident

    def execute(self, eopatch):

        no_data_value = NO_DATA_VALUE

        in_poly = eopatch.mask_timeless["IN_POLYGON"].squeeze()
        mask = ~in_poly
        data = eopatch.data_timeless[self.ident].squeeze()
        masked = np.ma.masked_where(mask, data)

        filled = masked.filled(no_data_value)
        fill3d = filled[..., np.newaxis] # add d for (w*h*1)
        eopatch[FeatureType.DATA_TIMELESS, f"{self.ident}_masked"] = fill3d
        return eopatch

def CreateExportWorkflow(areas, eopatch_dir, trait_name, objective, model_type):
    "Creates a workflow to export trait and prediction of validation eopatches. "

    tiff_location = RESULTS_DIR
    # set data timeless identifier, for prediction case and trait only case
    identifier = f"PREDICTED_{trait_name}_{objective}_{model_type}"
    if objective == None and model_type == None:
        identifier = f"{trait_name}"

    # test if identifier is in the layers, else print layers
    eopatch = EOPatch.load(os.path.join(eopatch_dir, 'eopatch_0'))
    data_keys = sorted(list(eopatch.data_timeless.keys()))
    if not identifier in data_keys:
        raise ValueError(f" identifier ({identifier}) not found in data_keys ({data_keys})")

    load_task = LoadTask(eopatch_dir)
    # mask outside of poly
    mask_task = MaskTask(identifier)
    export_task = ExportToTiffTask((FeatureType.DATA_TIMELESS, f"{identifier}_masked"), tiff_location)

    # node list
    workflow_nodes = linearly_connect_tasks(load_task, mask_task, export_task)
    # workflow
    workflow = EOWorkflow(workflow_nodes)

    # additional arguments
    execution_args = []
    for idx, bbox in enumerate(areas):
        execution_args.append(
            {
                workflow_nodes[0]: {"eopatch_folder": f"eopatch_{idx}"}, # load task is first
                workflow_nodes[-1]: {"filename": f"{tiff_location}/{identifier}_eopatch_{idx}.tiff"} # export task is last
            }
        )

    return workflow, execution_args

def merge_exports(trait_name, objective, model_type):
    # at this point is  it is known that there are multiple files on disk ending like ...eopatch_1.tiff

    no_data_value = NO_DATA_VALUE

    # set data timeless identifier, for prediction case and trait only case
    identifier = f"PREDICTED_{trait_name}_{objective}_{model_type}"
    if objective == None and model_type == None:
        identifier = f"{trait_name}"

    input_files = glob.glob(f"{RESULTS_DIR}/{identifier}_eopatch_*.tiff")
    output_file = f"{RESULTS_DIR}/{identifier}.tiff"

    src_files = [rasterio.open(f) for f in input_files]
    mosaic, out_trans = riomerge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "nodata": no_data_value,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })

    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()
    for f in input_files:
        os.remove(f)


def ask_ExportPatches():

                # set both objective and model_type are None to export only trait map
    exports = [{'trait_name': 'HEIGHT', 'objective': None, 'model_type': None},
               {'trait_name': 'HEIGHT', 'objective': 'regression', 'model_type': 'TSAI'},
               {'trait_name': 'HEIGHT', 'objective': 'multiclass', 'model_type': 'TSAI'},
               {'trait_name': 'HEIGHT', 'objective': 'regression', 'model_type': 'GBM'},
               {'trait_name': 'HEIGHT', 'objective': 'multiclass', 'model_type': 'GBM'}]

    print("export predictions?")
    proceed = input("Do you want to proceed? (y/n): ").lower().strip() == 'y'
    if proceed:
        for e in exports:
            print(f"exporting: {e}")
            execute_prepared_workflow(CreateExportWorkflow(areas=area_grid(DATA_validate),
                                                               eopatch_dir=EOPATCH_VALIDATE_DIR,
                                                               trait_name=e['trait_name'],
                                                               objective=e['objective'],
                                                               model_type =e['model_type']))
            merge_exports(trait_name=e['trait_name'],
                          objective=e['objective'],
                          model_type =e['model_type'])

# USER
ask_ExportPatches()

############################################ Fin
