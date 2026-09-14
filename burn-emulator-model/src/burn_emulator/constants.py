import os

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# switch for bare metal -> cloud
USE_CLOUD_PATHS = bool(os.environ.get("USE_CLOUD_PATHS"))
if USE_CLOUD_PATHS:
    from cloudpathlib import AnyPath as Path
else:
    from pathlib import Path
__all__ = ["Path"]

# TODO: potentially change to enums
# training and inference constants
RUN_DEVICE = os.environ.get("RUN_DEVICE", "cuda")
RUN_DTYPE = getattr(torch, os.environ.get("RUN_DTYPE", "bfloat16"))
DEFAULT_DEVICE = torch.device(RUN_DEVICE)   # default device for training
DEFAULT_DTYPE = torch.bfloat16              # default trainining dtype for memory saving
NO_DATA = -3                                # no data value for NN inputs (-3σ of normalized data)
RAW_NO_DATA = -999                     

INF_PROFILE = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": RAW_NO_DATA,
    "crs": "EPSG:5070",
    "blockxsize": 256,
    "blockysize": 256,
    "tiled": True,
    "compress": "lzw",
    "interleave": "band",
}
TARGET_CRS = INF_PROFILE["crs"]  # raster CRS everything is reprojected into before use
ROS_FL_CLASSES = ["N", "VL", "L", "M", "H", "VH", "X"]
INPUT_KEYS = ["cbd", "cbh", "cc", "fbfm", "th"]
# inputs that are mean-std normalized
NORM_KEYS = ["cbd", "cbh", "cc", "th", "gtr_ros", "gtr_fl", "slope"]
# inputs that are log1p transformed
LOG1P_KEYS = ["cbd", "cbh", "th", "gtr_ros", "gtr_fl"]
ROLE_KEYS = ['baseline', 'treatment']

# cli path constants
METHODS = ["train", "evaluate", "evaluate_iterations", "run", "bundle", "ignite"]
OUTDIR = Path("data/outputs")
BUNDLE_DIR = Path("data/bundles")
CONFIG_DIR = Path("configs")
TRAINING_DATA_DIR = Path("data/training_data")
WIND_DIRECTIONS = TRAINING_DATA_DIR / "wind_directions.csv"

# training-data generation (src/burn_emulator/ignite.py)
WEST_FUELS_DIR_PREFIX = "West_Fuels_DN"
VARLOCS_GPKG = TRAINING_DATA_DIR / "western_varlocs_5070_cleaned.gpkg"
TOPO_SOURCE_DIR = TRAINING_DATA_DIR / "topo" / "LF"
ASPECT_FILE = TOPO_SOURCE_DIR / "aspect.tif"
SLOPE_FILE = TOPO_SOURCE_DIR / "slope_degrees.tif"
