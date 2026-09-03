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
RAW_NO_DATA = -9999

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
ROS_FL_CLASSES = ["N", "VL", "L", "M", "H", "VH", "X"]
INPUT_KEYS = ["cbd", "cbh", "cc", "fbfm", "th"]
# inputs that are mean-std normalized
NORM_KEYS = ["cbd", "cbh", "cc", "th", "gtr_ros", "gtr_fl", "slope"]
# inputs that are log1p transformed
LOG1P_KEYS = ["cbd", "cbh", "th", "gtr_ros", "gtr_fl"]
ROLE_KEYS = ['baseline', 'treatment']

# cli path constants
METHODS = ["train", "evaluate", "evaluate_iterations", "run", "bundle"]
OUTDIR = Path("data/outputs")
BUNDLE_DIR = Path("data/bundles")
CONFIG_DIR = Path("configs")
WIND_DIRECTIONS = Path("data/training_data/wind_directions.csv")
