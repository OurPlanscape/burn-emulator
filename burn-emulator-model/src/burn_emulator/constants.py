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
NO_DATA = -1                                # no data value for NN inputs
RAW_NO_DATA = -9999

# input fuel specific constants
FBFM_OH_MAP = {
    -999: 0,  # no data
    0: 1,  # should be 91 but is in the same spot one-hot encoded anyway
    91: 1,
    101: 2,
    105: 3,
    106: 4,
    141: 5,
    144: 6,
    148: 7,
    161: 8,
    163: 9,
    186: 10,
    189: 11,
    202: 12,
    203: 13,
}
INF_PROFILE = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": -999,
    "crs": "EPSG:5070",
    "blockxsize": 256,
    "blockysize": 256,
    "tiled": True,
    "compress": "lzw",
    "interleave": "band",
}
INPUT_KEYS = ["cbd", "cbh", "cc", "fbfm", "th"]

# cli path constants
METHODS = ["train", "test", "test_iterations", "run", "bundle"]
OUTDIR = Path("data/outputs")
BUNDLE_DIR = Path("data/bundles")
