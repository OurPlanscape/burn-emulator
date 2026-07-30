import os

import torch

RUN_DEVICE = os.environ.get("RUN_DEVICE", "cuda")
RUN_DTYPE = os.environ.get("RUN_DTYPE", "float")
USE_CLOUD_PATHS = os.environ.get("USE_CLOUD_PATHS") == 1
if USE_CLOUD_PATHS:
    from cloudpathlib import AnyPath as Path
else:
    from pathlib import Path

__all__ = ["Path"]


# training constants
DEFAULT_DTYPE = torch.bfloat16  # default trainining dtype for memory saving
NO_DATA = -1  # no data value for NN inputs

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

# path constants
METHODS = ["train", "test", "test_iterations", "package", "run"]
OUTDIR = Path("data/outputs")
DOCKERDIR = Path("docker")

# test constants
TEST_BATCH_SIZE = 32
TEST_IMAGE_SIZE = 129
TEST_IN_CHANS = 19
TEST_OUT_CHANS = 3

# TODO: give these proper paths
TEST_FIXTURES_DIR = Path(__file__).parent.parent.parent / "configs"
TEST_CONFIG_PATH = TEST_FIXTURES_DIR / "ignition_dataset_config.yaml"
