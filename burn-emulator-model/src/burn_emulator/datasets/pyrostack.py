from collections import OrderedDict

import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from burn_emulator.constants import NO_DATA, RAW_NO_DATA, Path
from burn_emulator.models.utils import circular_components

DEFAULT_REQUIRED_VARS = [
    "fire_spread/fline",
    "fire_spread/farea",
    "fire_spread/nfp",
    "fire_spread/frp",
    "fuel_structure/cbd",
    "fuel_structure/cbh",
    "fuel_structure/cc",
    "fuel_structure/ch",
    "high_res_climate/lh",
    "high_res_climate/lw",
    "high_res_climate/m1",
    "high_res_climate/m10",
    "high_res_climate/m100",
    "high_res_climate/wd",
    "high_res_climate/ws",
    "veg_fm_topo/evt",
    "veg_fm_topo/fbfm13",
    "veg_fm_topo/fbfm40",
    "veg_fm_topo/roads",
    "veg_fm_topo/asp",
    "veg_fm_topo/elev",
    "veg_fm_topo/slpd",
]

NON_CONTINUOUS_VARS = {
    "fire_spread/fline",
    "fire_spread/farea",
    "fire_spread/nfp",
}

FBFM40_MAP = {
    -9999: 0,
    91: 1,
    92: 2,
    93: 3,
    98: 4,
    99: 5,
    101: 6,
    102: 7,
    103: 8,
    104: 9,
    105: 10,
    106: 11,
    107: 12,
    108: 13,
    121: 14,
    122: 15,
    123: 16,
    124: 17,
    141: 18,
    142: 19,
    143: 20,
    144: 21,
    145: 22,
    146: 23,
    147: 24,
    148: 25,
    149: 26,
    161: 27,
    162: 28,
    163: 29,
    164: 30,
    165: 31,
    181: 32,
    182: 33,
    183: 34,
    184: 35,
    185: 36,
    186: 37,
    187: 38,
    188: 39,
    189: 40,
    201: 41,
    202: 42,
    203: 43,
}

FBFM13_MAP = {
    -9999: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    91: 13,
    92: 14,
    93: 15,
    98: 16,
    99: 17,
}

FBFM13_NUM_CLASSES = len(set(FBFM13_MAP.values()))
FBFM40_NUM_CLASSES = len(set(FBFM40_MAP.values()))

# Dataset_README.md documents these as encoded raw units (cbd: 100 kg/m^3,
# cbh/ch: m*10, cc: percent); scale factors convert each to physical units.
FUEL_STRUCTURE_SCALE = {"cbd": 0.01, "cbh": 0.1, "cc": 0.01, "ch": 0.1}
FUEL_STRUCTURE_VARS = ["cbd", "cbh", "cc", "ch"]
TOPO_CATEGORICAL_VARS = ["evt", "roads"]  # kept as raw codes, not one-hot
HIGH_RES_CLIMATE_VARS = ["lh", "lw", "m1", "m10", "m100", "ws", "wd"]
LOW_RES_CLIMATE_VARS = ["d2m", "sp", "t2m", "tp"]


def _find_layer_file(dir_path: Path, layer: str) -> Path | None:
    # veg_fm_topo filenames carry inconsistent year/region prefixes and
    # suffixes (e.g. "230fbfm13.tif", "lf2022_fbfm13_ak.tif", "fbfm13.tif"),
    # and per the Dataset_README, fbfm13/fbfm40 may instead be named f13/f40
    aliases = {"fbfm13": ("fbfm13", "f13"), "fbfm40": ("fbfm40", "f40")}.get(layer, (layer,))
    for file in sorted(dir_path.glob("*.tif")):
        if any(alias in file.stem.lower() for alias in aliases):
            return file
    return None


def _read_raster(path: Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        return torch.from_numpy(src.read()).float()


def _remap_categories(raw: torch.Tensor, mapping: dict) -> torch.Tensor:
    out = torch.zeros_like(raw)
    for k, v in mapping.items():
        out[raw == k] = v
    return out


class PyroStack(Dataset):
    def __init__(
        self,
        fire_dirs: list[Path] | Path,
        cache_size: int = 4,
    ) -> None:
        if isinstance(fire_dirs, (str, Path)):
            fire_dirs = sorted(Path(fire_dirs).glob("fires_*/cubes/*"))
        self.fire_dirs = [Path(d) for d in fire_dirs]

        self.cache_size = cache_size
        self._cache: OrderedDict[Path, dict] = OrderedDict()

        self._true_hours: dict[Path, list[int]] = {}
        self._index: list[tuple[int, int]] = []
        for fidx, fire_dir in enumerate(self.fire_dirs):
            times = pd.read_csv(fire_dir / "fire_times.csv")
            true_hours = times.index[times["feds"]].tolist()
            self._true_hours[fire_dir] = true_hours
            self._index.extend((fidx, t) for t in range(len(true_hours) - 1))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fidx, t = self._index[idx]
        fire_dir = self.fire_dirs[fidx]
        fire = self._get_fire(fire_dir)

        a, b = fire["true_hours"][t], fire["true_hours"][t + 1]

        sample = {
            "farea": fire["farea"][t : t + 1],
            "y": fire["farea"][t + 1 : t + 2],
            **fire["static"],
        }
        for name, layer in {**fire["hr_climate"], **fire["lr_climate"]}.items():
            sample[name] = layer[a:b].mean(dim=0)
        return sample

    def _get_fire(self, fire_dir: Path) -> dict:
        if fire_dir in self._cache:
            self._cache.move_to_end(fire_dir)
            return self._cache[fire_dir]

        fire = self._load_fire(fire_dir)
        self._cache[fire_dir] = fire
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return fire

    def _load_fire(self, fire_dir: Path) -> dict:
        farea = _read_raster(fire_dir / "fire_spread" / "farea.tif")
        fuel_structure = self._load_fuel_structure(fire_dir / "fuel_structure")
        # veg_fm_topo shares fuel_structure's 30m resolution and per-fire
        # bounding box, so its shape is a valid fallback for missing layers
        ref_shape = tuple(fuel_structure["cbd"].shape[-2:])

        static_path = fire_dir / "veg_fm_topo"
        static = {
            **fuel_structure,
            **self._load_fbfm(static_path, "fbfm13", FBFM13_MAP, FBFM13_NUM_CLASSES),
            **self._load_fbfm(static_path, "fbfm40", FBFM40_MAP, FBFM40_NUM_CLASSES),
            **self._load_topo_categorical(static_path, ref_shape),
            **self._load_topo_continuous(static_path),
        }

        hr_climate = self._load_climate(fire_dir / "high_res_climate", HIGH_RES_CLIMATE_VARS)
        lr_climate = self._load_climate(fire_dir / "low_res_climate", LOW_RES_CLIMATE_VARS)

        return {
            "farea": farea,
            "static": static,
            "hr_climate": hr_climate,
            "lr_climate": lr_climate,
            "true_hours": self._true_hours[fire_dir],
        }

    def _load_fuel_structure(self, fs_dir: Path) -> dict[str, torch.Tensor]:
        out = {}
        for name in FUEL_STRUCTURE_VARS:
            raw = _read_raster(fs_dir / f"{name}.tif")
            valid = raw != RAW_NO_DATA
            raw[valid] *= FUEL_STRUCTURE_SCALE[name]
            raw[~valid] = NO_DATA
            out[name] = raw
        return out

    def _load_fbfm(
        self,
        dir_path: Path,
        layer: str,
        mapping: dict,
        num_classes: int,
    ) -> dict[str, torch.Tensor]:
        raw = _read_raster(_find_layer_file(dir_path, layer))
        remapped = _remap_categories(raw, mapping)
        onehot = F.one_hot(remapped.long(), num_classes=num_classes)
        onehot = onehot.squeeze(0).permute(2, 0, 1).float()
        return {layer: onehot}

    def _load_topo_categorical(
        self, dir_path: Path, ref_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        out = {}
        for name in TOPO_CATEGORICAL_VARS:
            file = _find_layer_file(dir_path, name)
            if file is None:
                # a missing roads tif means the fire has no operational roads
                # in its bounding box, not that the data is unavailable
                fill = 0.0 if name == "roads" else float(NO_DATA)
                out[name] = torch.full((1, *ref_shape), fill)
                continue
            raw = _read_raster(file)
            raw[raw == RAW_NO_DATA] = NO_DATA
            out[name] = raw
        return out

    def _load_topo_continuous(self, dir_path: Path) -> dict[str, torch.Tensor]:
        elev = _read_raster(_find_layer_file(dir_path, "elev"))
        slpd = _read_raster(_find_layer_file(dir_path, "slpd"))
        asp = _read_raster(_find_layer_file(dir_path, "asp"))

        valid = asp != RAW_NO_DATA
        asp_sin, asp_cos = circular_components(asp)
        asp_sin[~valid] = NO_DATA
        asp_cos[~valid] = NO_DATA

        elev[elev == RAW_NO_DATA] = NO_DATA
        slpd[slpd == RAW_NO_DATA] = NO_DATA

        return {"elev": elev, "slpd": slpd, "asp_sin": asp_sin, "asp_cos": asp_cos}

    def _load_climate(self, dir_path: Path, var_names: list[str]) -> dict[str, torch.Tensor]:
        layers = {}
        for name in var_names:
            raw = torch.nan_to_num(_read_raster(dir_path / f"{name}.tif"), nan=0.0)
            if name == "wd":
                sin, cos = circular_components(raw)
                layers["wd_sin"], layers["wd_cos"] = sin, cos
            else:
                layers[name] = raw
        return layers


def pyrostack_collate(batch: list[dict]) -> dict[str, list[torch.Tensor]]:
    return {key: [sample[key] for sample in batch] for key in batch[0]}


def build_pyrostack_dataloader(dataset: PyroStack, **kwargs) -> DataLoader:
    return DataLoader(dataset, collate_fn=pyrostack_collate, **kwargs)
