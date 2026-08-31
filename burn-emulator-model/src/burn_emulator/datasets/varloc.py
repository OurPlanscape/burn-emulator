from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import yaml
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds, rowcol
from rasterio.windows import Window, from_bounds
from shapely import Polygon
from torch.utils.data import Dataset

from burn_emulator.constants import (
    DEFAULT_DTYPE,
    FBFM_OH_MAP,
    INPUT_KEYS,
    NO_DATA,
    NONBURN_FBFM_CH,
    ROLE_KEYS,
    USE_CLOUD_PATHS,
    Path,
)
from burn_emulator.datasets.utils import compute_bounds, compute_padding
from burn_emulator.types import IgnitionMethod
from burn_emulator.utils import circle_mask, to_flow


MAX_IGNITION_RESAMPLE = 42 # likely will never reach this limit


def _window(src, window_bounds: tuple | None) -> Window | None:
    """Pixel-aligned rasterio Window for the given bounds, or None for a full read."""
    if window_bounds is None:
        return None
    return (
        from_bounds(*window_bounds, transform=src.transform).round_offsets().round_lengths()
    )


def _windowed_profile(src, window: Window | None, dat: torch.Tensor) -> dict:
    profile = src.profile
    if window is not None:
        profile.update(
            transform=src.window_transform(window),
            width=dat.shape[-1],
            height=dat.shape[-2],
        )
    return profile


def _encode_fbfm(dat: torch.Tensor) -> torch.Tensor:
    for k, v in FBFM_OH_MAP.items():
        dat[dat == k] = v
    dat = F.one_hot(dat.long(), num_classes=len(np.unique(list(FBFM_OH_MAP.values()))))
    return dat.squeeze(0).permute(2, 0, 1)[1:].to(DEFAULT_DTYPE)  # drop the no-data channel


def _clean_continuous(dat: torch.Tensor, nodata: float | None) -> torch.Tensor:
    dat = dat.to(DEFAULT_DTYPE)
    dat[dat == nodata] = torch.nan
    dat[dat < 0] = 0
    return dat


def _read_fuel_dir(
    fuels_path: Path,
    window_bounds: tuple | None,
    sample_region: Polygon | None,
    window_size: int,
) -> tuple[dict, torch.Tensor | None, dict | None, tuple | None]:
    """Read + preprocess one fuel directory. Returns (layer, mask, profile, window_bounds)"""
    # TODO: convert to zarr and/or icechunk inputs
    files = {f.stem.rsplit("_", 1)[1]: f for f in fuels_path.glob("*.tif")}
    layer, mask, profile = {}, None, None
    for name in INPUT_KEYS:
        file = files.get(name)
        if file is None:
            continue
        with rasterio.open(file) as src:
            if window_bounds is None and sample_region is not None:
                # margin so every ignition's full model window fits in the read
                window_bounds = tuple(
                    sample_region.buffer(window_size * max(src.res), join_style="mitre").bounds
                )
            window = _window(src, window_bounds)
            dat = torch.tensor(src.read(window=window))
            if name == "fbfm":
                profile = _windowed_profile(src, window, dat)
                mask = dat != src.nodata
                dat = _encode_fbfm(dat)
            else:
                dat = _clean_continuous(dat, src.nodata)
        layer[name] = dat
    return layer, mask, profile, window_bounds


def _normalize_inputs(inputs: dict, stats_path: Path) -> None:
    """Standardize continuous inputs in place; load stats.yaml or compute + persist it."""
    stats_data = stats_path.exists()
    if stats_data:
        with stats_path.open() as f:
            stats = yaml.safe_load(f)
    else:
        stats = {}
    for key in INPUT_KEYS:
        if key == "fbfm":
            continue
        if stats_data:
            mean, stdv = stats[key]["mean"], stats[key]["stdv"]
        else:
            arrs = torch.concat([inputs[r][key] for r in inputs])
            mean = torch.nanmean(arrs).item()
            stdv = torch.sqrt(torch.nanmean((arrs - mean) ** 2)).item()
            stats[key] = {"mean": mean, "stdv": stdv}
        for r in inputs:
            inputs[r][key] = (inputs[r][key] - mean) / stdv
            inputs[r][key][torch.isnan(inputs[r][key])] = NO_DATA
    if not stats_data and not USE_CLOUD_PATHS:
        with stats_path.open("w") as f:
            yaml.dump(stats, f, sort_keys=False)


@lru_cache(maxsize=4)
def _load_topos(topo_path: str, flow: bool, window_bounds: tuple | None) -> dict:
    topo_path = Path(topo_path)
    topos = {}

    with rasterio.open(topo_path / "aspect.tif") as src:
        aspect = torch.tensor(src.read(window=_window(src, window_bounds))).to(DEFAULT_DTYPE)
    with rasterio.open(topo_path / "slope_degrees.tif") as src:
        slope = torch.tensor(src.read(window=_window(src, window_bounds))).to(DEFAULT_DTYPE)

    if flow:
        flow_x, flow_y = to_flow(aspect, slope)
        topos["flow_x"] = flow_x
        topos["flow_y"] = flow_y
    else:
        topos["aspect"] = aspect
        topos["slope"] = slope
    return topos


def cache_fuels_inputs(
    fuels_paths: dict[str, Path],
    topo_path: Path,
    stats_path: Path,
    sample_region: Polygon | None = None,
    window_size: int = 129,
    flow: bool = True,
) -> tuple[dict, dict, dict, dict]:
    inputs, masks, profile, bounds = {}, {}, None, None
    ref_fkey, ref_shape, ref_bounds = None, None, None
    for fkey, fuels_path in fuels_paths.items():
        inputs[fkey], masks[fkey], profile, bounds = _read_fuel_dir(
            fuels_path, bounds, sample_region, window_size
        )
        if ref_fkey is None:
            shape = (profile["height"], profile["width"])
            bbox = array_bounds(profile["height"], profile["width"], profile["transform"])
            ref_fkey, ref_shape, ref_bounds = fkey, shape, bbox
        else:
            assert shape == ref_shape, (
                f"fuels shape mismatch: {fkey} {shape} != {ref_fkey} {ref_shape}"
            )
            assert bbox == ref_bounds, (
                f"fuels bounds mismatch: {fkey} {bbox} != {ref_fkey} {ref_bounds}"
            )
    _normalize_inputs(inputs, stats_path)
    topos = _load_topos(str(topo_path), flow, bounds)
    return inputs, topos, masks, profile


class VarLoc(Dataset):
    def __init__(
        self,
        ignitions_path: str | Path | None,
        fuels_paths: dict[str, str | Path],
        burn_paths: dict[str, str | Path] | None,
        wind_ang_paths: dict[str, str | Path] | None,
        topo_path: str | Path,
        stats_path: str | Path,
        burn_times: list[int] | None = None, # NOTE(burn_times): unused for v1
        window_size: int = 129,
        jitter: int | None = None,
        treatment_area: Polygon | None = None,
        treatment_buff: float | None = None,  # metres to buffer treatment_area by
        treatment_seed: int = 42,
        ignition_method: IgnitionMethod = "uniform",
        ignition_density: float | None = 5e-5,
        wind_seed: int = 42,
        wind_range: list[float] | None = None,
        circle_mask: bool = True,
        one_hot: bool = True,
        num_classes: int = 4,  # unburned, surface, passive crown, active crown
    ) -> None:
        # fuels_paths / burn_paths / wind_ang_paths are role-keyed: baseline / treatment
        self.fuels_paths = {k: Path(p) for k, p in fuels_paths.items()}
        assert self.fuels_paths.keys() <= set(ROLE_KEYS), (
            f"fuels_paths keys {list(self.fuels_paths)} must be a subset of {ROLE_KEYS}"
        )

        self.burn_paths = burn_paths
        if burn_paths is not None:
            self.burn_paths = {k: Path(p) for k, p in burn_paths.items()}
            assert self.fuels_paths.keys() == self.burn_paths.keys(), (
                "fuels_paths and burn_paths must have the same roles"
            )

        self.wind_ang_paths = wind_ang_paths
        if wind_ang_paths is not None:
            self.wind_ang_paths = {k: Path(p) for k, p in wind_ang_paths.items()}
            assert self.fuels_paths.keys() == self.wind_ang_paths.keys(), (
                "fuels_paths and wind_ang_paths must have the same roles"
            )

        self.topo_path = Path(topo_path)
        self.stats_path = Path(stats_path)

        # NOTE(burn_times): unused for v1 but leaving for future development
        self.burn_times = [str(bt) for bt in burn_times] if burn_times else burn_times
        self.window_size = window_size
        self.jitter = jitter

        self.treatment_area = treatment_area
        self.treatment_buff = treatment_buff
        self.treatment_seed = treatment_seed
        self.ignition_density = ignition_density
        self.ignition_method = ignition_method

        self.wind_seed = wind_seed
        self.wind_range = wind_range
        self.circle_mask = circle_mask
        self.one_hot = one_hot
        self.num_classes = num_classes

        self._sample_gs = self._sampling_region(ignitions_path, treatment_area, treatment_buff)
        self.sample_region = (
            None if self._sample_gs is None else self._sample_gs.union_all().envelope
        )
        self.fuels, self.topos, self.masks, self.profile = cache_fuels_inputs(
            self.fuels_paths,
            self.topo_path,
            self.stats_path,
            sample_region=self.sample_region,
            window_size=self.window_size,
        )

        self._set_ignitions(
            ignitions_path, treatment_seed, ignition_density, ignition_method
        )

        # sampled ignitions carry only geometry; derive raster row/col from the profile
        if "row" not in self.ignitions.columns:
            self._locate_ignitions()

        if self.treatment_area is not None:
            self._collate_treatments(treatment_area)
        self._set_wind_angles()

        if self.circle_mask:
            self._set_circle_mask()

    def __len__(self) -> int:
        if self.burn_paths is None:
            return len(self.ignitions)
        else:
            return len(self.ignitions) * len(self.burn_paths)

    def __getitem__(self, idx: int) -> dict:
        if self.burn_paths is None:
            sidx = idx
            bidx = 0
        else:
            sidx = idx // len(self.burn_paths)
            bidx = idx % len(self.burn_paths)

        fkey = list(self.fuels_paths)[bidx]
        if self.burn_paths is not None:
            burn_path = self.burn_paths[fkey]
        ignition = self.ignitions.iloc[sidx]

        y = int(ignition["row"].item())
        x = int(ignition["col"].item())

        if self.wind_angles is not None:
            wind = self.wind_angles[fkey].iloc[sidx]

        # only to be used for training
        if self.jitter is not None:
            y += np.random.randint(-(self.jitter + 1), self.jitter)
            x += np.random.randint(-(self.jitter + 1), self.jitter)

        _, h, w = self.fuels[fkey]["fbfm"].shape
        ymin, ymax, xmin, xmax, yslc, xslc = compute_bounds(y, x, h, w, self.window_size)
        ydiff, xdiff, ypad, xpad = compute_padding(ymin, ymax, xmin, xmax, self.window_size)

        # stacking treated area as a secondary X
        mask = self._build_mask(fkey, yslc, xslc, ydiff, xdiff, ypad, xpad)
        if self.treatment_area is not None:
            arrX = torch.stack([
                self._build_arrx(k, yslc, xslc, ydiff, xdiff, ypad, xpad)
                for k in self.fuels_paths
            ])
        else:
            arrX = self._build_arrx(fkey, yslc, xslc, ydiff, xdiff, ypad, xpad)

        # padding information is not necessary for training
        if self.burn_paths is not None:
            arrY = self._build_arry(ignition, burn_path, yslc, xslc, ydiff, xdiff, ypad, xpad)
            return {
                "x": arrX,
                "y": arrY,
                "wind": wind,
                "mask": mask,
            }
        # burns are not necessary for inference
        else:
            return {
                "x": arrX,
                "wind": wind,
                "mask": mask,
                "pdiffs": (ydiff, xdiff),
                "bounds": (ymin, ymax, xmin, xmax),
                "indxes": (sidx, bidx),
            }

    def _pad(
        self,
        arr: torch.Tensor,
        ydiff: int,
        xdiff: int,
        ypad: tuple[int, int, int, int],
        xpad: tuple[int, int, int, int],
        value: float,
    ) -> torch.Tensor:
        if ydiff > 0:
            arr = F.pad(arr, ypad, mode="constant", value=value)
        if xdiff > 0:
            arr = F.pad(arr, xpad, mode="constant", value=value)
        return arr

    def _build_mask(
        self,
        fkey: str,
        yslc: slice,
        xslc: slice,
        ydiff: int,
        xdiff: int,
        ypad: tuple[int, int, int, int],
        xpad: tuple[int, int, int, int],
    ) -> torch.Tensor:
        # the mask is fbfm shaped. see utils.cache_intputs
        mask = self.masks[fkey][:, yslc, xslc]
        mask = self._pad(mask, ydiff, xdiff, ypad, xpad, value=0)
        if self.circle_mask:
            mask = mask & self.cmask
        return mask

    def _build_arrx(
        self,
        fkey: str,
        yslc: slice,
        xslc: slice,
        ydiff: int,
        xdiff: int,
        ypad: tuple[int, int, int, int],
        xpad: tuple[int, int, int, int],
    ) -> torch.Tensor:
        # slopes (after caching) 0: x flow 1: y flow
        arrX = []
        for _key, values in self.topos.items():
            arr = values[:, yslc, xslc]
            arr = self._pad(arr, ydiff, xdiff, ypad, xpad, value=NO_DATA)
            arrX.append(arr)

        # one hots should be padded with 0 not -1
        for key, values in self.fuels[fkey].items():
            arr = values[:, yslc, xslc]
            no_data = 0 if key == "fbfm" else NO_DATA
            arr = self._pad(arr, ydiff, xdiff, ypad, xpad, value=no_data)
            if key == "fbfm":
                arr_fbfm = arr
            else:
                arrX.append(arr)

        arrX = torch.concat(arrX)
        if self.circle_mask:
            arrX[:, ~self.cmask] = NO_DATA
            arr_fbfm[:, ~self.cmask] = 0
        arrX = torch.concat([arrX, arr_fbfm])
        return arrX

    def _build_arry_raw(
        self,
        ignition: pd.Series,
        burn_path: Path,
        yslc: slice,
        xslc: slice,
    ) -> torch.Tensor:
        # raw fire_type class per pixel: 0 unburned | 1 surface | 2 passive crown | 3 active crown
        arrY = []
        igd = burn_path / str(int(ignition["ignition_number"].item()))
        # NOTE(burn_times): unused for v1
        if self.burn_times:
            bps = [igd / bt / "fire_type.tif" for bt in self.burn_times]
        else:
            bps = [igd / "fire_type.tif"]
        for bp in bps:
            with rasterio.open(bp) as src:
                arr = torch.tensor(src.read())
            arr = arr[:, yslc, xslc]
            arrY.append(arr)
        return torch.concat(arrY)

    def _build_arry(
        self,
        ignition: pd.Series,
        burn_path: Path,
        yslc: slice,
        xslc: slice,
        ydiff: int,
        xdiff: int,
        ypad: tuple[int, int, int, int],
        xpad: tuple[int, int, int, int],
    ) -> torch.Tensor:
        arrY = self._build_arry_raw(ignition, burn_path, yslc, xslc)
        if self.one_hot:
            arrY = F.one_hot(arrY.long(), num_classes=self.num_classes)
            arrY = arrY.permute(0, 3, 1, 2).flatten(0, 1)
        else:
            arrY = arrY >= 1
        arrY = self._pad(arrY, ydiff, xdiff, ypad, xpad, value=0)
        return arrY

    def _sampling_region(
        self,
        ignitions_path: Path | None,
        treatment_area: Polygon | None,
        treatment_buff: float | None,
    ) -> gpd.GeoSeries | None:
        if ignitions_path is None:
            region = treatment_area
            if treatment_buff is not None:
                region = treatment_area.buffer(treatment_buff)  # metres
            return gpd.GeoSeries([region])
        if Path(ignitions_path).suffix in {".gpkg", ".geojson", ".zip"}:
            return gpd.read_file(ignitions_path).geometry
        return None

    def _set_ignitions(
        self,
        ignitions_path: Path | None = None,
        treatment_seed: int = 42,
        ignition_density: float | None = None,
        ignition_method: IgnitionMethod = "uniform",
    ) -> None:
        if self._sample_gs is not None:
            self._sample_ignitions(
                self._sample_gs, ignition_density, treatment_seed, ignition_method
            )
            return
        match Path(ignitions_path).suffix:
            case ".csv":
                self.ignitions = pd.read_csv(ignitions_path)
                if "cbp_burn" not in self.ignitions.columns:
                    name = Path(ignitions_path).stem.split("_")[0]
                    self.ignitions.loc[:, "cbp_burn"] = name
                if "index" in self.ignitions.columns:
                    self.ignitions["ignition_number"] = self.ignitions["index"]
                if "ignition_number" not in self.ignitions.columns:
                    self.ignitions = self.ignitions.reset_index(names="ignition_number")
            case _:
                ignitions_paths = Path(ignitions_path).glob("**/*.csv")
                self.ignitions = []
                for ignitions_path in sorted(ignitions_paths):
                    name = ignitions_path.stem.split("_")[0]
                    ignition = pd.read_csv(ignitions_path)
                    ignition = ignition.reset_index(names="ignition_number")
                    ignition.loc[:, "cbp_burn"] = name
                    self.ignitions.append(ignition)
                self.ignitions = pd.concat(self.ignitions)

    def _set_wind_angles(self) -> None:
        self.wind_angles = {}
        assert self.wind_ang_paths is not None or self.wind_range is not None, (
            "Either wind_ang_paths or wind_range must be provided to set wind angles"
        )
        if self.wind_ang_paths:
            for fkey, wind_ang_path in self.wind_ang_paths.items():
                self.wind_angles[fkey] = pd.read_csv(wind_ang_path)["upwind_direction"]
        if self.wind_range is not None:
            rng = np.random.default_rng(seed=self.wind_seed)
            wind_angle = rng.uniform(
                                self.wind_range[0], self.wind_range[1], len(self.ignitions)
                            )
            for fkey in self.fuels_paths:
                self.wind_angles[fkey] = pd.Series(wind_angle, name="upwind_direction")

    def _set_circle_mask(self) -> None:
        self.cmask = circle_mask(self.window_size)

    def _burnable_mask(self, points: gpd.GeoSeries) -> np.ndarray:
        rows, cols = rowcol(self.profile["transform"], points.x.to_numpy(), points.y.to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        burnable = np.ones(len(points), dtype=bool)
        for fkey in self.fuels_paths:
            _, h, w = self.fuels[fkey]["fbfm"].shape
            inbounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            r, c = np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)
            present = self.masks[fkey][0][r, c].numpy()
            nonburn = self.fuels[fkey]["fbfm"][NONBURN_FBFM_CH][r, c].float().numpy() > 0
            burnable &= inbounds & present & ~nonburn
        return burnable

    def _sample_ignitions(
        self,
        gs: gpd.GeoSeries,
        ignition_density: float = 5e-5,
        treatment_seed: int = 42,
        method: IgnitionMethod = "uniform",
    ) -> None:
        n_points = round(gs.area.sum() * ignition_density)
        # using evelope since ignitions aren't going to be defined by PA
        env = gs.envelope
        sampled = env.sample_points(size=n_points, method=method, rng=treatment_seed)
        sampled = sampled.explode(index_parts=False).reset_index(drop=True)

        # TODO: for method="uniform" this could be a single vectorized draw over the
        # cached burnable pixel grid instead of a rejection loop; however, speed-up is
        # probably marginal at best (i.e ~ 10ms difference per batch)...
        target = len(sampled)
        sampled = sampled[self._burnable_mask(sampled)]
        seed = treatment_seed
        while len(sampled) < target and seed - treatment_seed < MAX_IGNITION_RESAMPLE:
            seed += 1
            extra = env.sample_points(size=n_points, method=method, rng=seed)
            extra = extra.explode(index_parts=False).reset_index(drop=True)
            extra = extra[self._burnable_mask(extra)]
            sampled = pd.concat([sampled, extra], ignore_index=True)
        sampled = sampled.iloc[:target].reset_index(drop=True)

        self.ignitions = gpd.GeoDataFrame(geometry=sampled, crs=gs.crs)

    def _locate_ignitions(self) -> None:
        xs = self.ignitions.geometry.x.to_numpy()
        ys = self.ignitions.geometry.y.to_numpy()
        rows, cols = rowcol(self.profile["transform"], xs, ys)
        self.ignitions["row"] = rows
        self.ignitions["col"] = cols

    def _collate_treatments(self, treatment_area: Polygon) -> None:
        assert {"baseline", "treatment"} <= self.fuels.keys(), (
            "treatment_area needs fuels_paths as {'baseline': ..., 'treatment': ...}"
        )
        # treatment layer becomes: treatment fuels inside the area, baseline outside
        base, treat = self.fuels["baseline"], self.fuels["treatment"]

        _, H, W = base["fbfm"].shape
        treated = geometry_mask(
            [treatment_area],
            out_shape=(H, W),
            transform=self.profile["transform"],
            invert=True,
        )
        treated = torch.from_numpy(treated)

        for key, arr in base.items():
            treat[key][:, ~treated] = arr[:, ~treated]
