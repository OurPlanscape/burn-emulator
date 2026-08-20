import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from shapely import Polygon
from torch.utils.data import Dataset

from burn_emulator.constants import NO_DATA, Path
from burn_emulator.utils import cache_bfuels_inputs, circle_mask


class VarLoc(Dataset):
    def __init__(
        self,
        ignitions_path: Path | None,
        fuels_paths: list[Path],
        burn_paths: list[Path] | None,
        wind_ang_paths: list[Path] | None,
        topo_path: Path,
        stats_path: Path,
        burn_times: list[int] = None,
        window_size: int = 129,
        jitter: int | None = None,
        treatment_area: Polygon | None = None,
        treatment_buff: Polygon | None = None,
        treatment_seed: int = 42,
        ignition_density: float | None = None,
        wind_seed: int = 42,
        wind_range: list[float] | None = None,
        circle_mask: bool = True,
        one_hot: bool = True,
        num_classes: int = 4,  # unburned, surface, passive crown, active crown
    ) -> None:
        self.fuels_paths = [Path(p) for p in fuels_paths]

        self.burn_paths = burn_paths
        if self.burn_paths is not None:
            assert len(self.fuels_paths) == len(self.burn_paths), (
                "Each fuel layer set should have a single associated burn directory"
            )
            self.burn_paths = [Path(p) for p in burn_paths]

        self.wind_ang_paths = wind_ang_paths
        if self.wind_ang_paths is not None:
            assert len(self.fuels_paths) == len(self.wind_ang_paths), (
                "Each fuel layer set should have a single associated wind angle directory"
            )
            self.wind_ang_paths = [Path(p) for p in wind_ang_paths]

        self.topo_path = Path(topo_path)
        self.stats_path = Path(stats_path)

        self.burn_times = [str(bt) for bt in burn_times] if burn_times else [480]
        self.window_size = window_size
        self.jitter = jitter

        self.treatment_area = treatment_area
        self.treatment_buff = treatment_buff
        self.treatment_seed = treatment_seed
        self.ignition_density = ignition_density

        self.wind_seed = wind_seed
        self.wind_range = wind_range
        self.circle_mask = circle_mask
        self.one_hot = one_hot
        self.num_classes = num_classes

        self._set_ignitions(ignitions_path, treatment_buff, treatment_seed, ignition_density)
        self.fuels, self.topos, self.masks, self.profile = cache_bfuels_inputs(
            self.fuels_paths, self.topo_path, self.stats_path, self.window_geo
        )

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

    def __getitem__(self, idx: int):
        if self.burn_paths is None:
            sidx = idx
            bidx = 0
        else:
            sidx = idx // len(self.burn_paths)
            bidx = idx % len(self.burn_paths)
            burn_path = self.burn_paths[bidx]

        fkey = self.fuels_paths[bidx].stem
        ignition = self.ignitions.iloc[sidx]

        y = int(ignition["row"].item())
        x = int(ignition["col"].item())

        if self.wind_angles is not None:
            wind = self.wind_angles[bidx].iloc[sidx]

        if self.jitter is not None:
            y += np.random.randint(-(self.jitter + 1), self.jitter)
            x += np.random.randint(-(self.jitter + 1), self.jitter)

        ymin, ymax, xmin, xmax, yslc, xslc = self._compute_bounds(fkey, y, x)
        ydiff, xdiff, ypad, xpad = self._compute_padding(ymin, ymax, xmin, xmax)

        mask = self._build_mask(fkey, yslc, xslc, ydiff, xdiff, ypad, xpad)
        arrX = self._build_arrx(fkey, yslc, xslc, ydiff, xdiff, ypad, xpad)

        # burns are not necessary for inference
        if self.burn_paths is not None:
            arrY = self._build_arry(ignition, burn_path, yslc, xslc, ydiff, xdiff, ypad, xpad)
            return {
                "x": arrX,
                "y": arrY,
                "wind": wind,
                "mask": mask,
            }
        else:
            return {
                "x": arrX,
                "wind": wind,
                "mask": mask,
                "pdiffs": (ydiff, xdiff),
                "bounds": (ymin, ymax, xmin, xmax),
                "indxes": (sidx, bidx),
            }

    def _compute_bounds(self, fkey: str, y: int, x: int) -> tuple[int, int, int, int, slice, slice]:
        # occasionally ignitions are at a border
        S = self.window_size // 2
        Off = self.window_size % 2
        _, H, W = self.fuels[fkey]["fbfm"].shape
        ymin, ymax = max(0, y - S), min(y + S + Off, H)
        xmin, xmax = max(0, x - S), min(x + S + Off, W)
        yslc = slice(ymin, ymax)
        xslc = slice(xmin, xmax)
        return ymin, ymax, xmin, xmax, yslc, xslc

    def _compute_padding(
        self, ymin: int, ymax: int, xmin: int, xmax: int
    ) -> tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]:
        ydiff = self.window_size - (ymax - ymin)
        xdiff = self.window_size - (xmax - xmin)
        ypad = (0, 0, ydiff, 0) if ymin == 0 else (0, 0, 0, ydiff)
        xpad = (xdiff, 0, 0, 0) if xmin == 0 else (0, xdiff, 0, 0)
        return ydiff, xdiff, ypad, xpad

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

    def _set_ignitions(
        self, ignitions_path=None, treatment_buff=None, treatment_seed=42, ignition_density=None
    ):
        if ignitions_path is None:
            buffer_gs = gpd.GeoSeries([treatment_buff])
            self._sample_ignitions(buffer_gs, ignition_density, treatment_seed)
        else:
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
                    self.window_geo = None
                case ".gpkg" | ".geojson" | ".zip":
                    gdf = gpd.read_file(self.ignitions)
                    self._sample_ignitions(gdf.geometry, ignition_density, self.treatment_seed)
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
                    self.window_geo = None

    def _set_wind_angles(self):
        self.wind_angles = []
        assert self.wind_ang_paths is not None or self.wind_range is not None, (
            "Either wind_ang_paths or wind_range must be provided to set wind angles"
        )
        if self.wind_ang_paths:
            for wind_ang_path in self.wind_ang_paths:
                self.wind_angles.append(pd.read_csv(wind_ang_path)["upwind_direction"])
        if self.wind_range is not None:
            np.random.seed(self.wind_seed)
            for _ in range(len(self.fuels_paths)):
                wind_angle = np.random.uniform(
                    self.wind_range[0], self.wind_range[1], len(self.ignitions)
                )
                self.wind_angles.append(pd.Series(wind_angle, name="upwind_direction"))

    def _set_circle_mask(self) -> torch.Tensor:
        self.cmask = circle_mask(self.window_size)

    def _sample_ignitions(
        self, gs: gpd.GeoSeries, ignition_density: float = 0.0001, treatment_seed: int = 42
    ):
        n_points = round(gs.area.sum() * ignition_density)
        sampled = gs.sample_points(size=n_points, rng=treatment_seed)
        sampled = sampled.explode(index_parts=False).reset_index(drop=True)
        self.ignitions = gpd.GeoDataFrame(geometry=sampled, crs=gs.crs)
        self.window_geo = gs.union_all().envelope

    def _locate_ignitions(self):
        xs = self.ignitions.geometry.x.to_numpy()
        ys = self.ignitions.geometry.y.to_numpy()
        rows, cols = rowcol(self.profile["transform"], xs, ys)
        self.ignitions["row"] = rows
        self.ignitions["col"] = cols

    def _collate_treatments(self, treatment_area: Polygon):
        # TODO: get a better naming convention
        # we're assuming the second fuel layer is the treatment layer
        fkey0 = self.fuels_paths[0].stem
        fkey1 = self.fuels_paths[1].stem

        _, H, W = self.fuels[fkey0]["fbfm"].shape
        treated = geometry_mask(
            [treatment_area],
            out_shape=(H, W),
            transform=self.profile["transform"],
            invert=True,
        )
        treated = torch.from_numpy(treated)

        for key, arr in self.fuels[fkey1].items():
            self.fuels[fkey0][key][:, treated] = arr[:, treated]
        del self.fuels[fkey1]


class VarLocDiff(VarLoc):
    def __init__(self, percent_no_change=None, **kwargs):
        super().__init__(**kwargs)
        self.percent_no_change = percent_no_change

    def __len__(self) -> int:
        return len(self.ignitions)

    def __getitem__(self, idx: int):
        sidx = idx
        ignition = self.ignitions.iloc[sidx]

        y = int(ignition["row"].item())
        x = int(ignition["col"].item())

        if self.wind_angles is not None:
            wind = self.wind_angles[0].iloc[sidx]

        if self.jitter is not None:
            y += np.random.randint(-(self.jitter + 1), self.jitter)
            x += np.random.randint(-(self.jitter + 1), self.jitter)

        fkey0 = self.fuels_paths[0].stem
        ymin, ymax, xmin, xmax, yslc, xslc = self._compute_bounds(fkey0, y, x)
        ydiff, xdiff, ypad, xpad = self._compute_padding(ymin, ymax, xmin, xmax)

        mask = self._build_mask(fkey0, yslc, xslc, ydiff, xdiff, ypad, xpad)

        fkey1 = self.fuels_paths[1].stem

        if self.percent_no_change is not None and np.random.rand() < self.percent_no_change:
            fkey = fkey0  # maintain baseline only
            arr = self._build_arrx(fkey, yslc, xslc, ydiff, xdiff, ypad, xpad)
            arrX = torch.stack([arr, arr])
            arrY = torch.zeros((self.num_classes, self.window_size, self.window_size))
            arrY[0] = 1
            return {
                "x": arrX,
                "y": arrY,
                "wind": wind,
                "mask": mask,
            }
        else:
            arrX = torch.stack(
                [
                    self._build_arrx(fkey0, yslc, xslc, ydiff, xdiff, ypad, xpad),
                    self._build_arrx(fkey1, yslc, xslc, ydiff, xdiff, ypad, xpad),
                ]
            )

            if self.burn_paths is not None:
                arrY_baseline = self._build_arry_raw(ignition, self.burn_paths[0], yslc, xslc)
                arrY_treatment = self._build_arry_raw(ignition, self.burn_paths[1], yslc, xslc)

                # fire_type classes: 0 unburned | 1 surface | 2 passive crown | 3 active crown
                # crowned (passive or active) is class >= 2
                baseline_crowned = arrY_baseline >= 2
                treatment_crowned = arrY_treatment >= 2

                # 0 (no change) | 1 (surface or nonburned to passive or active)
                # | 2 (passive or active to surface or non-burned)
                arrY = torch.zeros_like(arrY_baseline)
                arrY[~baseline_crowned & treatment_crowned] = 1
                arrY[baseline_crowned & ~treatment_crowned] = 2

                if self.one_hot:
                    arrY = F.one_hot(arrY.long(), num_classes=self.num_classes)
                    arrY = arrY.permute(0, 3, 1, 2).flatten(0, 1)
                else:
                    arrY = arrY >= 1
                arrY = self._pad(arrY, ydiff, xdiff, ypad, xpad, value=0)

                return {
                    "x": arrX,
                    "y": arrY,
                    "wind": wind,
                    "mask": mask,
                }
            else:
                return {
                    "x": arrX,
                    "wind": wind,
                    "mask": mask,
                    "pdiffs": (ydiff, xdiff),
                    "bounds": (ymin, ymax, xmin, xmax),
                    "indxes": (sidx, torch.empty(0)),
                }
