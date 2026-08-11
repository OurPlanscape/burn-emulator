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

from burn_emulator.constants import DEFAULT_DTYPE, NO_DATA, Path
from burn_emulator.utils import cache_inputs


class IgnitionDataset(Dataset):
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
        num_classes: int = 4, # unburned, surface, passive crown, active crown
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
        
        self._set_ignitions(ignitions_path,
                            treatment_area,
                            treatment_buff,
                            treatment_seed,
                            ignition_density)
        self.fuels, self.topos, self.masks, self.profile = cache_inputs(
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # occasionally ignitions are at a border
        S = self.window_size // 2
        Off = self.window_size % 2
        _, H, W = self.fuels[fkey]["fbfm"].shape
        ymin, ymax = max(0, y - S), min(y + S + Off, H)
        xmin, xmax = max(0, x - S), min(x + S + Off, W)
        yslc = slice(ymin, ymax)
        xslc = slice(xmin, xmax)

        # the mask is fbfm shaped. see utils.cache_intputs
        mask = self.masks[fkey][:, yslc, xslc]
        ydiff = self.window_size - mask.shape[1]
        xdiff = self.window_size - mask.shape[2]

        if ydiff > 0:
            ypad = (0, 0, ydiff, 0) if ymin == 0 else (0, 0, 0, ydiff)
            mask = F.pad(mask, ypad, mode="constant", value=0)
        if xdiff > 0:
            xpad = (xdiff, 0, 0, 0) if xmin == 0 else (0, xdiff, 0, 0)
            mask = F.pad(mask, xpad, mode="constant", value=0)

        # slopes (after caching) 0: x flow 1: y flow
        arrX = []
        for _key, values in self.topos.items():
            arr = values[:, yslc, xslc]
            if ydiff > 0:
                arr = F.pad(arr, ypad, mode="constant", value=NO_DATA)
            if xdiff > 0:
                arr = F.pad(arr, xpad, mode="constant", value=NO_DATA)
            arrX.append(arr)

        # one hots should be padded with 0 not -1
        for key, values in self.fuels[fkey].items():
            arr = values[:, yslc, xslc]
            no_data = 0 if key == "fbfm" else NO_DATA
            if ydiff > 0:
                arr = F.pad(arr, ypad, mode="constant", value=no_data)
            if xdiff > 0:
                arr = F.pad(arr, xpad, mode="constant", value=no_data)
            if key == "fbfm":
                arr_fbfm = arr
            else:
                arrX.append(arr)

        arrX = torch.concat(arrX)
        if self.circle_mask:
            arrX[:, ~self.cmask] = NO_DATA
            arr_fbfm[:, ~self.cmask] = 0
            mask = mask & self.cmask
        arrX = torch.concat([arrX, arr_fbfm])

        # burns are not necessary for inference
        if self.burn_paths is not None:
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
            arrY = torch.concat(arrY)
            if self.one_hot:
                arrY = F.one_hot(arrY.long(), num_classes=self.num_classes)
                arrY = arrY.permute(0, 3, 1, 2).flatten(0, 1)
            else:
                arrY = arrY >= 1
            if ydiff > 0:
                arrY = F.pad(arrY, ypad, mode="constant", value=0)
            if xdiff > 0:
                arrY = F.pad(arrY, xpad, mode="constant", value=0)

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

    def _set_ignitions(self,
                       ignitions_path=None,
                       treatment_area=None,
                       treatment_buff=None,
                       treatment_seed=42,
                       ignition_density=None):
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
                    if "ignition_number" not in self.ignitions.columns:
                        self.ignitions = self.ignitions.reset_index(names="ignition_number")
                    self.window_geo = None
                case ( ".gpkg" | ".geojson" | ".zip" ):
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
                wind_angle = np.random.uniform(self.wind_range[0], self.wind_range[1], len(self.ignitions))
                self.wind_angles.append(pd.Series(wind_angle, name="upwind_direction"))

    def _set_circle_mask(self) -> torch.Tensor:
        h = w = self.window_size
        cy, cx = (h - 1) / 2, (w - 1) / 2
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=DEFAULT_DTYPE),
            torch.arange(w, dtype=DEFAULT_DTYPE),
            indexing="ij",
        )
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radius = min(h, w) / 2
        self.cmask = (dist <= radius)
    
    def _sample_ignitions(self,
                          gs: gpd.GeoSeries,
                          ignition_density: float = 0.0001,
                          treatment_seed: int = 42):
        n_points = round(gs.area.sum() * ignition_density)
        sampled = gs.sample_points(size=n_points, seed=treatment_seed)
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
