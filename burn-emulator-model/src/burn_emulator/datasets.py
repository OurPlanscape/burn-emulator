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
        ignitions_path: Path | list[Path],
        fuels_paths: list[Path],
        burn_paths: list[Path],
        topo_path: Path,
        stats_path: Path,
        burn_times: list[int] = None,
        chip_size: int = 129,
        jitter: int | None = 1,
        treatment_area: Polygon | None = None, 
        ignition_density: float | None = None,
        circle_mask: bool = True
    ) -> None:
        self.fuels_paths = [Path(p) for p in fuels_paths]
        self.burn_paths = [Path(p) for p in burn_paths] if burn_paths else None
        if self.burn_paths is not None:
            assert len(self.fuels_paths) == len(self.burn_paths), (
                "Each fuel layer set should have a single associated burn directory"
            )

        self.topo_path = Path(topo_path)
        self.stats_path = Path(stats_path)

        self.burn_times = [str(bt) for bt in burn_times] if burn_times else [480]
        self.chip_size = chip_size
        self.jitter = jitter
        self.treatment_area = treatment_area
        self.ignition_density = ignition_density
        self.circle_mask = circle_mask
        self._set_ignitions(ignitions_path)
        self.fuels, self.topos, self.masks, self.profile = cache_inputs(
            self.fuels_paths, self.topo_path, self.stats_path, self.window
        )

        if self.treatment_area is not None:
            self._collate_treatments(treatment_area)
        
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

        if self.jitter is not None:
            y += np.random.randint(-(self.jitter + 1), self.jitter)
            x += np.random.randint(-(self.jitter + 1), self.jitter)

        # occasionally ignitions are at a border
        S = self.chip_size // 2
        Off = self.chip_size % 2
        _, H, W = self.fuels[fkey]["fbfm"].shape
        ymin, ymax = max(0, y - S), min(y + S + Off, H)
        xmin, xmax = max(0, x - S), min(x + S + Off, W)
        yslc = slice(ymin, ymax)
        xslc = slice(xmin, xmax)

        # the mask is fbfm shaped. see utils.cache_intputs
        mask = self.masks[fkey][:, yslc, xslc]
        ydiff = self.chip_size - mask.shape[1]
        xdiff = self.chip_size - mask.shape[2]

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
            arrX[self.cmask] = NO_DATA
            arr_fbfm[self.cmask] = 0
            mask &= self.cmask
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
                    # windowing was slower overall by a lot so reading entire array
                    # caching 10000 images was unreasonable
                    arr = torch.tensor(src.read())
                arr = arr[:, yslc, xslc]
                if ydiff > 0:
                    arr = F.pad(arr, ypad, mode="constant", value=0)
                if xdiff > 0:
                    arr = F.pad(arr, xpad, mode="constant", value=0)
                arrY.append(arr)
            arrY = torch.concat(arrY)
            arrY = arrY >= 1
            return arrX, arrY, mask
        else:
            return arrX, mask, (ydiff, xdiff), (ymin, ymax, xmin, xmax), (sidx, bidx)

    def _set_ignitions(self, ignitions_path):
        match Path(ignitions_path).suffix:
            case ".csv":
                self.ignitions = pd.read_csv(ignitions_path)
                if "cbp_burn" not in self.ignitions.columns:
                    name = Path(ignitions_path).stem.split("_")[0]
                    self.ignitions.loc[:, "cbp_burn"] = name
                if "ignition_number" not in self.ignitions.columns:
                    self.ignitions = self.ignitions.reset_index(names="ignition_number")
                self.window = None
            case (".gpkg" | ".geojson" | ".zip" ):
                # I'm assuming the shapefile is zipped and in the appropriate EPSG code
                gdf = gpd.read_file(self.ignitions)
                self._sample_ignitions(gdf.geometry)
            case "_":
                ignitions_paths = Path(ignitions_path).glob("**/*.csv")
                self.ignitions = []
                for ignitions_path in sorted(ignitions_paths):
                    name = ignitions_path.stem.split("_")[0]
                    ignition = pd.read_csv(ignitions_path)
                    ignition = ignition.reset_index(names="ignition_number")
                    ignition.loc[:, "cbp_burn"] = name
                    self.ignitions.append(ignition)
                self.ignitions = pd.concat(self.ignitions)
                self.window = None

    def _set_circle_mask(self) -> Tensor:
        h = w = self.chip_size
        cy, cx = (h - 1) / 2, (w - 1) / 2
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=DEFAULT_DTYPE),
            torch.arange(w, dtype=DEFAULT_DTYPE),
            indexing="ij",
        )
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radius = min(h, w) / 2
        self.cmask = (dist <= radius).float()
        return self.cmask.view(1, h, w)
    
    def _sample_ignitions(self, gs: gpd.GeoSeries):
        n_points = round(gs.area.sum() * self.ignition_density)

        sampled = gs.sample_points(size=n_points)
        sampled = sampled.explode(index_parts=False).reset_index(drop=True)

        self.ignitions = gpd.GeoDataFrame(geometry=sampled, crs=gdf.crs)
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
