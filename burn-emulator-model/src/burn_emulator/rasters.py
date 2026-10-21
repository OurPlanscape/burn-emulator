import os
import threading
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from burn_emulator.constants import Path

# path -> CachedRaster once enable_cache() is called (the runner's claim loop); None reads from disk
_cache: dict[str, "CachedRaster"] | None = None
_lock = threading.Lock()


def enable_cache() -> None:
    global _cache
    with _lock:
        if _cache is None:
            _cache = {}


# a raster read whole into memory; read() slices windows from it like rasterio's DatasetReader
class CachedRaster:
    def __init__(self, path: str) -> None:
        self.path = path
        with rasterio.open(path) as src:
            self.data = src.read()
            self._profile = src.profile
            self.transform = src.transform
            self.nodata = src.nodata
            self.res = src.res

    @property
    def profile(self) -> dict:
        return self._profile.copy()

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def window_transform(self, window: Window) -> Any:
        return window_transform(window, self.transform)

    def read(self, indexes: int | None = None, window: Window | None = None) -> np.ndarray:
        data = self.data if indexes is None else self.data[indexes - 1]
        if window is None:
            return data.copy()
        r0, c0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        _, rows, cols = self.data.shape
        if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
            # past the edge: rasterio's own handling
            with rasterio.open(self.path) as src:
                return src.read(indexes, window=window)
        return data[..., r0 : r0 + h, c0 : c0 + w].copy()

    def __enter__(self) -> "CachedRaster":
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def open_raster(path: str | Path) -> Any:
    if _cache is None:
        return rasterio.open(path)
    key = os.fspath(path)
    with _lock:
        raster = _cache.get(key)
        if raster is None:
            raster = _cache[key] = CachedRaster(key)
    return raster


# reads each path into the cache; returns the bytes held
def preload(paths: list[str | Path]) -> int:
    enable_cache()
    return sum(open_raster(p).nbytes for p in paths)
