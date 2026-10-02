import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import psutil
import pyretechnics.eulerian_level_set as els
import rasterio
from pyretechnics.space_time_cube import SpaceTimeCube
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

from burn_emulator.constants import INPUT_KEYS, RAW_NO_DATA, Path

CUBE_BANDS = 96  # 3 days + 3 hours @ 1 hour/band
CUBE_RESOLUTION = (
    60,  # band_duration: minutes
    30,  # cell_height:   meters
    30,  # cell_width:    meters
)
MAX_DURATION = 8 * 60  # minutes

WIND_SPEED_10M = 10.0  # km/hr (10 km/hr ~= 6.2 mph)
FUEL_MOISTURES = {
    "1hr": 0.05,
    "10hr": 0.10,
    "100hr": 0.15,
    "LH": 0.90,
    "LW": 0.60,
    "FMC": 0.90,
}
DEFAULT_PT_ADJUSTMENTS = {"fuel_spread": 1.0, "weather_spread": 1.0}

# rough per-worker footprint: input arrays (both roles) + SpreadState (~29 B/cell) + output copies
PT_BYTES_PER_CELL = 128


def pt_inputs(raw: dict[str, np.ndarray], pt_adjustments: dict = DEFAULT_PT_ADJUSTMENTS) -> dict:
    # raw: INPUT_KEYS + slope (degrees) + aspect (degrees), RAW_NO_DATA for nodata
    fuel_model = raw["fbfm"].astype("float32")
    fuel_model[(fuel_model == RAW_NO_DATA) | (fuel_model == 0.0)] = 91.0

    cc = raw["cc"].astype("float32")
    cc[cc > RAW_NO_DATA] /= 100  # convert to 0-1

    cbd = raw["cbd"].astype("float32")
    cbd[cbd > RAW_NO_DATA] /= 100  # convert to kg/m^3

    cbh = raw["cbh"].astype("float32")
    cbh[cbh > RAW_NO_DATA] /= 10  # convert to m

    ch = raw["th"].astype("float32")
    ch[ch > RAW_NO_DATA] /= 10  # convert to m

    slope = raw["slope"].astype("float32")  # deg
    slope_nodata = slope == RAW_NO_DATA
    slope = np.tan(np.deg2rad(slope))  # convert from degrees to rise/run
    slope[slope_nodata] = 0.0

    aspect = raw["aspect"].astype("float32")  # degrees
    aspect[aspect == RAW_NO_DATA] = 0.0

    # fuel_moisture units: kg moisture/kg ovendry weight
    return {
        "slope": slope,  # rise/run
        "aspect": aspect,  # degrees clockwise from North
        "fuel_model": fuel_model,  # index in fm.fuel_model_table
        "canopy_cover": cc,  # 0-1
        "canopy_height": ch,  # m
        "canopy_base_height": cbh,  # m
        "canopy_bulk_density": cbd,  # kg/m^3
        "wind_speed_10m": WIND_SPEED_10M,
        "fuel_moisture_dead_1hr": FUEL_MOISTURES["1hr"],
        "fuel_moisture_dead_10hr": FUEL_MOISTURES["10hr"],
        "fuel_moisture_dead_100hr": FUEL_MOISTURES["100hr"],
        "fuel_moisture_live_herbaceous": FUEL_MOISTURES["LH"],
        "fuel_moisture_live_woody": FUEL_MOISTURES["LW"],
        "foliar_moisture": FUEL_MOISTURES["FMC"],
        "fuel_spread_adjustment": pt_adjustments["fuel_spread"],  # >= 0.0
        "weather_spread_adjustment": pt_adjustments["weather_spread"],  # >= 0.0
    }


def simulate(
    inputs: dict,
    ignition_point: tuple[int, int],
    upwind_direction: float,
    max_duration: int = MAX_DURATION,
) -> dict:
    height, width = inputs["fuel_model"].shape
    cube_shape = (CUBE_BANDS, height, width)
    space_time_cubes = {name: SpaceTimeCube(cube_shape, value) for name, value in inputs.items()}
    space_time_cubes["upwind_direction"] = SpaceTimeCube(cube_shape, float(upwind_direction))

    start_time = 0  # minutes
    spread_state = els.SpreadState(cube_shape).ignite_cell(
        (int(ignition_point[0]), int(ignition_point[1]))
    )
    spread_state.set_start_time(start_time)

    runtime_start = perf_counter()
    fire_spread_results = els.spread_fire_with_phi_field(
        space_time_cubes,
        spread_state,
        CUBE_RESOLUTION,
        start_time,
        max_duration,
        surface_lw_ratio_model="rothermel",
    )
    runtime = perf_counter() - runtime_start
    output_matrices = fire_spread_results["spread_state"].get_full_matrices()

    return {
        "fire_type": output_matrices["fire_type"],  # 0 unburned | 1 surface | 2 passive | 3 active
        "stop_condition": fire_spread_results["stop_condition"],
        "runtime": runtime,
    }


def _read_raw(path: Path, bounds: tuple, shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=window).astype("float32")
        nodata = src.nodata
    assert arr.shape == shape, f"{path} window {arr.shape} != dataset {shape}"
    if nodata is not None:
        arr[arr == nodata] = RAW_NO_DATA
    arr[np.isnan(arr)] = RAW_NO_DATA
    return arr


def read_raw_inputs(fuels_path: Path, topo_path: Path, profile: dict) -> dict[str, np.ndarray]:
    shape = (profile["height"], profile["width"])
    bounds = array_bounds(*shape, profile["transform"])
    raw = {name: _read_raw(Path(fuels_path) / f"{name}.tif", bounds, shape) for name in INPUT_KEYS}
    raw["slope"] = _read_raw(Path(topo_path) / "slope_degrees.tif", bounds, shape)
    raw["aspect"] = _read_raw(Path(topo_path) / "aspect.tif", bounds, shape)
    return raw


def collate_treatment(
    baseline: dict[str, np.ndarray], treatment: dict[str, np.ndarray], treatment_area, profile: dict
) -> dict[str, np.ndarray]:
    # treatment fuels inside the area, baseline outside (same as VarLoc._collate_treatments)
    treated = geometry_mask(
        [treatment_area],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        invert=True,
    )
    return {
        name: np.where(treated, treatment[name], arr) if name in INPUT_KEYS else arr
        for name, arr in baseline.items()
    }


_WORKER_STATE: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER_STATE.update(state)


def resolve_workers(n_cells: int, max_workers: int | None = None) -> int:
    # every worker simulates over the full extent, so cap by memory as well as cpus
    if max_workers:
        return max_workers
    by_memory = psutil.virtual_memory().available // (n_cells * PT_BYTES_PER_CELL)
    return max(1, min(os.cpu_count() or 1, int(by_memory)))


def pool(state: dict, max_workers: int) -> ProcessPoolExecutor:
    # spawn: the parent may already hold a CUDA context
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(state,),
    )


def change_task(ignition_point: tuple[int, int], winds: dict[str, float]) -> dict:
    # state: {"inputs": {"baseline": ..., "treatment": ...}, "keep_mask": bool (H, W)}
    sims = {
        role: simulate(inputs, ignition_point, winds[role])
        for role, inputs in _WORKER_STATE["inputs"].items()
    }
    baseline_ft, treatment_ft = sims["baseline"]["fire_type"], sims["treatment"]["fire_type"]
    result = {
        "runtime": sum(s["runtime"] for s in sims.values()),
        "stop_conditions": [s["stop_condition"] for s in sims.values()],
        "touches": False,
    }

    burned = (baseline_ft != 0) | (treatment_ft != 0)
    if not (burned & _WORKER_STATE["keep_mask"]).any():
        return result

    # crowned (passive or active) is class >= 2; only ship the burned bbox back
    rows, cols = np.nonzero(burned)
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    baseline_crowned = baseline_ft[y0:y1, x0:x1] >= 2
    treatment_crowned = treatment_ft[y0:y1, x0:x1] >= 2
    result.update(
        touches=True,
        origin=(int(y0), int(x0)),
        to_crown=~baseline_crowned & treatment_crowned,
        from_crown=baseline_crowned & ~treatment_crowned,
    )
    return result


def fire_type_task(ignition_point: tuple[int, int], wind: float, out_file: Path) -> dict:
    # state: {"inputs": inputs, "profile": profile with count = out channels}
    sim = simulate(_WORKER_STATE["inputs"], ignition_point, wind)
    profile = _WORKER_STATE["profile"]
    one_hot = np.stack([sim["fire_type"] == c for c in range(profile["count"])]).astype("float32")
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(one_hot)
    return {"runtime": sim["runtime"], "stop_conditions": [sim["stop_condition"]]}
