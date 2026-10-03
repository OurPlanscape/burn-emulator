import contextlib
import random
import resource
import shutil
import threading

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import pyretechnics.fuel_models as fm
import rasterio as rio
import ray
from pyretechnics.space_time_cube import SpaceTimeCube
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from burn_emulator.config import check_data_version, inputs_version, wind_range
from burn_emulator.constants import (
    INPUT_KEYS,
    RAW_NO_DATA,
    TARGET_CRS,
    TOPO_DIR_PREFIX,
    TRAINING_DATA_DIR,
    VARLOCS_GPKG,
    FUELS_DIR_PREFIX,
)
from burn_emulator.pt import CUBE_BANDS, DEFAULT_PT_ADJUSTMENTS, pt_inputs, simulate

TREATMENTS = ["baseline", "legalmax"]

RAY_MEM = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_DURATIONS = [8 * 60]

MEM_SAMPLE_INTERVAL = 5  # seconds
GIB = 1024**3


def _process_tree_pss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total += p.memory_full_info().pss
    return total


def _sample_memory(peak: dict):
    peak["tree_pss"] = max(peak["tree_pss"], _process_tree_pss(psutil.Process()))
    peak["system_used"] = max(peak["system_used"], psutil.virtual_memory().used)


def _monitor_memory(stop: threading.Event, peak: dict):
    while not stop.wait(MEM_SAMPLE_INTERVAL):
        _sample_memory(peak)


def _print_memory_summary(peak: dict):
    driver_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KiB on Linux
    print("Memory Summary:")
    print(f"   Peak Total (driver + Ray, PSS): {peak['tree_pss'] / GIB:.2f} GiB")
    print(f"   Peak Driver RSS: {driver_peak_rss / GIB:.2f} GiB")
    print(
        f"   Peak System Used: {peak['system_used'] / GIB:.2f} GiB "
        f"of {psutil.virtual_memory().total / GIB:.2f} GiB"
    )


def _load_varloc_geom(varloc: str):
    gdf = gpd.read_file(VARLOCS_GPKG)
    rows = gdf[gdf["varloc"] == varloc]
    if rows.empty:
        raise ValueError(f"varloc {varloc!r} not found in {VARLOCS_GPKG}")
    return rows.union_all()


def _read_masked_window(src_path, geom, bounds: tuple):
    with rio.open(src_path) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        transform = src.window_transform(window)
        arr = src.read(1, window=window).astype("float32")

        inside = geometry_mask([geom], out_shape=arr.shape, transform=transform, invert=True)
        arr[~inside] = RAW_NO_DATA
        arr[np.isnan(arr)] = RAW_NO_DATA

        profile = src.profile | {
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": transform,
            "dtype": "int16",
            "nodata": RAW_NO_DATA,
        }
    return arr.astype("int16"), profile


def attach_shared_array(shared_array_ref):
    return ray.get(shared_array_ref)


def is_burnable(fuel_model_cube, y, x):
    fuel_model_number = fuel_model_cube.get(0, y, x)
    return fm.fuel_model_exists(fuel_model_number) and not (91 <= fuel_model_number <= 99)


def outside_buffer(fuel_model_cube, point: tuple, buffered_gdf, transform):
    #  single point = (row, col)
    x, y = rio.transform.xy(transform=transform, rows=point[0], cols=point[1])
    points_geom = gpd.points_from_xy(x=[x], y=[y])

    return points_geom.within(buffered_gdf["geometry"][0])[0]  # return first element of list


def count_allowable_cells(fuel_model_array, buffered_gdf, transform) -> int:
    # vectorized is_burnable + outside_buffer; sampling never terminates when this is 0
    fuel_models = np.unique(fuel_model_array)
    burnable_models = [n for n in fuel_models if fm.fuel_model_exists(n) and not (91 <= n <= 99)]
    burnable = np.isin(fuel_model_array, burnable_models)
    inside = geometry_mask(
        buffered_gdf["geometry"], out_shape=fuel_model_array.shape, transform=transform, invert=True
    )
    return int(np.count_nonzero(burnable & inside))


def sample_ignited_cells_buffered(fuel_model_cube, num_ignitions, buffered_gdf, transform, seed):
    _bands, rows, cols = fuel_model_cube.shape

    random.seed(seed)

    ignited_cells = []
    bad_count = 0
    while len(ignited_cells) < num_ignitions:
        y = random.randrange(rows)
        x = random.randrange(cols)
        if is_burnable(fuel_model_cube, y, x) and outside_buffer(
            fuel_model_cube, (y, x), buffered_gdf, transform
        ):
            ignited_cells.append((y, x))
        else:
            bad_count += 1

    print(
        f"Failed {bad_count} unallowable locations to get {len(ignited_cells)} allowable locations."
    )

    return ignited_cells


def preserve_ignition_locations(ignitions: list, transform, out_file):
    df = pd.DataFrame(ignitions, columns=["row", "col"])
    df["x"], df["y"] = rio.transform.xy(transform, df["row"].values, df["col"].values)
    df["ignition_number"] = df.index
    df.to_csv(out_file, index=False)
    print("Preserved ignition locations")


def write_raster(training_data_dir, treatment, ignition_number, profile, ft_array):
    out_dir = training_data_dir / treatment / str(ignition_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rio.open(out_dir / "fire_type.tif", "w", **profile) as dst:
        dst.write(ft_array, 1)


@ray.remote
def run_ignition_ray(
    treatment,
    md,
    ignition_number,
    ignition_point,
    shared_array_refs,
    upwind_direction,
    training_data_dir,
    profile,
    collate_ignitions,
):
    print(f"{treatment=}; {ignition_number=}; {md=}; {upwind_direction=}")
    shared_arrays = {
        name: attach_shared_array(shared_array_ref)
        for (name, shared_array_ref) in shared_array_refs.items()
    }
    sim = simulate(shared_arrays, ignition_point, upwind_direction, md)
    stop_condition = sim["stop_condition"]  # "max duration reached" or "no burnable cells"
    fire_type = sim["fire_type"]

    num_burned_cells = np.count_nonzero(fire_type)  # cells
    acres_burned = num_burned_cells / 4.5  # acres
    simulation_runtime = sim["runtime"]  # seconds
    runtime_per_burned_cell = (
        1000.0 * simulation_runtime / num_burned_cells if num_burned_cells > 0 else 0.0
    )  # ms/cell; some FBFMs at short burn periods might not burn anything

    print("   Acres Burned: " + str(acres_burned))
    print("   Total Runtime: " + str(simulation_runtime) + " seconds")
    print("   Runtime Per Burned Cell: " + str(runtime_per_burned_cell) + " ms/cell")
    print("   Stop Condition: " + stop_condition)

    result = {
        "ignition_number": ignition_number,
        "treatment": treatment,
        "max_duration": md,
        "acres_burned": acres_burned,
        "total_runtime": runtime_per_burned_cell,
        "stop_condition": stop_condition,
        "upwind_direction": upwind_direction,
    }
    if collate_ignitions:
        result["burned"] = fire_type != 0
    else:
        write_raster(training_data_dir, treatment, ignition_number, profile, fire_type)
    return result


def ignite(
    varloc: str,
    data_version: str,
    num_ignitions: int = 5000,
    collate_ignitions: bool = False,
    overwrite: bool = False,
    buffer_dist: float = -2000,  # m; ignitions can't be within this distance of the edge
    max_durations: list[int] = DEFAULT_MAX_DURATIONS,
    upwind_direction_quadrant: list[float] | None = None,  # default: wind_directions.csv
    seed: int = 42,
    pt_adjustments: dict = DEFAULT_PT_ADJUSTMENTS,
    **kwargs,
) -> None:
    # resolved before anything is deleted or written; must match the range bundles bake in
    if upwind_direction_quadrant is None:
        upwind_direction_quadrant = wind_range(varloc)
    print(f"{varloc}: upwind directions {upwind_direction_quadrant}")

    data_version = check_data_version(data_version)
    fuels = inputs_version(data_version)
    fuels_source_dir = TRAINING_DATA_DIR / f"{FUELS_DIR_PREFIX}_{fuels}"
    topo_source_dir = TRAINING_DATA_DIR / f"{TOPO_DIR_PREFIX}_{fuels}"
    for src_dir in (fuels_source_dir, topo_source_dir):
        if not src_dir.is_dir():
            raise FileNotFoundError(f"{src_dir} not found for data_version {data_version}")

    training_data_dir = TRAINING_DATA_DIR / varloc / data_version
    if training_data_dir.exists() and not overwrite:
        raise FileExistsError(f"{training_data_dir} already exists; pass overwrite to replace it")

    geom = _load_varloc_geom(varloc)
    bounds = geom.bounds

    aoi_gdf = gpd.GeoDataFrame(geometry=[geom], crs=TARGET_CRS)
    buffered_geom = aoi_gdf.buffer(buffer_dist)  # ignitions can't be within this of the edge
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geom)

    # checked in memory before anything is deleted or written, so a varloc with no fuels
    # coverage fails without leaving (or wiping) a training data dir
    fbfm_src = fuels_source_dir / TREATMENTS[0] / "fbfm.tif"
    if not fbfm_src.is_file():
        raise FileNotFoundError(f"{fbfm_src} not found")
    fbfm_arr, fbfm_profile = _read_masked_window(fbfm_src, geom, bounds)
    num_allowable = count_allowable_cells(fbfm_arr, buffered_gdf, fbfm_profile["transform"])
    if num_allowable == 0:
        raise ValueError(
            f"{varloc}: no burnable cells more than {-buffer_dist} m inside the varloc boundary; "
            f"check fuels coverage in {fuels_source_dir}"
        )
    print(f"{varloc}: {num_allowable} allowable ignition cells")
    del fbfm_arr

    if training_data_dir.exists():
        shutil.rmtree(training_data_dir)

    fuels_files = {}
    for treatment in TREATMENTS:
        fuels_files[treatment] = {}
        for r in INPUT_KEYS:
            src_path = fuels_source_dir / treatment / f"{r}.tif"
            if not src_path.is_file():
                raise FileNotFoundError(f"{src_path} not found")
            arr, profile = _read_masked_window(src_path, geom, bounds)

            dst_path = training_data_dir / f"{treatment}_fuels" / f"{r}.tif"
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rio.open(dst_path, "w", **profile) as dst:
                dst.write(arr, 1)
            fuels_files[treatment][r] = dst_path

    topo_dir = training_data_dir / "topo"
    topo_dir.mkdir(parents=True, exist_ok=True)
    for name in ("aspect", "slope_degrees"):
        arr, profile = _read_masked_window(topo_source_dir / f"{name}.tif", geom, bounds)
        with rio.open(topo_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(arr, 1)

    template_raster = rio.open(fuels_files[TREATMENTS[0]]["fbfm"])
    template_array = template_raster.read(1).astype("float32")
    template_profile = dict(template_raster.profile)

    cube_shape = (CUBE_BANDS, template_raster.height, template_raster.width)

    ignition_locations = sample_ignited_cells_buffered(
        fuel_model_cube=SpaceTimeCube(cube_shape, template_array),
        num_ignitions=num_ignitions,
        buffered_gdf=buffered_gdf,
        transform=template_raster.transform,
        seed=seed,
    )
    preserve_ignition_locations(
        ignitions=ignition_locations,
        transform=template_raster.transform,
        out_file=training_data_dir / "ignition_locations.csv",
    )

    print("Starting Pyretechnics")
    ray.init()

    mem_peak = {"tree_pss": 0, "system_used": 0}
    mem_stop = threading.Event()
    mem_thread = threading.Thread(target=_monitor_memory, args=(mem_stop, mem_peak), daemon=True)
    mem_thread.start()

    np.random.seed(seed)
    upwind_directions = np.random.randint(
        low=upwind_direction_quadrant[0],
        high=upwind_direction_quadrant[1],
        size=len(ignition_locations),
    )

    for treatment in TREATMENTS:
        (training_data_dir / treatment).mkdir(parents=True, exist_ok=True)

        raw = {r: rio.open(fuels_files[treatment][r]).read(1) for r in INPUT_KEYS}
        raw["slope"] = rio.open(topo_dir / "slope_degrees.tif").read(1)
        raw["aspect"] = rio.open(topo_dir / "aspect.tif").read(1)
        shared_array_refs = {
            name: ray.put(value) for name, value in pt_inputs(raw, pt_adjustments).items()
        }

        pending = [
            run_ignition_ray.options(memory=RAY_MEM).remote(
                treatment,
                md,
                ignition_number,
                ignition_point,
                shared_array_refs,
                upwind_directions[ignition_number],
                training_data_dir,
                template_profile,
                collate_ignitions,
            )
            for ignition_number, ignition_point in enumerate(ignition_locations)
            for md in max_durations
        ]

        rows = []
        burned_sum = np.zeros((template_raster.height, template_raster.width), dtype=np.float32)
        while pending:
            done, pending = ray.wait(pending, num_returns=min(100, len(pending)))
            for result in ray.get(done):
                burned = result.pop("burned", None)
                if burned is not None:
                    burned_sum += burned
                rows.append(result)

        outputs_columns = [
            "ignition_number",
            "treatment",
            "max_duration",
            "acres_burned",
            "total_runtime",
            "stop_condition",
            "upwind_direction",
        ]
        csv_outputs = pd.DataFrame(rows)[outputs_columns].sort_values(
            ["ignition_number", "max_duration"]
        )
        csv_outputs.to_csv(training_data_dir / treatment / "outputs_table.csv", index=False)

        if collate_ignitions:
            cbp = burned_sum / len(rows)
            cbp_profile = template_profile | {"dtype": "float32", "count": 1}
            with rio.open(training_data_dir / treatment / "cbp.tif", "w", **cbp_profile) as dst:
                dst.write(cbp, 1)
            print(f"Wrote collated CBP raster for {treatment}")

    mem_stop.set()
    mem_thread.join()
    _sample_memory(mem_peak)
    ray.shutdown()
    _print_memory_summary(mem_peak)
