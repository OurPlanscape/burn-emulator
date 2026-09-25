import random
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import pyretechnics.eulerian_level_set as els
import pyretechnics.fuel_models as fm
import rasterio as rio
import ray
from pyretechnics.space_time_cube import SpaceTimeCube
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from burn_emulator.constants import (
    ASPECT_FILE,
    INPUT_KEYS,
    RAW_NO_DATA,
    SLOPE_FILE,
    TARGET_CRS,
    TRAINING_DATA_DIR,
    VARLOCS_GPKG,
    WEST_FUELS_DIR_PREFIX,
)

TREATMENTS = ["baseline", "legalmax"]

RAY_MEM = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_DURATIONS = [8 * 60]
DEFAULT_UPWIND_DIRECTION_QUADRANT = [225, 270]
DEFAULT_PT_ADJUSTMENTS = {"fuel_spread": 1.0, "weather_spread": 1.0}

WIND_SPEED_10M = 10.0  # km/hr (10 km/hr ~= 6.2 mph)
FUEL_MOISTURES = {
    "1hr": 0.05,
    "10hr": 0.10,
    "100hr": 0.15,
    "LH": 0.90,
    "LW": 0.60,
    "FMC": 0.90,
}

CUBE_RESOLUTION = (
    60,  # band_duration: minutes
    30,  # cell_height:   meters
    30,  # cell_width:    meters
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


def write_raster(training_data_dir, treatment, ignition_number, template_raster, ft_array):
    out_dir = training_data_dir / treatment / str(ignition_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rio.open(out_dir / "fire_type.tif", "w", **template_raster.profile) as dst:
        dst.write(ft_array, 1)


def compute_cbp(results: list, shape: tuple) -> np.ndarray:
    burned_sum = np.zeros(shape, dtype=np.float32)
    for result in results:
        burned_sum += (result["ft_array"] != 0).astype(np.float32)
    return burned_sum / len(results)


@ray.remote
def run_ignition_ray(
    cube_shape,
    treatment,
    md,
    ignition_number,
    ignition_point,
    shared_array_refs,
    upwind_direction,
):
    print(f"{treatment=}; {ignition_number=}; {md=}; {upwind_direction=}")
    spread_state = els.SpreadState(cube_shape).ignite_cell(ignition_point)

    start_time = 0  # minutes

    shared_arrays = {
        name: attach_shared_array(shared_array_ref)
        for (name, shared_array_ref) in shared_array_refs.items()
    }

    space_time_cubes = {
        name: SpaceTimeCube(cube_shape, shared_array)
        for (name, shared_array) in shared_arrays.items()
    }
    space_time_cubes["upwind_direction"] = SpaceTimeCube(
        cube_shape, upwind_direction
    )  # manually add this now that it's not passed via Ray

    spread_state.set_start_time(start_time)

    runtime_start = perf_counter()
    fire_spread_results = els.spread_fire_with_phi_field(
        space_time_cubes,
        spread_state,
        CUBE_RESOLUTION,
        start_time,
        md,
        surface_lw_ratio_model="rothermel",
    )
    runtime_stop = perf_counter()
    stop_condition = fire_spread_results[
        "stop_condition"
    ]  # "max duration reached" or "no burnable cells"
    spread_state = fire_spread_results[
        "spread_state"
    ]  # updated SpreadState object (mutated from inputs)
    output_matrices = spread_state.get_full_matrices()

    num_burned_cells = np.count_nonzero(output_matrices["fire_type"])  # cells
    acres_burned = num_burned_cells / 4.5  # acres
    simulation_runtime = runtime_stop - runtime_start  # seconds
    runtime_per_burned_cell = (
        1000.0 * simulation_runtime / num_burned_cells if num_burned_cells > 0 else 0.0
    )  # ms/cell; some FBFMs at short burn periods might not burn anything

    print("   Acres Burned: " + str(acres_burned))
    print("   Total Runtime: " + str(simulation_runtime) + " seconds")
    print("   Runtime Per Burned Cell: " + str(runtime_per_burned_cell) + " ms/cell")
    print("   Stop Condition: " + stop_condition)

    return {
        "ignition_number": ignition_number,
        "treatment": treatment,
        "max_duration": md,
        "acres_burned": acres_burned,
        "total_runtime": runtime_per_burned_cell,
        "stop_condition": stop_condition,
        "ft_array": output_matrices["fire_type"],
        "upwind_direction": upwind_direction,
    }


def ignite(
    varloc: str,
    data_version: str,
    num_ignitions: int = 5000,
    collate_ignitions: bool = False,
    buffer_dist: float = -2000,  # m; ignitions can't be within this distance of the edge
    max_durations: list[int] = DEFAULT_MAX_DURATIONS,
    upwind_direction_quadrant: list[float] = DEFAULT_UPWIND_DIRECTION_QUADRANT,
    seed: int = 42,
    pt_adjustments: dict = DEFAULT_PT_ADJUSTMENTS,
    **kwargs,
) -> None:
    training_data_dir = TRAINING_DATA_DIR / varloc / data_version

    geom = _load_varloc_geom(varloc)
    bounds = geom.bounds

    west_fuels_dir_matches = list(TRAINING_DATA_DIR.glob(f"{WEST_FUELS_DIR_PREFIX}_*"))
    if len(west_fuels_dir_matches) != 1:
        raise ValueError(
            f"expected exactly one {WEST_FUELS_DIR_PREFIX!r} directory in {TRAINING_DATA_DIR}, "
            f"found {west_fuels_dir_matches}"
        )
    west_fuels_dir = west_fuels_dir_matches[0]

    fuels_files = {}
    for treatment in TREATMENTS:
        fuels_files[treatment] = {}
        for r in INPUT_KEYS:
            matches = list(west_fuels_dir.glob(f"{treatment.capitalize()}_*_{r}.tif"))
            if len(matches) != 1:
                raise ValueError(
                    f"expected exactly one {r!r} file for treatment {treatment!r} in "
                    f"{west_fuels_dir}, found {matches}"
                )
            src_path = matches[0]
            arr, profile = _read_masked_window(src_path, geom, bounds)

            dst_path = training_data_dir / f"{treatment}_FF" / f"{data_version}_{r}.tif"
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rio.open(dst_path, "w", **profile) as dst:
                dst.write(arr, 1)
            fuels_files[treatment][r] = dst_path

    topo_dir = training_data_dir / "topo"
    topo_dir.mkdir(parents=True, exist_ok=True)
    for name, src_path in (("aspect", ASPECT_FILE), ("slope_degrees", SLOPE_FILE)):
        arr, profile = _read_masked_window(src_path, geom, bounds)
        with rio.open(topo_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(arr, 1)

    template_raster = rio.open(fuels_files[TREATMENTS[0]]["fbfm"])
    template_array = template_raster.read(1).astype("float32")

    cube_shape = (
        96,  # bands: 3 days + 3 hours @ 1 hour/band
        template_raster.height,
        template_raster.width,
    )

    aoi_gdf = gpd.GeoDataFrame(geometry=[geom], crs=TARGET_CRS)
    buffered_geom = aoi_gdf.buffer(buffer_dist)  # ignitions can't be within this of the edge
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geom)

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

    np.random.seed(seed)
    upwind_directions = np.random.randint(
        low=upwind_direction_quadrant[0],
        high=upwind_direction_quadrant[1],
        size=len(ignition_locations),
    )

    for treatment in TREATMENTS:
        (training_data_dir / treatment).mkdir(parents=True, exist_ok=True)

        fuel_model_a = rio.open(fuels_files[treatment]["fbfm"]).read(1).astype("float32")
        fuel_model_a[(fuel_model_a == RAW_NO_DATA) | (fuel_model_a == 0.0)] = 91.0

        cc_a = rio.open(fuels_files[treatment]["cc"]).read(1).astype("float32")
        cc_a[cc_a > RAW_NO_DATA] /= 100  # convert to 0-1

        cbd_a = rio.open(fuels_files[treatment]["cbd"]).read(1).astype("float32")
        cbd_a[cbd_a > RAW_NO_DATA] /= 100  # convert to kg/m^3

        cbh_a = rio.open(fuels_files[treatment]["cbh"]).read(1).astype("float32")
        cbh_a[cbh_a > RAW_NO_DATA] /= 10  # convert to m

        ch_a = rio.open(fuels_files[treatment]["th"]).read(1).astype("float32")
        ch_a[ch_a > RAW_NO_DATA] /= 10  # convert to m

        slope_a = rio.open(topo_dir / "slope_degrees.tif").read(1).astype("float32")  # deg
        slope_nodata = slope_a == RAW_NO_DATA
        slope_a = np.tan(np.deg2rad(slope_a))  # convert from degrees to rise/run
        slope_a[slope_nodata] = 0.0

        aspect_a = rio.open(topo_dir / "aspect.tif").read(1).astype("float32")  # degrees
        aspect_a[aspect_a == RAW_NO_DATA] = 0.0

        # fuel_moisture units: kg moisture/kg ovendry weight
        shared_array_refs = {
            "slope": ray.put(slope_a),  # rise/run
            "aspect": ray.put(aspect_a),  # degrees clockwise from North
            "fuel_model": ray.put(fuel_model_a),  # index in fm.fuel_model_table
            "canopy_cover": ray.put(cc_a),  # 0-1
            "canopy_height": ray.put(ch_a),  # m
            "canopy_base_height": ray.put(cbh_a),  # m
            "canopy_bulk_density": ray.put(cbd_a),  # kg/m^3
            "wind_speed_10m": ray.put(WIND_SPEED_10M),
            "fuel_moisture_dead_1hr": ray.put(FUEL_MOISTURES["1hr"]),
            "fuel_moisture_dead_10hr": ray.put(FUEL_MOISTURES["10hr"]),
            "fuel_moisture_dead_100hr": ray.put(FUEL_MOISTURES["100hr"]),
            "fuel_moisture_live_herbaceous": ray.put(FUEL_MOISTURES["LH"]),
            "fuel_moisture_live_woody": ray.put(FUEL_MOISTURES["LW"]),
            "foliar_moisture": ray.put(FUEL_MOISTURES["FMC"]),
            "fuel_spread_adjustment": ray.put(pt_adjustments["fuel_spread"]),  # >= 0.0
            "weather_spread_adjustment": ray.put(pt_adjustments["weather_spread"]),  # >= 0.0
        }

        run_ignition_ray_ids = [
            run_ignition_ray.options(memory=RAY_MEM).remote(
                cube_shape,
                treatment,
                md,
                ignition_number,
                ignition_point,
                shared_array_refs,
                upwind_directions[ignition_number],
            )
            for ignition_number, ignition_point in enumerate(ignition_locations)
            for md in max_durations
        ]

        # ensure these finish before starting the next treatment's batch
        ray.wait(run_ignition_ray_ids, num_returns=len(run_ignition_ray_ids))
        results = ray.get(run_ignition_ray_ids)

        outputs_columns = [
            "ignition_number",
            "treatment",
            "max_duration",
            "acres_burned",
            "total_runtime",
            "stop_condition",
            "upwind_direction",
        ]
        csv_outputs = pd.DataFrame(results)[outputs_columns]
        csv_outputs.to_csv(training_data_dir / treatment / "outputs_table.csv", index=False)

        if collate_ignitions:
            cbp = compute_cbp(results, (template_raster.height, template_raster.width))
            cbp_profile = template_raster.profile | {"dtype": "float32", "count": 1}
            with rio.open(training_data_dir / treatment / "cbp.tif", "w", **cbp_profile) as dst:
                dst.write(cbp, 1)
            print(f"Wrote collated CBP raster for {treatment}")
        else:
            for result in results:
                write_raster(
                    training_data_dir,
                    result["treatment"],
                    result["ignition_number"],
                    template_raster,
                    result["ft_array"],
                )
