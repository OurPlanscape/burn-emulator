# [[file:../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Definitions and Setup][Definitions and Setup:1]]
import os
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
from rasterio.windows import from_bounds

NUM_IGNITIONS = 5000
BASE = "/mnt/share/rem"
PROJ = "PC612_emulator_WC711_v1"

# TRAINING_DATA_DIR = f"{BASE}/{PROJ}/training_data_v9"
# TRAINING_DATA_DIR = f"{BASE}/{PROJ}/training_data_05Aug2026" # output dir; new fuels data from Dawn, fixed wind dir, old PT version (2026.3.25)
# TRAINING_DATA_DIR = f"{BASE}/{PROJ}/training_data_07Aug2026" # output dir; new fuels data, variable wind dir, old PT version (2026.3.25)
# TRAINING_DATA_DIR = f"{BASE}/{PROJ}/training_data_10Aug2026" # output dir; new fuels data, variable wind dir, new PT version (2026.8.10)
TRAINING_DATA_DIR = f"{BASE}/{PROJ}/training_data_14Aug2026"  # output dir; new fuels data, variable wind dir, new PT version (2026.8.10), upwind_direction matched between baseline & legalmax

# R6_DATA_DIR = "R6_2026_fuels" # raw data from Dawn
R6_DATA_DIR = "R6_2026_fuels_05Aug2026"  # raw data from Dawn

# BLENDED_DATA_DIR_VERSION = "20260624" # incorrect rxb extent
BLENDED_DATA_DIR_VERSION = "05Aug2026"  # correct rxb extent

WRITE_FT_RASTERS = True  # preserve the fire type arrays as geotif files

RAY_MEM = (
    4 * 1024 * 1024 * 1024
)  # limit the number of ignitions that can run in parallel via memory constraints

aoi_shape = "/mnt/share/rem-inputs/PC612_emulator_WC711_5070.shp"
project_aoi_f = f"{BASE}/{PROJ}/aoi/{PROJ}_aoi_5070.tif"  # large extent
treated_extent_f = f"{BASE}/{PROJ}/fvs/treatments/{PROJ}_treatment_5070.tif"  # varloc extent within large extent
ff_fbfm_f = f"{BASE}/CONUS/FF/fuels_fm40_2021_12.tif"
fbfm_files = {}

template_raster_file = f"{BASE}/{PROJ}/legalmax_baseline_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_fbfm.tif"

buffer_dist = -2000  # m; ignitions can't be within this distance of the edge

aspect_file = f"{BASE}/CONUS/LF/aspect.tif"
slope_file = f"{BASE}/CONUS/LF/slope_degrees.tif"

max_durations = [8 * 60]

wind_speed_10m = 10.0  # km/hr (10 km/hr ~= 6.2 mph)

# define EITHER a constant wind direction (upwind_direction; must be float) or sample uniformly within a range (upwind_direction_quadrant; must be list)
# upwind_direction = 180.0 # degrees clockwise from North # fixed wind dir for all runs up until 07Aug2026
upwind_direction_quadrant = [
    225,
    270,
]  # min, max of compass rose quadrant in degrees clockwise from North

fuel_moistures = {
    "1hr": 0.05,
    "10hr": 0.10,
    "100hr": 0.15,
    "LH": 0.90,
    "LW": 0.60,
    "FMC": 0.90,
}

pt_adjustments = {"fuel_spread": 1.0, "weather_spread": 1.0}

seed = 123456789

# set up stuff
treatments = ["baseline", "legalmax"]

np.random.seed(seed)

os.makedirs(f"{BASE}/{PROJ}/topo", exist_ok=True)
for treatment in treatments:
    os.makedirs(f"{TRAINING_DATA_DIR}/{treatment}", exist_ok=True)
    fbfm_files[treatment] = (
        f"{BASE}/{PROJ}/{R6_DATA_DIR}/{treatment}/PC612_R6_{treatment}_2026_fbfm.tif"
    )


def attach_shared_array(shared_array_ref):
    shared_array = ray.get(shared_array_ref)
    return shared_array


def is_burnable(fuel_model_cube, y, x):
    fuel_model_number = fuel_model_cube.get(0, y, x)
    return fm.fuel_model_exists(fuel_model_number) and not (
        91 <= fuel_model_number <= 99
    )


def outside_buffer(fuel_model_cube, point: tuple, buffered_gdf, transform):
    #  single point = (row, col)

    x, y = rio.transform.xy(transform=transform, rows=point[0], cols=point[1], crs=5070)
    points_geom = gpd.points_from_xy(x=[x], y=[y])

    return points_geom.within(buffered_gdf["geometry"][0])[0]  # return first element of list


def sample_ignited_cells_buffered(
    fuel_model_cube, num_ignitions, buffered_gdf, transform
):
    # Extract the SpaceTimeCube shape

    _bands, rows, cols = fuel_model_cube.shape

    # Set the random seed
    random.seed(seed)

    # Prepare the accumulator
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


def preserve_ignition_locations(ignitions: list, transform, out_file: str):
    df = pd.DataFrame(ignitions, columns=["row", "col"])
    df["x"], df["y"] = rio.transform.xy(transform, df["row"].values, df["col"].values)
    df["ignition_number"] = df.index
    df.to_csv(out_file, index=False)
    print("Preserved ignition locations")


def write_raster(treatment, ignition_number, md, template_raster, ft_array):
    with rio.open(
        f"{TRAINING_DATA_DIR}/{treatment}/{ignition_number}/{md}/fire_type.tif",
        "w",
        **template_raster.profile,
    ) as dst:
        dst.write(ft_array, 1)


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
    os.makedirs(
        f"{TRAINING_DATA_DIR}/{treatment}/{ignition_number}/{md}", exist_ok=True
    )
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

    # Embed the start time in the SpreadState object
    spread_state.set_start_time(start_time)

    runtime_start = perf_counter()
    fire_spread_results = els.spread_fire_with_phi_field(
        space_time_cubes,
        spread_state,
        cube_resolution,
        start_time,
        md,
        surface_lw_ratio_model="rothermel",
    )
    runtime_stop = perf_counter()
    stop_time = fire_spread_results["stop_time"]  # minutes
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
    # if WRITE_FT_ARRAYS:
    #    write_raster(treatment, ignition_number, md, template_raster, output_matrics)
# Definitions and Setup:1 ends here

# [[file:../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Prepare input fuels rasters][Prepare input fuels rasters:1]]
aoi_r = rio.open(project_aoi_f)
tx_area_r = rio.open(treated_extent_f)
tx_area_a = tx_area_r.read(1)

raster_names = ["cbd", "cbh", "cc", "fbfm", "th"]

# need to clip all R6 rasters to AOI extents
# tx_area_r and/or ff_fbfm has dtype=uint16 and -999 overflows as 25
nodata = -999
out_profile = tx_area_r.profile
out_profile.update({"dtype": "int16"})

with rio.open(ff_fbfm_f) as ff_fbfm:
    ff_fbfm_aoi_a = ff_fbfm.read(1, window=from_bounds(left=aoi_r.bounds.left, bottom=aoi_r.bounds.bottom, right=aoi_r.bounds.right, top=aoi_r.bounds.top, transform=ff_fbfm.transform)).astype("int16") # read FF data within the largest extent

    for treatment in treatments:
        os.makedirs(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}", exist_ok=True)

        for r in raster_names:
            with rio.open(f"{BASE}/{PROJ}/{R6_DATA_DIR}/{treatment}/PC612_R6_{treatment}_2026_{r}.tif", "r+") as src: # Dawn's rasters have nan for nodata; fixing with this: https://rasterio.groups.io/g/main/topic/change_the_nodata_value_in_a/28801885
                src.nodata = nodata # this changes the original data
                a = src.read(1, window=from_bounds(left=aoi_r.bounds.left, bottom=aoi_r.bounds.bottom, right=aoi_r.bounds.right, top=aoi_r.bounds.top, transform=src.transform))
                a[a == np.nan] = nodata
                if r == "fbfm":
                    a[a == nodata] = ff_fbfm_aoi_a[a == nodata] # combine FVS data with FF data within the largest extent

                a[tx_area_a == 0] = nodata # keep nodata outside of the small extent

                with rio.open(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_{r}.tif", "w", **out_profile) as dst:
                    dst.nodata = nodata
                    dst.write(a, 1)

template_raster = rio.open(f"{BASE}/{PROJ}/legalmax_baseline_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_fbfm.tif")
template_array = template_raster.read(1).astype("float32")

# LANDFIRE data is already in 5070
aspect_r = rio.open(aspect_file)
aspect_a = aspect_r.read(1, window=from_bounds(left=aoi_r.bounds.left, bottom=aoi_r.bounds.bottom, right=aoi_r.bounds.right, top=aoi_r.bounds.top, transform=aspect_r.transform)) # read FF data within the largest extent
with rio.open(f"{BASE}/{PROJ}/topo/aspect.tif", "w", **tx_area_r.profile) as dst:
    dst.write(aspect_a, 1)

slope_r = rio.open(slope_file)
slope_a = slope_r.read(1, window=from_bounds(left=aoi_r.bounds.left, bottom=aoi_r.bounds.bottom, right=aoi_r.bounds.right, top=aoi_r.bounds.top, transform=slope_r.transform)) # read FF data within the largest extent
with rio.open(f"{BASE}/{PROJ}/topo/slope_degrees.tif", "w", **tx_area_r.profile) as dst:
    dst.write(slope_a, 1)
# Prepare input fuels rasters:1 ends here

# [[file:../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Determine ignition locations][Determine ignition locations:1]]
cube_shape = (
    96,   # bands: 3 days + 3 hours @ 1 hour/band
    template_raster.height, # rows:  ? @ 30 meters/row
    template_raster.width, # cols:  ? @ 30 meters/col
)

aoi_gdf = gpd.read_file(aoi_shape)
buffered_geom = aoi_gdf.buffer(buffer_dist) # prevent ignitions within ~buffer_dist~ of the edge
buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geom)
ignition_locations = sample_ignited_cells_buffered(fuel_model_cube=SpaceTimeCube(cube_shape, template_array),
                                                   num_ignitions=NUM_IGNITIONS,
                                                   buffered_gdf=buffered_gdf,
                                                   transform=template_raster.transform)
preserve_ignition_locations(ignitions=ignition_locations,
                            transform=template_raster.transform,
                            out_file=f"{TRAINING_DATA_DIR}/ignition_locations.csv")
# Determine ignition locations:1 ends here

# [[file:../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Run Pyretechnics in parallel with Ray][Run Pyretechnics in parallel with Ray:1]]
print("Starting Pyretechnics")

ray.init()

cube_resolution = (
    60, # band_duration: minutes
    30, # cell_height:   meters
    30, # cell_width:    meters
)

# this assumes ~upwind_direction_quadrant~ has been defined!
upwind_directions = np.random.randint(low=upwind_direction_quadrant[0], high=upwind_direction_quadrant[1], size=len(ignition_locations)) # save random upwind_direction within range

print("Reading fuels arrays")
for treatment in treatments:
    shared_array_refs = {}

    fuel_model_a = rio.open(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_fbfm.tif").read(1).astype("float32")
    fuel_model_a[(fuel_model_a == nodata) | (fuel_model_a == 0.0)] = 91.0

    cc_a = rio.open(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_cc.tif").read(1).astype("float32")
    cc_a[cc_a > nodata] /= 100 # convert to 0-1

    cbd_a = rio.open(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_cbd.tif").read(1).astype("float32")
    cbd_a[cbd_a > nodata] /= 100 # convert to kg/m^3

    cbh_a = rio.open(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_cbh.tif").read(1).astype("float32")
    cbh_a[cbh_a > nodata] /= 10 # convert to m

    ch_a = rio.open(f"{BASE}/{PROJ}/legalmax_{treatment}_FF_{BLENDED_DATA_DIR_VERSION}/{PROJ}_Project_2026_th.tif").read(1).astype("float32")
    ch_a[ch_a > nodata] /= 10 # convert to m

    slope_a = rio.open(f"{BASE}/{PROJ}/topo/slope_degrees.tif").read(1).astype("float32") # 0-67 deg; nodata = 0
    slope_a = np.tan(slope_a) # convert from degrees (0-359) to rise/run (-226-8.33)

    aspect_a = rio.open(f"{BASE}/{PROJ}/topo/aspect.tif").read(1).astype("float32") # degrees; nodata = 0

    shared_array_refs = {
        "slope"                        : ray.put(slope_a),   # rise/run
        "aspect"                       : ray.put(aspect_a), # degrees clockwise from North
        "fuel_model"                   : ray.put(fuel_model_a),   # integer index in fm.fuel_model_table
        "canopy_cover"                 : ray.put(cc_a),   # 0-1
        "canopy_height"                : ray.put(ch_a),   # m
        "canopy_base_height"           : ray.put(cbh_a),   # m
        "canopy_bulk_density"          : ray.put(cbd_a),   # kg/m^3
        "wind_speed_10m"               : ray.put(wind_speed_10m),
        "fuel_moisture_dead_1hr"       : ray.put(fuel_moistures["1hr"]),  # kg moisture/kg ovendry weight
        "fuel_moisture_dead_10hr"      : ray.put(fuel_moistures["10hr"]),  # kg moisture/kg ovendry weight
        "fuel_moisture_dead_100hr"     : ray.put(fuel_moistures["100hr"]),  # kg moisture/kg ovendry weight
        "fuel_moisture_live_herbaceous": ray.put(fuel_moistures["LH"]),  # kg moisture/kg ovendry weight
        "fuel_moisture_live_woody"     : ray.put(fuel_moistures["LW"]),  # kg moisture/kg ovendry weight
        "foliar_moisture"              : ray.put(fuel_moistures["FMC"]),  # kg moisture/kg ovendry weight
        "fuel_spread_adjustment"       : ray.put(pt_adjustments["fuel_spread"]),   # float >= 0.0 (Optional: defaults to 1.0)
        "weather_spread_adjustment"    : ray.put(pt_adjustments["weather_spread"]),   # float >= 0.0 (Optional: defaults to 1.0)
    }

    # handle both constant and variable wind direction
    if "upwind_direction" in globals():
        if type(upwind_direction) is float:
            run_ignition_ray_ids = [run_ignition_ray.options(memory=RAY_MEM).remote(cube_shape,
                                                                            treatment,
                                                                            md,
                                                                            ignition_number,
                                                                            ignition_point,
                                                                            shared_array_refs,
                                                                            upwind_direction)
                             for ignition_number, ignition_point in enumerate(ignition_locations)
                             for md in max_durations]
    elif "upwind_direction_quadrant" in globals():
        if type(upwind_direction_quadrant) is list:
            run_ignition_ray_ids = [run_ignition_ray.options(memory=RAY_MEM).remote(cube_shape,
                                                                                    treatment,
                                                                                    md,
                                                                                    ignition_number,
                                                                                    ignition_point,
                                                                                    shared_array_refs,
                                                                                    upwind_directions[ignition_number])
                                    for ignition_number, ignition_point in enumerate(ignition_locations)
                                    for md in max_durations]

    ray.wait(run_ignition_ray_ids, num_returns=len(run_ignition_ray_ids)) # ensure that these finish before starting the next batch
    results = ray.get(run_ignition_ray_ids)

    csv_outputs = pd.DataFrame(results)[["ignition_number", "treatment", "max_duration", "acres_burned", "total_runtime", "stop_condition", "upwind_direction"]]
    csv_outputs.to_csv(f"{TRAINING_DATA_DIR}/{treatment}/outputs_table.csv", index=False)

    # save the fire type arrays
    for result in results:
        write_raster(result["treatment"], result["ignition_number"], result["max_duration"], template_raster, result["ft_array"])
# Run Pyretechnics in parallel with Ray:1 ends here
