# Credit: https://github.com/aryarksub/PyroStack/

"""
run_pyretechnics.py

Purpose:
    For an individual fire interval:
        Retrieves the environmental fields for PyroStack and pre-processes them
        to become Pyretechnics inputs
        Runs a single Pyretechnics simulation
        Plots an example output figure

Usage:
    python src/run_pyretechnics.py
        --fire-id FIRE_ID
        --start-day START_DAY
        --duration DURATION
        --pyrostack-path /path/to/pyrostack_dir
        --output-path /path/to/output_dir

Arguments:
    --fire-id: Fire ID from FEDS database.
    --start-day: Day since the first FEDS observation for this fire (i.e. the first row of
        fire_times.csv with feds == True).
    --duration: Length of interval in days.
    --pyrostack-path: Path to the PyroStack directory (containing fires_{YYYY}/cubes/{fid}
        subdirectories and manifest.csv).
    --output-path: Path to the output directory to save figure.
"""

# pre-process / simulation requirements
import argparse
import os
import time

# plotting requirements
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.axes as maxes
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyretechnics.eulerian_level_set as els
import pyretechnics.load_landfire as lf
import rasterio
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyretechnics.space_time_cube import SpaceTimeCube
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import reproject
from shapely.geometry import shape as shapely_shape


def parse_args(args=None):
    def str2float(v):
        return float(v)

    parser = argparse.ArgumentParser(
        description="Example Pyretechnics Simulation Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--fire-id", dest="fid", required=True, help="Fire ID")
    parser.add_argument(
        "--pyrostack-path", required=True, help="Path to the PyroStack base directory."
    )
    parser.add_argument(
        "--start-day",
        type=str2float,
        required=False,
        default=None,
        help="Start day of the desired interval to plot.",
    )
    parser.add_argument(
        "--duration",
        type=str2float,
        required=False,
        default=None,
        help="Duration in days of the desired interval to plot.",
    )
    parser.add_argument(
        "--output-path", required=True, default=True, help="Where to save output figure"
    )

    args = parser.parse_args(args)
    return args


def get_fire_dir(args):
    year = args.fid[-8:-4]
    fire_dir = os.path.join(args.pyrostack_path, f"fires_{year}", "cubes", args.fid)

    if not os.path.isdir(fire_dir):
        raise RuntimeError("Directory does not exist: " + fire_dir)

    return fire_dir


def get_pyrostack_indices(args, fire_dir):
    times = pd.read_csv(os.path.join(fire_dir, "fire_times.csv"), parse_dates=["time"])
    feds_rows = times.index[times["feds"]].to_numpy()
    feds_times = times.loc[feds_rows, "time"]

    # 1) build a "day since first FEDS observation" axis, mirroring the old gpkg "duration" column
    t0 = feds_times.iloc[0]
    durations = (feds_times - t0).dt.total_seconds().to_numpy() / 86400.0

    def nearest_feds_position(day):
        diff = np.abs(durations - day)
        pos = np.argmin(diff)
        if diff[pos] > 1e-5:
            raise ValueError(f"Day {day} not found in fire_times.csv!")
        return int(pos)

    # 2) find the fire_spread band position (0-based) for the start and final observations
    start_pos = nearest_feds_position(args.start_day)
    final_pos = nearest_feds_position(args.start_day + args.duration)

    # 3) high_res_climate/low_res_climate carry one band per row of fire_times.csv (hourly),
    #   so the absolute row index doubles as the climate band index
    start_index = int(feds_rows[start_pos])
    timesteps = int(args.duration * 24)
    end_index = start_index + timesteps + 1

    return start_index, end_index, timesteps, start_pos, final_pos


def find_topo_file(dir_path, aliases):
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".tif") and any(alias in fname.lower() for alias in aliases):
            return os.path.join(dir_path, fname)
    raise RuntimeError(f"Could not find topo tif for aliases {aliases} in {dir_path}")


def get_landfire_tif_path(fire_dir, raster):
    # veg_fm_topo filenames carry inconsistent year/region prefixes and suffixes
    # (e.g. "fbfm40.tif" vs "lf2020_asp_ak.tif"), so match by substring rather
    # than an exact expected name.
    aliases = {
        "fbfm40.tif": ("f40", "m40"),
        "slp.tif": ("slp",),
        "asp.tif": ("asp",),
        "elev.tif": ("elev",),
    }[raster]
    return find_topo_file(os.path.join(fire_dir, "veg_fm_topo"), aliases)


def rename_keys(the_dict, keys_dict):
    for f, t in keys_dict.items():
        the_dict[t] = the_dict.pop(f)


def load_partial_raster(
    file_path,
    band_slice,
    dtype=None,
    cube_shape_divisors=(1, 1, 1),
    resampling_policy="nearest_match",
    resampling_method=Resampling.nearest,
):
    """
    Adapted from: https://github.com/pyregence/pyretechnics
    resampling_policy: "always_upsample" or "nearest_match"
    resampling_method: any rasterio.enums.Resampling method
    """
    with rasterio.open(file_path, "r") as input_raster:
        metadata = lf.raster_metadata(input_raster)

        if band_slice:
            start, stop = band_slice
            target_indexes = list(range(start + 1, stop + 1))
            bands = len(target_indexes)
        else:
            target_indexes = None  # None implies "read all bands"
            bands = metadata["bands"]

        rows = metadata["rows"]
        cols = metadata["cols"]
        (b, r, c) = cube_shape_divisors
        new_bands = lf.maybe_resample_resolution(bands, b, resampling_policy)
        new_rows = lf.maybe_resample_resolution(rows, r, resampling_policy)
        new_cols = lf.maybe_resample_resolution(cols, c, resampling_policy)
        if new_bands == bands and new_rows == rows and new_cols == cols:
            return {
                "array": input_raster.read(indexes=target_indexes, out_dtype=dtype),
                "metadata": metadata,
            }
        else:
            metadata["bands"] = new_bands
            metadata["rows"] = new_rows
            metadata["cols"] = new_cols
            metadata["transform"] = input_raster.transform * input_raster.transform.scale(
                cols / new_cols,
                rows / new_rows,
            )
            array = input_raster.read(
                indexes=target_indexes,
                out_dtype=dtype,
                out_shape=(new_bands, new_rows, new_cols),
                resampling=resampling_method,
            )
            return {
                "array": array,
                "metadata": metadata,
            }


def resample_to_grid(
    array_2d,
    src_transform,
    src_crs,
    dst_shape,
    dst_transform,
    dst_crs,
    resampling=Resampling.nearest,
):
    destination = np.zeros(dst_shape, dtype=array_2d.dtype)
    reproject(
        source=array_2d,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return destination


def raster_mask_to_geoseries(mask, transform, crs):
    mask_shapes = rasterio_shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform)
    geoms = [shapely_shape(geom) for geom, value in mask_shapes if value == 1]
    return gpd.GeoSeries(geoms, crs=crs)


def feds_ignition_field(fire_dir, start_pos):
    with rasterio.open(os.path.join(fire_dir, "fuel_structure", "cc.tif")) as fuel_src:
        dst_shape = fuel_src.shape
        dst_transform = fuel_src.transform
        dst_crs = fuel_src.crs

    with rasterio.open(os.path.join(fire_dir, "fire_spread", "farea.tif")) as farea_src:
        cold_burn_lowres = farea_src.read(start_pos + 1)
        src_transform = farea_src.transform
        src_crs = farea_src.crs
    with rasterio.open(os.path.join(fire_dir, "fire_spread", "fline.tif")) as fline_src:
        active_burn_lowres = fline_src.read(start_pos + 1)

    if np.all(np.isnan(active_burn_lowres)):
        # no distinct active-fireline layer at this timestep (e.g. the very first
        # FEDS observation) -- treat the whole observed area as freshly ignited
        active_burn_lowres = cold_burn_lowres
    cold_burn_lowres = np.nan_to_num(cold_burn_lowres, nan=0.0)
    active_burn_lowres = np.nan_to_num(active_burn_lowres, nan=0.0)

    cold_burn_raster = resample_to_grid(
        cold_burn_lowres, src_transform, src_crs, dst_shape, dst_transform, dst_crs
    )
    active_burn_raster = resample_to_grid(
        active_burn_lowres, src_transform, src_crs, dst_shape, dst_transform, dst_crs
    )

    pre_burned = np.where((cold_burn_raster == 1) & (active_burn_raster == 0), 1, 0)
    ignition = active_burn_raster.astype("uint8")

    times = pd.read_csv(os.path.join(fire_dir, "fire_times.csv"), parse_dates=["time"])
    feds_rows = times.index[times["feds"]].to_numpy()
    ignition_hour = times.loc[feds_rows[start_pos], "time"].hour

    return pre_burned, ignition, ignition_hour


def load_and_transform_rasters(args, fire_dir, band_slice, start_pos):
    # load rasters
    pyregence_rasters = {
        raster: lf.load_raster(os.path.join(fire_dir, "fuel_structure", raster), dtype="float32")
        for raster in [
            "cc.tif",
            "ch.tif",
            "cbh.tif",
            "cbd.tif",
        ]
    }
    landfire_rasters = {
        raster: lf.load_raster(get_landfire_tif_path(fire_dir, raster), dtype="float32")
        for raster in [
            "fbfm40.tif",
            "slp.tif",
            "asp.tif",
        ]
    }
    non_landfire_rasters = {
        raster: load_partial_raster(
            os.path.join(fire_dir, "high_res_climate", raster), band_slice, dtype="float32"
        )
        for raster in [
            "ws.tif",
            "wd.tif",
            "m1.tif",
            "m10.tif",
            "m100.tif",
            "lh.tif",
            "lw.tif",
        ]
    }
    t2m_path = os.path.join(fire_dir, "low_res_climate", "t2m.tif")
    temp_raster = {"t2m.tif": load_partial_raster(t2m_path, band_slice, dtype="float32")}
    rasters_dict = pyregence_rasters | landfire_rasters | non_landfire_rasters | temp_raster
    rename_keys(
        rasters_dict,
        {
            "slp.tif": "slope",  # magnitude
            "asp.tif": "aspect",  # direction
            "fbfm40.tif": "fuel_model",  # surface vegetation
            "cc.tif": "canopy_cover",  # how close trees are 0% (no trees) to 100% (all trees)
            "ch.tif": "canopy_height",  # how tall the trees are
            "cbh.tif": "canopy_base_height",  # how high to the lowest branches
            "cbd.tif": "canopy_bulk_density",  # amount of material in tree branches
            "ws.tif": "wind_speed_10m",  # magnitude
            "wd.tif": "upwind_direction",  # direction
            "m1.tif": "fuel_moisture_dead_1hr",  # thinner
            "m10.tif": "fuel_moisture_dead_10hr",  # 0.5 cm to 2cm
            "m100.tif": "fuel_moisture_dead_100hr",  # 2cm to 6.5cm
            "lh.tif": "fuel_moisture_live_herbaceous",  # leaves, grasses etc.
            "lw.tif": "fuel_moisture_live_woody",  # branches, trunks etc.
            "t2m.tif": "temperature",  # 2 meter temperature
        },
    )

    # SLOPE
    slope_array = rasters_dict["slope"]["array"]
    np.tan(np.radians(slope_array, out=slope_array), out=slope_array)  # convert degrees to ratio

    # ASPECT
    aspect_array = rasters_dict["aspect"]["array"]
    np.subtract(180, aspect_array, out=aspect_array)
    np.mod(aspect_array, 360, out=aspect_array)
    aspect_array[np.isnan(aspect_array)] = -1

    # FUEL_MODEL
    fuel_model_array = rasters_dict["fuel_model"]["array"]
    fuel_model_array[np.isnan(fuel_model_array)] = -9999

    # CANOPY_COVER
    canopy_cover_array = rasters_dict["canopy_cover"]["array"]
    np.multiply(canopy_cover_array, 0.01, out=canopy_cover_array)
    canopy_cover_array[np.isnan(canopy_cover_array)] = -9999

    # CANOPY_HEIGHT
    canopy_height_array = rasters_dict["canopy_height"]["array"]
    np.multiply(canopy_height_array, 0.1, out=canopy_height_array)
    canopy_height_array[np.isnan(canopy_height_array)] = -9999

    # CANOPY_BASE_HEIGHT
    canopy_base_height_array = rasters_dict["canopy_base_height"]["array"]
    np.multiply(canopy_base_height_array, 0.1, out=canopy_base_height_array)
    canopy_base_height_array[np.isnan(canopy_base_height_array)] = -9999

    # CANOPY_BULK_DENSITY
    canopy_bulk_density_array = rasters_dict["canopy_bulk_density"]["array"]
    np.multiply(canopy_bulk_density_array, 0.01, out=canopy_bulk_density_array)
    canopy_bulk_density_array[np.isnan(canopy_bulk_density_array)] = -9999

    # WIND_SPEED
    wind_speed_10m_array = rasters_dict["wind_speed_10m"]["array"]  # miles/hour at 20 ft
    # convert to km/h at 10 meters
    np.multiply(wind_speed_10m_array, 1.609344 * 1.15, out=wind_speed_10m_array)

    # UPWIND_DIRECTION
    upwind_direction_array = rasters_dict["upwind_direction"]["array"]
    np.subtract(180, upwind_direction_array, out=upwind_direction_array)
    np.mod(upwind_direction_array, 360, out=upwind_direction_array)

    # FUEL_MOISTURE_DEAD_1HR
    fuel_moisture_dead_1hr_array = rasters_dict["fuel_moisture_dead_1hr"]["array"]
    np.multiply(fuel_moisture_dead_1hr_array, 0.01, out=fuel_moisture_dead_1hr_array)
    fuel_moisture_dead_1hr_array[np.isnan(fuel_moisture_dead_1hr_array)] = -9999

    # FUEL_MOISTURE_DEAD_10HR
    fuel_moisture_dead_10hr_array = rasters_dict["fuel_moisture_dead_10hr"]["array"]
    np.multiply(fuel_moisture_dead_10hr_array, 0.01, out=fuel_moisture_dead_10hr_array)
    fuel_moisture_dead_10hr_array[np.isnan(fuel_moisture_dead_10hr_array)] = -9999

    # FUEL_MOISTURE_DEAD_100HR
    fuel_moisture_dead_100hr_array = rasters_dict["fuel_moisture_dead_100hr"]["array"]
    np.multiply(fuel_moisture_dead_100hr_array, 0.01, out=fuel_moisture_dead_100hr_array)
    fuel_moisture_dead_100hr_array[np.isnan(fuel_moisture_dead_100hr_array)] = -9999

    # FUEL_MOISTURE_LIVE_HERBACEOUS
    fuel_moisture_live_herbaceous_array = rasters_dict["fuel_moisture_live_herbaceous"]["array"]
    np.multiply(fuel_moisture_live_herbaceous_array, 0.01, out=fuel_moisture_live_herbaceous_array)

    # FUEL_MOISTURE_LIVE_WOODY
    fuel_moisture_live_woody_array = rasters_dict["fuel_moisture_live_woody"]["array"]
    np.multiply(fuel_moisture_live_woody_array, 0.01, out=fuel_moisture_live_woody_array)

    # TEMPERATURE
    temperature_array = rasters_dict["temperature"]["array"]
    np.subtract(temperature_array, 273.15, out=temperature_array)

    # GENERATE IGNITION FIELD
    pre_burned, ignition, ignition_hour = feds_ignition_field(fire_dir, start_pos)
    # set pre-burned fuel to unburnable
    fuel_model_array[:, pre_burned == 1] = 99.0

    return rasters_dict, pre_burned, ignition


def make_simple_plot(args, fire_dir, pre_burn, ignition, toa_matrix, start_pos, final_pos):
    # retrieve fire name
    firelist = pd.read_csv(os.path.join(args.pyrostack_path, "manifest.csv"))
    firename = firelist.loc[firelist[firelist["Event_ID"] == args.fid].index[0]]["Incid_Name"]
    sup_title = f"{firename} (ID: {args.fid})"

    # retrieve tif data
    dem_path = get_landfire_tif_path(fire_dir, "elev.tif")
    slp_path = get_landfire_tif_path(fire_dir, "slp.tif")
    asp_path = get_landfire_tif_path(fire_dir, "asp.tif")
    with (
        rasterio.open(dem_path) as dem_src,
        rasterio.open(slp_path) as slp_src,
        rasterio.open(asp_path) as asp_src,
    ):
        slp = slp_src.read(1)
        asp = asp_src.read(1)
        asp[np.isnan(asp)] = 0

        # compute hillshade
        slp_rad = np.radians(slp)
        asp_rad = np.radians(asp)

        azimuth_sun = 315
        altitude_sun = 45

        zenith_rad = np.radians(90 - altitude_sun)
        azimuth_rad = np.radians(azimuth_sun)

        hillshade = (np.cos(zenith_rad) * np.cos(slp_rad)) + (
            np.sin(zenith_rad) * np.sin(slp_rad) * np.cos(azimuth_rad - asp_rad)
        )
        hillshade = np.maximum(hillshade, 0)

        # retrieve metadata
        map_proj = ccrs.epsg(dem_src.crs.to_epsg())
        transform = dem_src.transform
        b = dem_src.bounds
        extent = [b.left, b.right, b.bottom, b.top]

    # retrieve perimeter/fireline geometries directly from the fire_spread rasters
    # (300m grid; already in the same EPSG:5070 CRS as the 30m fuel_structure/veg_fm_topo grid)
    with rasterio.open(os.path.join(fire_dir, "fire_spread", "farea.tif")) as farea_src:
        farea_init = np.nan_to_num(farea_src.read(start_pos + 1), nan=0.0)
        farea_final = np.nan_to_num(farea_src.read(final_pos + 1), nan=0.0)
        fs_transform = farea_src.transform
        fs_crs = farea_src.crs
    with rasterio.open(os.path.join(fire_dir, "fire_spread", "fline.tif")) as fline_src:
        fline_init = np.nan_to_num(fline_src.read(start_pos + 1), nan=0.0)

    perimeter_init_geoms = raster_mask_to_geoseries(farea_init, fs_transform, fs_crs)
    perimeter_final_geoms = raster_mask_to_geoseries(farea_final, fs_transform, fs_crs)
    perimeters_reprojected = gpd.GeoDataFrame(
        geometry=pd.concat([perimeter_init_geoms, perimeter_final_geoms], ignore_index=True),
        crs=fs_crs,
    )
    fline_reprojected = gpd.GeoDataFrame(
        geometry=raster_mask_to_geoseries(fline_init, fs_transform, fs_crs),
        crs=fs_crs,
    )

    # crop to the region where the burn occurs
    rows, cols = np.where((toa_matrix > 0) | (pre_burn))
    buffer = 10
    perim_bounds = perimeters_reprojected.total_bounds
    perim_xmin, perim_ymin, perim_xmax, perim_ymax = perim_bounds
    perim_buffer = 300

    r0, r1 = max(0, rows.min() - buffer), min(toa_matrix.shape[0], rows.max() + buffer)
    c0, c1 = max(0, cols.min() - buffer), min(toa_matrix.shape[1], cols.max() + buffer)

    x_at_c0, y_at_r0 = transform * (c0, r0)
    x_at_c1, y_at_r1 = transform * (c1, r1)

    left = min(x_at_c0, x_at_c1, perim_xmin - perim_buffer)
    right = max(x_at_c0, x_at_c1, perim_xmax + perim_buffer)
    bottom = min(y_at_r0, y_at_r1, perim_ymin - perim_buffer)
    top = max(y_at_r0, y_at_r1, perim_ymax + perim_buffer)

    # set up figure
    fig_width = 7
    fig_height = (fig_width * (top - bottom) / (right - left)) + 1

    current_width = right - left
    zoom_scale = min((toa_matrix.shape[1] * transform[0]) / current_width, 4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw={"projection": map_proj})

    # plot hillshade layer
    data_crs = ccrs.epsg(5070)
    ax.imshow(
        hillshade,
        cmap="gray",
        extent=extent,
        origin="upper",
        transform=data_crs,
        interpolation="none",
        alpha=1.0,
    )

    # plot time of arrival field
    ax.imshow(
        toa_matrix,
        cmap="Oranges",
        extent=extent,
        origin="upper",
        transform=data_crs,
        interpolation="none",
        alpha=0.5,
    )

    # plot final observed perimeter
    perimeters_reprojected.iloc[[1]].plot(
        ax=ax,
        facecolor="none",
        edgecolor="red",
        transform=data_crs,
        linewidth=1 * zoom_scale,
        zorder=3,
    )

    # plot initial cold perimeter
    perimeters_reprojected.iloc[[0]].plot(
        ax=ax,
        facecolor="none",
        edgecolor="blue",
        transform=data_crs,
        linewidth=0.5 * zoom_scale,
        zorder=4,
    )

    # plot initial active fireline
    fline_reprojected.plot(
        ax=ax,
        facecolor="none",
        edgecolor="lime",
        transform=data_crs,
        linewidth=0.7 * zoom_scale,
        zorder=5,
    )

    # crop
    ax.set_extent([left, right, bottom, top], crs=data_crs)

    # make colorbar and bottom legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=maxes.Axes)
    cbar_mappable = cm.ScalarMappable(cmap=plt.colormaps["Oranges"])
    cbar_mappable.set_array(toa_matrix)
    fig.colorbar(cbar_mappable, cax=cax, label="Time of Arrival (hours)")

    lax = divider.append_axes("bottom", size="5%", pad=0.6, axes_class=maxes.Axes)
    lax.axis("off")
    proxy_handles = [
        Line2D([0], [0], color="blue", lw=1, label="Initial Cold Perimeter"),
        Line2D([0], [0], color="lime", lw=1, label="Initial Fireline"),
        Line2D([0], [0], color="red", lw=1, label="Final Perimeter"),
    ]
    labels = [h.get_label() for h in proxy_handles]
    lax.legend(
        proxy_handles,
        labels,
        loc="center",
        ncol=4,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.5),
    )

    # add titles
    fig.suptitle("Time of Arrival Field of " + sup_title, fontsize=10, fontweight="bold")

    # add lat/lon grid
    gl = ax.gridlines(
        draw_labels=True, crs=ccrs.PlateCarree(), linestyle="--", color="black", alpha=0.3
    )
    gl.top_labels = False
    gl.right_labels = False

    fname = args.fid + "_interval_" + str(args.start_day).replace(".", "_") + "_test.png"
    output_file = os.path.join(args.output_path, fname)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def spread_one_fire(args):
    # locate the fire's PyroStack cube and its observation intervals
    fire_dir = get_fire_dir(args)
    indices = get_pyrostack_indices(args, fire_dir)
    start_index, end_index, timesteps, start_pos, final_pos = indices

    # preprocess input data
    band_slice = (start_index, end_index)
    rasters_dict, pre_burned, ignition = load_and_transform_rasters(
        args, fire_dir, band_slice, start_pos
    )

    # set foliar moisture (no observation available)
    rasters_dict["foliar_moisture"] = {"array": 1.0}

    # set spread adjustment factor (e.g., prescribed burn window during daytime)
    # NOTE: here we prescribe no adjustment
    rasters_dict["weather_spread_adjustment"] = {"array": np.ones(end_index - start_index)}

    # set up dimensions
    sample_shape = rasters_dict["slope"]["array"].shape
    cube_shape = (timesteps + 1, sample_shape[1], sample_shape[2])

    cube_resolution = (
        60,  # band_duration: minutes
        30,  # cell_height:   meters
        30,  # cell_width:    meters
    )

    # specify cube refresh rates
    cube_refresh_rates = {
        "wind_speed_10m": 1.0 / 15.0,
        "upwind_direction": 1.0 / 15.0,
        "fuel_moisture_dead_1hr": 1.0 / 30.0,
        "temperature": 1.0 / 30.0,
        "fuel_spread_adjustment": 0.0,
        "weather_spread_adjustment": 1.0 / 30.0,
    }

    spot_config = {
        "random_seed": 1234567890,
        "firebrands_per_unit_heat": 1e-9,  # firebrands/kJ
        "downwind_distance_mean": 10.0,  # meters
        # downwind_distance_mean multiplier [I^fireline_intensity_exponent]
        "fireline_intensity_exponent": 0.3,
        # downwind_distance_mean multiplier [U^wind_speed_exponent]
        "wind_speed_exponent": 0.55,
        # meters^2 / meter [downwind_variance_mean_ratio = Var(X) / E(X)]
        "downwind_variance_mean_ratio": 425.0,
        "crosswind_distance_stdev": 100.0,  # meters
        "decay_distance": 200.0,  # meters
    }

    # create SpaceTimeCube objects from input fields
    space_time_cubes = {
        name: SpaceTimeCube(cube_shape, array["array"]) for (name, array) in rasters_dict.items()
    }

    # create a SpreadState object and specify ignition from the FEDS fireline
    phi = np.where(ignition == 1, -1, 1).astype(np.float32)
    spread_state = els.SpreadState(cube_shape).ignite_cells(
        lower_left_corner=(0, 0), ignition_matrix=phi
    )

    # run simulation
    runtime_start = time.perf_counter()
    # account for 30 min offset between VIIRS overpass and PyroStack layers
    fire_spread_results = els.spread_fire_with_phi_field(
        space_time_cubes,
        spread_state,
        cube_resolution,
        30,
        timesteps * 60,
        spot_config=spot_config,
        surface_lw_ratio_model="rothermel",
        cube_refresh_rates=cube_refresh_rates,
    )
    runtime_stop = time.perf_counter()
    print("Simulation finished. Runtime = " + str(runtime_stop - runtime_start) + " seconds")

    # extract output matrices
    output_layers = [
        "fire_type",
        "spread_rate",
        "fireline_intensity",
        "flame_length",
        "time_of_arrival",
        "phi",
    ]
    spread_state = fire_spread_results["spread_state"]  # mutated SpreadState object
    output_matrices = spread_state.get_full_matrices(layers=output_layers)
    # _fire_type_matrix: 0 no fire 1 surface fire 2 passive crown fire 3 active crown fire
    _fire_type_matrix = output_matrices["fire_type"]
    _spread_rate_matrix = output_matrices["spread_rate"]  # in m/s
    _fireline_intensity_matrix = output_matrices["fireline_intensity"]  # kW/m
    _flame_length_matrix = output_matrices["flame_length"]  # how tall in m
    # minutes (subtract out the offset), then convert to hours
    time_of_arrival_matrix = output_matrices["time_of_arrival"] - 30
    np.divide(time_of_arrival_matrix, 60.0, out=time_of_arrival_matrix)

    # generate simple example plot
    make_simple_plot(
        args, fire_dir, pre_burned, ignition, time_of_arrival_matrix, start_pos, final_pos
    )


def main():
    args = parse_args()

    spread_one_fire(args)


if __name__ == "__main__":
    main()
