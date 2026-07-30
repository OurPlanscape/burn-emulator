# [[file:../../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Visualize SD data][Visualize SD data:1]]
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import pandas as pd
import geopandas as gpd
import numpy as np

BASE = "/mnt/share/rem"
PROJ = "PC612_emulator_WC711_v1"
PS_DATA_DIR = f"{BASE}/{PROJ}/PS_scenarios"
PS_OUTPUT_DIR_NAME = "ignition_location_uncertainty"

PS_SCENARIO = "s2"
PS_OUTPUT_DIR = f"{PS_DATA_DIR}/{PS_SCENARIO}/{PS_OUTPUT_DIR_NAME}/CNN_inference"

USE_GMFR = False

md = 480
model = "circle"
data_source = "basemax"

window_buffer = 10 # number of pixels to pad around the outer extents of the ignition points

template_raster = rio.open(f"{PS_OUTPUT_DIR}/iteration_0/1/1/{model}_{data_source}_{md}_baseline.tif")
template_profile = template_raster.profile

# get project areas
PS_layer = "project_areas_outputs"
project_areas_f = glob(f"{PS_DATA_DIR}/{PS_SCENARIO}/*.gpkg")
project_areas_gdf = gpd.read_file(project_areas_f[0], layer=PS_layer).to_crs(5070) # there should be only one file here

for proj_id in project_areas_gdf["proj_id"][:3]:
    print(f"{proj_id=}")
    fig, ax = plt.subplots(1, 4, figsize=(14, 7))
    #plt.tight_layout()

    pa_gdf = project_areas_gdf[project_areas_gdf["proj_id"] == proj_id]

    # use PT first iteration for representative ignitions
    ignitions_df = pd.read_csv(f"{PS_OUTPUT_DIR.replace('CNN_inference', '')}/iteration_0/{proj_id}_baseline/{proj_id}_ignitions_locations.csv")
    ignitions_gdf = gpd.GeoDataFrame(ignitions_df,
                                     geometry=gpd.points_from_xy(ignitions_df["x"], ignitions_df["y"])).set_crs(5070)

    minx = ignitions_gdf["row"].min()
    maxx = ignitions_gdf["row"].max()
    miny = ignitions_gdf["col"].min()
    maxy = ignitions_gdf["col"].max()
    window = Window(miny, minx, maxy-miny+window_buffer, maxx-minx+window_buffer)

    with rio.open(f"{PS_OUTPUT_DIR}/pa{proj_id}_rd_cbp_mean.tif") as src:
        rd_cbp_mean_a = src.read(1, window=window)
        #rd_cbp_mean_a = np.ma.masked_where(rd_cbp_mean_a == 0, rd_cbp_mean_a) # no color where RD CBP is 0
        win_transform = src.window_transform(window) # save for later

    with rio.open(f"{PS_OUTPUT_DIR}/pa{proj_id}_rd_cbp_std.tif") as src:
        rd_cbp_std_a = src.read(1, window=window)

    # create a single color map to use for all plots
    min_rd_cbp = np.min(rd_cbp_mean_a - rd_cbp_std_a)
    max_rd_cbp = np.max(rd_cbp_mean_a + rd_cbp_std_a)
    print(f"    {min_rd_cbp=}, {max_rd_cbp=}")

    frac_neg = abs(min_rd_cbp) / (abs(min_rd_cbp) + max_rd_cbp)
    frac_pos = 1 - frac_neg
    print(f"    {frac_neg=}, {frac_pos=}")
    neg_cmap_range = plt.cm.Blues_r(np.linspace(0, 1, int(256 * frac_neg)))
    pos_cmap_range = plt.cm.Reds(np.linspace(0, 1, int(256 * frac_pos)))
    two_colors = np.vstack((neg_cmap_range, pos_cmap_range))
    cmap = colors.LinearSegmentedColormap.from_list("B2R", two_colors)

    kwargs = {"clim": (min_rd_cbp, max_rd_cbp)}
    for i, _subplot in enumerate(ax):
        if i == 0: # mean - 1 SD
            a = rd_cbp_mean_a - rd_cbp_std_a
            rio.plot.show(a, transform=win_transform, ax=ax[i], cmap=cmap, **kwargs)
        if i == 1: # mean
            mfig = rio.plot.show(rd_cbp_mean_a, transform=win_transform, ax=ax[i], cmap=cmap, **kwargs)
        if i == 2: # mean + 1 SD
            a = rd_cbp_mean_a + rd_cbp_std_a
            mfig = rio.plot.show(a, transform=win_transform, ax=ax[i], cmap=cmap, **kwargs)
        if i == 3: # SD
            rio.plot.show(rd_cbp_std_a, transform=win_transform, ax=ax[i], cmap=cmap, **kwargs)

        pa_gdf.plot(column="proj_id", ax=ax[i], alpha=1.0, facecolor="none", edgecolor="black", linewidth=2, zorder=5)
        ignitions_gdf.plot(ax=ax[i], markersize=1, color="black")

    ax[0].title.set_text("Mean RD CBP - 1 SD")
    ax[1].title.set_text("Mean RD CBP")
    ax[2].title.set_text("Mean RD CBP + 1 SD")
    ax[3].title.set_text("RD CBP Std Dev")

    fig.colorbar(mfig.get_images()[0], ax=ax[3])
    plt.suptitle(f"CNN-CBP Relative Difference for Scenario {PS_SCENARIO}, Project Area {proj_id}")
    plt.savefig(f"/tmp/cnn_{PS_SCENARIO}-{proj_id}_rd_CBP_il_uncertainty.png")
# Visualize SD data:1 ends here
