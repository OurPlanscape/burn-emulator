from pathlib import Path

import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from omegaconf import OmegaConf
from rasterio.plot import show
from rasterio.windows import Window

from burn_emulator.constants import OUTDIR

MODEL_NAME = "wc711_480_single_df64"
CONFIG_DIR = Path("configs/wc711/s2_baseline")
INFERENCE_DIR = OUTDIR / "inference"
WINDOW_BUFFER = 10  # pixels to pad around the ignition points' extent


def ignitions_for_scenario(scenario: str) -> pd.DataFrame:
    config = OmegaConf.load(CONFIG_DIR / f"{scenario}_test.yaml")
    return pd.read_csv(config.dataset.init_args.ignitions_path)


def diverging_cmap(rd_cbp: np.ndarray) -> tuple[colors.Colormap, float, float]:
    vmin, vmax = float(rd_cbp.min()), float(rd_cbp.max())
    frac_neg = abs(vmin) / (abs(vmin) + vmax)
    neg = plt.cm.Blues_r(np.linspace(0, 1, max(int(256 * frac_neg), 1)))
    pos = plt.cm.Reds(np.linspace(0, 1, max(int(256 * (1 - frac_neg)), 1)))
    cmap = colors.LinearSegmentedColormap.from_list("B2R", np.vstack([neg, pos]))
    return cmap, vmin, vmax


def main() -> None:
    scenario_dirs = sorted(
        (d for d in INFERENCE_DIR.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        print(f"{scenario=}")

        ignitions_df = ignitions_for_scenario(scenario)
        ignitions_gdf = gpd.GeoDataFrame(
            ignitions_df, geometry=gpd.points_from_xy(ignitions_df["x"], ignitions_df["y"])
        ).set_crs(5070)

        minx, maxx = ignitions_df["row"].min(), ignitions_df["row"].max()
        miny, maxy = ignitions_df["col"].min(), ignitions_df["col"].max()
        window = Window(miny, minx, maxy - miny + WINDOW_BUFFER, maxx - minx + WINDOW_BUFFER)

        rd_cbp_path = scenario_dir / f"{MODEL_NAME}_{scenario}_rd_cbp.tif"
        with rio.open(rd_cbp_path) as src:
            rd_cbp = src.read(1, window=window)
            win_transform = src.window_transform(window)

        cmap, vmin, vmax = diverging_cmap(rd_cbp)

        fig, ax = plt.subplots(figsize=(7, 7))
        mfig = show(rd_cbp, transform=win_transform, ax=ax, cmap=cmap, clim=(vmin, vmax))
        ignitions_gdf.plot(ax=ax, markersize=1, color="black")
        ax.set_title(f"CNN Relative Difference CBP — scenario {scenario}")
        fig.colorbar(mfig.get_images()[0], ax=ax)

        out_path = scenario_dir / f"{MODEL_NAME}_{scenario}_rd_cbp.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_path}")


if __name__ == "__main__":
    main()
