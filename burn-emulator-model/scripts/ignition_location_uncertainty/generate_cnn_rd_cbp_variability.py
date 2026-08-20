import numpy as np
import rasterio as rio

from burn_emulator.constants import OUTDIR, Path

MODEL_NAME = "wc711_480_single_df64"
BASELINE, LEGALMAX = "baseline", "legalmax"

INFERENCE_DIR = OUTDIR / "inference"


def burned_mask(bands: np.ndarray) -> np.ndarray:
    return np.argmax(bands, axis=0) != 0


def compute_cbp(scenario_dir: Path, scenario: str, treatment: str) -> np.ndarray:
    ignition_dirs = sorted(
        (d for d in scenario_dir.iterdir() if d.is_dir()), key=lambda d: int(d.name)
    )

    burned_sum = None
    for ignition_dir in ignition_dirs:
        raster_path = ignition_dir / f"{MODEL_NAME}_{scenario}_{treatment}.tif"
        with rio.open(raster_path) as src:
            burned = burned_mask(src.read()).astype(np.float32)
        burned_sum = burned if burned_sum is None else burned_sum + burned

    return burned_sum / len(ignition_dirs)


def main() -> None:
    scenario_dirs = sorted(
        (d for d in INFERENCE_DIR.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        print(f"{scenario=}")

        template_path = next(scenario_dir.glob(f"*/{MODEL_NAME}_{scenario}_{BASELINE}.tif"))
        with rio.open(template_path) as src:
            cbp_profile = src.profile | {"count": 1}

        cbp = {}
        for treatment in (BASELINE, LEGALMAX):
            print(f"    {treatment=}")
            cbp[treatment] = compute_cbp(scenario_dir, scenario, treatment)

            cbp_path = scenario_dir / f"{MODEL_NAME}_{scenario}_{treatment}_cbp.tif"
            with rio.open(cbp_path, "w", **cbp_profile) as dst:
                dst.write(cbp[treatment], 1)
        print("    Finished saving CBP rasters")
        print("    Computing rel diff CBP")
        baseline_cbp, legalmax_cbp = cbp[BASELINE], cbp[LEGALMAX]
        masked_baseline_cbp = np.ma.masked_where(baseline_cbp == 0, baseline_cbp)
        rd_cbp = (legalmax_cbp - masked_baseline_cbp) / masked_baseline_cbp
        rd_cbp[(baseline_cbp == 0) & (legalmax_cbp > 0)] = np.max(rd_cbp)

        rd_cbp_path = scenario_dir / f"{MODEL_NAME}_{scenario}_rd_cbp.tif"
        with rio.open(rd_cbp_path, "w", **cbp_profile) as dst:
            dst.write(rd_cbp.astype(np.float32), 1)
        print("    Finished saving rel diff CBP raster")


if __name__ == "__main__":
    main()
