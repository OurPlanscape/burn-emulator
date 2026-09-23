import logging
import os
import sys
from copy import deepcopy

from burn_emulator.config import load_treatment_area
from burn_emulator.run import run

from burn_emulator_runner.artifacts import bundle_dir, load_spec, read_provenance
from burn_emulator_runner.utils import env_flag, warm_gpu

BASELINE_FUELS = os.environ["BURN_EMULATOR_BASELINE_FUELS"]
LEGALMAX_FUELS = os.environ["BURN_EMULATOR_LEGALMAX_FUELS"]
TOPO_PATH = os.environ["BURN_EMULATOR_TOPO_PATH"]
DEBUG = env_flag("BURN_EMULATOR_DEBUG")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("burn_emulator_runner")


def _run_config(
    spec: dict,
    varloc: str,
    treatment_area: str,
    treatment_area_crs: str,
    ignition_density: float | None,
    output_path: str,
) -> dict:
    cfg = deepcopy(spec)
    init = cfg.setdefault("dataset", {}).setdefault("init_args", {})
    # TODO: warn user if treatment area does not align with varloc
    init["treatment_area"] = load_treatment_area(treatment_area, treatment_area_crs)
    init["fuels_paths"] = {"baseline": BASELINE_FUELS, "treatment": LEGALMAX_FUELS}
    init["topo_path"] = TOPO_PATH
    init["ignitions_path"] = None
    if ignition_density is not None:
        if ignition_density <= 0:
            raise ValueError("ignition_density must be > 0")
        init["ignition_density"] = ignition_density
    cfg["out_path"] = f"{output_path.rstrip('/')}/{cfg['model_name']}_run.tif"
    cfg["debug"] = DEBUG
    return cfg


def main() -> None:
    run_hash = os.environ.get("BURN_EMULATOR_HASH", "<unknown>")
    try:
        varloc = os.environ["BURN_EMULATOR_VARLOC"]
        version = os.environ["BURN_EMULATOR_VERSION"]
        treatment_area = os.environ["BURN_EMULATOR_TREATMENT_AREA"]
        treatment_area_crs = os.environ["BURN_EMULATOR_TREATMENT_AREA_CRS"]
        run_hash = os.environ["BURN_EMULATOR_HASH"]
        output_path = os.environ["BURN_EMULATOR_OUTPUT_PATH"]
        ignition_density_raw = os.environ.get("BURN_EMULATOR_IGNITION_DENSITY")
        ignition_density = float(ignition_density_raw) if ignition_density_raw else None

        warm_gpu()

        bundle = bundle_dir(varloc, version)
        spec = load_spec(bundle)

        meta, runner_code_sha = read_provenance(bundle)
        if meta is None:
            log.warning(
                "bundle %s/%s has no bundle_meta.json; cannot check model architecture code",
                varloc,
                version,
            )
        elif runner_code_sha is None:
            log.warning(
                "bundle %s/%s names architecture %s, absent from this image; cannot check code",
                varloc,
                version,
                meta.get("model_class_path"),
            )
        elif meta.get("model_code_sha256") != runner_code_sha:
            log.warning(
                "model architecture code mismatch varloc=%s version=%s: "
                "bundle repo_sha=%s code_sha=%s, runner code_sha=%s",
                varloc,
                version,
                meta.get("model_repo_sha"),
                meta.get("model_code_sha256"),
                runner_code_sha,
            )

        cfg = _run_config(
            spec, varloc, treatment_area, treatment_area_crs, ignition_density, output_path
        )

        log.info("run start varloc=%s version=%s hash=%s", varloc, version, run_hash)
        run(**cfg)
        log.info("run done hash=%s output_path=%s", run_hash, output_path)
    except Exception:
        log.exception("run failed hash=%s", run_hash)
        sys.exit(1)


if __name__ == "__main__":
    main()
