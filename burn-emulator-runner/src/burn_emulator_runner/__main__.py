import logging
import os
import signal
import sys
import tempfile
from copy import deepcopy

import geopandas as gpd
from burn_emulator.config import load_treatment_area
from burn_emulator.constants import TARGET_CRS, Path
from burn_emulator.run import run
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from shapely.geometry.base import BaseGeometry

from burn_emulator_runner.artifacts import (
    bundle_dir,
    load_spec,
    pt_spec,
    read_provenance,
)
from burn_emulator_runner.utils import env_flag, warm_gpu

DEBUG = env_flag("BURN_EMULATOR_DEBUG")

REPORT_ERROR_MAX_CHARS = 1024  # GCS metadata cap is 8 KiB
REPORT_COMPLETED = "completed"  # api dispatch/report.go
REPORT_FAILED = "failed"  # api dispatch/report.go

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("burn_emulator_runner")


def _on_sigterm(signum, frame):
    raise TimeoutError("received SIGTERM (task timeout or cancellation)")


def _gcs_blob(gcs_path: str) -> storage.Blob:
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"not a gs:// path: {gcs_path!r}")
    bucket, _, name = gcs_path.removeprefix("gs://").partition("/")
    return storage.Client().bucket(bucket).blob(name)


def _upload_output(local_path: str, gcs_path: str) -> None:
    _gcs_blob(gcs_path).upload_from_filename(
        local_path, content_type="image/tiff", retry=DEFAULT_RETRY
    )


# overwrites the api's treatment area input with it as geometry and the run meta as properties;
# to_json writes a crs member for non-EPSG:4326
def _upload_meta(treatment_area: BaseGeometry, meta: dict, gcs_path: str) -> None:
    gdf = gpd.GeoDataFrame([meta], geometry=[treatment_area], crs=TARGET_CRS)
    _gcs_blob(gcs_path).upload_from_string(
        gdf.to_json(drop_id=True), content_type="application/geo+json", retry=DEFAULT_RETRY
    )


def _write_report(status: str, error: str = "") -> None:
    report_path = os.environ.get("BURN_EMULATOR_REPORT_PATH", "")
    if not report_path.startswith("gs://"):
        log.warning("no BURN_EMULATOR_REPORT_PATH; the api will only see this run go stale")
        return
    try:
        blob = _gcs_blob(report_path)
        blob.metadata = {
            "status": status,
            "claim_generation": os.environ.get("BURN_EMULATOR_CLAIM_GENERATION", ""),
            "error": error[:REPORT_ERROR_MAX_CHARS],
        }
        blob.upload_from_string(b"", retry=DEFAULT_RETRY)
    except Exception:
        log.exception("failed to write report %s", report_path)


def _run_config(
    spec: dict,
    treatment_area: BaseGeometry,
    ignition_density: float | None,
    out_dir: str,
) -> dict:
    cfg = deepcopy(spec)
    init = cfg.setdefault("dataset", {}).setdefault("init_args", {})
    init["treatment_area"] = treatment_area
    init["fuels_paths"] = {
        "baseline": os.environ["BURN_EMULATOR_BASELINE_FUELS"],
        "treatment": os.environ["BURN_EMULATOR_LEGALMAX_FUELS"],
    }
    init["topo_path"] = os.environ["BURN_EMULATOR_TOPO_PATH"]
    init["fbfm_map_path"] = os.environ["BURN_EMULATOR_FBFM_MAP_PATH"]
    init["ignitions_path"] = None
    init["burn_paths"] = None
    init["wind_ang_paths"] = None
    if ignition_density is not None:
        if ignition_density <= 0:
            raise ValueError("ignition_density must be > 0")
        init["ignition_density"] = ignition_density
    cfg["out_path"] = f"{out_dir.rstrip('/')}/{cfg['model_name']}.tif"
    cfg["debug"] = DEBUG
    return cfg


def _check_provenance(bundle: Path, varloc: str, model_version: str) -> dict | None:
    meta, runner_code_sha = read_provenance(bundle)
    if meta is None:
        log.warning(
            "bundle %s/%s has no bundle_meta.json; cannot check model architecture code",
            varloc,
            model_version,
        )
    elif runner_code_sha is None:
        log.warning(
            "bundle %s/%s names architecture %s, absent from this image; cannot check code",
            varloc,
            model_version,
            meta.get("model_class_path"),
        )
    elif meta.get("model_code_sha256") != runner_code_sha:
        log.warning(
            "model architecture code mismatch varloc=%s model_version=%s: "
            "bundle repo_sha=%s code_sha=%s, runner code_sha=%s",
            varloc,
            model_version,
            meta.get("model_repo_sha"),
            meta.get("model_code_sha256"),
            runner_code_sha,
        )
    return meta


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    run_hash = os.environ.get("BURN_EMULATOR_HASH", "<unknown>")
    try:
        output_meta = os.environ["BURN_EMULATOR_OUTPUT_META"]
        treatment_area = load_treatment_area(
            _gcs_blob(output_meta).download_as_text(retry=DEFAULT_RETRY)
        )
        run_hash = os.environ["BURN_EMULATOR_HASH"]
        output_path = os.environ["BURN_EMULATOR_OUTPUT_PATH"]
        backend = os.environ.get("BURN_EMULATOR_BACKEND", "DL").upper()
        ignition_density_raw = os.environ.get("BURN_EMULATOR_IGNITION_DENSITY")
        ignition_density = float(ignition_density_raw) if ignition_density_raw else None

        # from the api; model_version is the varloc's bundle, unset for PT without one
        varloc = os.environ["BURN_EMULATOR_VARLOC"]
        model_version = os.environ.get("BURN_EMULATOR_MODEL_VERSION")

        warm_gpu()

        bundle_meta = None
        if model_version:
            bundle = bundle_dir(os.environ["BURN_EMULATOR_BUNDLE_DIR"])
            spec = load_spec(bundle)
            if backend == "DL":
                bundle_meta = _check_provenance(bundle, varloc, model_version)
            else:
                bundle_meta, _ = read_provenance(bundle)
        elif backend == "DL":
            raise ValueError("DL run without BURN_EMULATOR_MODEL_VERSION")

        with tempfile.TemporaryDirectory() as local_dir:
            if not model_version:
                spec = pt_spec(varloc, os.environ["BURN_EMULATOR_INPUTS_VERSION"], local_dir)
            cfg = _run_config(spec, treatment_area, ignition_density, local_dir)
            cfg["backend"] = backend

            log.info(
                "run start varloc=%s model_version=%s backend=%s hash=%s",
                varloc,
                model_version,
                backend,
                run_hash,
            )
            result = run(**cfg)

            # SIGTERM is ignored from here on
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            meta = {
                "hash": run_hash,
                "backend": backend,
                "varloc": varloc,
                "inputs_version": os.environ.get("BURN_EMULATOR_INPUTS_VERSION"),
                "ignition_density": cfg["dataset"]["init_args"].get("ignition_density"),
                "output_path": output_path,
            }
            if model_version:
                meta |= {
                    "model_version": model_version,
                    "bundle_uri": os.environ.get("BURN_EMULATOR_BUNDLE_URI"),
                    **(bundle_meta or {}),
                }
            # meta first: the tif marks the run cached
            _upload_meta(treatment_area, meta, output_meta)
            _upload_output(os.fspath(result["out_path"]), output_path)

        log.info("run done hash=%s output_path=%s", run_hash, output_path)
        _write_report(REPORT_COMPLETED)
    except Exception as e:
        # SIGTERM is ignored while reporting the failure
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        log.exception("run failed hash=%s", run_hash)
        _write_report(REPORT_FAILED, f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
