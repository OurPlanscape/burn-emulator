import logging
import os
import signal
import sys
import tempfile
from copy import deepcopy

from burn_emulator.config import load_treatment_area
from burn_emulator.run import run
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

from burn_emulator_runner.artifacts import bundle_dir, load_spec, read_provenance
from burn_emulator_runner.utils import env_flag, warm_gpu

DEBUG = env_flag("BURN_EMULATOR_DEBUG")

# GCS custom metadata is capped at 8 KiB per object
REPORT_ERROR_MAX_CHARS = 1024

# report statuses; must match reportCompleted / reportFailed in
# burn-emulator-api's dispatch/report.go.
REPORT_COMPLETED = "completed"
REPORT_FAILED = "failed"

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
    varloc: str,
    treatment_area: str,
    treatment_area_crs: str,
    ignition_density: float | None,
    out_dir: str,
) -> dict:
    cfg = deepcopy(spec)
    init = cfg.setdefault("dataset", {}).setdefault("init_args", {})
    # TODO: warn user if treatment area does not align with varloc
    init["treatment_area"] = load_treatment_area(treatment_area, treatment_area_crs)
    # injecting fuels and topo from api env vars
    init["fuels_paths"] = {
        "baseline": os.environ["BURN_EMULATOR_BASELINE_FUELS"],
        "treatment": os.environ["BURN_EMULATOR_LEGALMAX_FUELS"],
    }
    init["topo_path"] = os.environ["BURN_EMULATOR_TOPO_PATH"]
    init["ignitions_path"] = None
    if ignition_density is not None:
        if ignition_density <= 0:
            raise ValueError("ignition_density must be > 0")
        init["ignition_density"] = ignition_density
    cfg["out_path"] = f"{out_dir.rstrip('/')}/{cfg['model_name']}.tif"
    cfg["debug"] = DEBUG
    return cfg


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    run_hash = os.environ.get("BURN_EMULATOR_HASH", "<unknown>")
    try:
        varloc = os.environ["BURN_EMULATOR_VARLOC"]
        model_version = os.environ["BURN_EMULATOR_MODEL_VERSION"]
        treatment_area = os.environ["BURN_EMULATOR_TREATMENT_AREA"]
        treatment_area_crs = os.environ["BURN_EMULATOR_TREATMENT_AREA_CRS"]
        run_hash = os.environ["BURN_EMULATOR_HASH"]
        output_path = os.environ["BURN_EMULATOR_OUTPUT_PATH"]
        ignition_density_raw = os.environ.get("BURN_EMULATOR_IGNITION_DENSITY")
        ignition_density = float(ignition_density_raw) if ignition_density_raw else None

        warm_gpu()

        bundle = bundle_dir(varloc, model_version)
        spec = load_spec(bundle)

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

        # TODO: test memory limits on the largest varlocs; /tmp is in-memory on Cloud
        # Run, so the raster counts against burn_emulator_runner_{gpu,cpu}_ram on top of the run.
        with tempfile.TemporaryDirectory() as local_dir:
            cfg = _run_config(
                spec, varloc, treatment_area, treatment_area_crs, ignition_density, local_dir
            )

            log.info(
                "run start varloc=%s model_version=%s backend=%s hash=%s",
                varloc,
                model_version,
                os.environ.get("BURN_EMULATOR_BACKEND", "DL"),
                run_hash,
            )
            result = run(**cfg)

            # NOTE: the run succeeded; friends don't let a late SIGTERM turn them into
            # a failure (mid-upload or between the upload and the completed report).
            # the local file is complete here, and an interrupted upload creates no
            # object, so the api can never serve a partial or blank raster.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            local_out = os.fspath(result["out_path"])
            _upload_output(local_out, f"{output_path.rstrip('/')}/{os.path.basename(local_out)}")

        log.info("run done hash=%s output_path=%s", run_hash, output_path)
        _write_report(REPORT_COMPLETED)
    except Exception as e:
        # the run already failed; don't let a SIGTERM stop the reporting
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        log.exception("run failed hash=%s", run_hash)
        _write_report(REPORT_FAILED, f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
