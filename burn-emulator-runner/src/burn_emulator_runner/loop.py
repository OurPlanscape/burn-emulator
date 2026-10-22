import json
import logging
import os
import signal
import socket
import tempfile
import threading
import time
from pathlib import Path

from burn_emulator import rasters
from burn_emulator.constants import INPUT_KEYS
from burn_emulator.run import RunCancelled, enable_caching
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud.storage import Blob

from burn_emulator_runner.artifacts import MODEL_CONFIGS
from burn_emulator_runner.job import (
    REPORT_COMPLETED,
    REPORT_FAILED,
    publish,
    run_job,
    split_gcs,
    storage_client,
    write_report,
)
from burn_emulator_runner.utils import warm_gpu

log = logging.getLogger("burn_emulator_runner")

CLAIM_PREFIX = "_claims/"  # api dispatch.claimPrefix
STATUS_STAGING = "staging"  # api dispatch.statusStaging
STATUS_QUEUED = "queued"  # api dispatch.statusQueued
STATUS_RUNNING = "running"  # api dispatch.statusRunning
STATUS_FAILED = "failed"  # api dispatch.statusFailed

EXPIRED_ERRORS = {
    STATUS_STAGING: "the api stopped before queueing the claim",
    STATUS_QUEUED: "no runner took the claim within {minutes:.0f} min",
    STATUS_RUNNING: "run did not report back before its claim went stale",
}
REPORT_GRACE_SECONDS = 600

# make env-vars sets both roots; the default is this install's burn-emulator-model/data
MODEL_DATA = Path(MODEL_CONFIGS).parent / "data"


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"{name} is not set")
    return v


class ClaimLoop:
    def __init__(self) -> None:
        bucket, _ = split_gcs(_env("BURN_EMULATOR_OUTPUT_URI"))
        self.bucket = storage_client().bucket(bucket)
        self.inputs_root = Path(
            os.environ.get("BURN_EMULATOR_INPUTS_ROOT") or MODEL_DATA / "inputs"
        )
        self.bundles_root = Path(
            os.environ.get("BURN_EMULATOR_BUNDLES_ROOT") or MODEL_DATA / "bundles"
        )
        self.max_models = int(os.environ.get("BURN_EMULATOR_MAX_MODELS", "8"))
        self.poll_seconds = float(os.environ.get("BURN_EMULATOR_POLL_SECONDS", "10"))
        self.run_timeout = float(os.environ.get("BURN_EMULATOR_RUN_TIMEOUT", "1800"))
        job_id = os.environ.get("SLURM_JOB_ID")
        self.runner_id = socket.gethostname() + (f"/{job_id}" if job_id else "")

        self.stop = threading.Event()
        self.cancel = threading.Event()  # the current run's

    def _on_signal(self, signum, frame) -> None:
        log.info("received %s; cancelling the current run, stopping", signal.Signals(signum).name)
        self.stop.set()
        self.cancel.set()

    def serve(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        warm_gpu()
        enable_caching(self.max_models)
        self._preload()
        log.info(
            "watching gs://%s/%s runner=%s max_models=%d inputs=%s bundles=%s",
            self.bucket.name,
            CLAIM_PREFIX,
            self.runner_id,
            self.max_models,
            self.inputs_root,
            self.bundles_root,
        )
        while not self.stop.is_set():
            try:
                taken = self._take_next()
            except Exception:
                log.exception("listing claims failed")
                taken = None
            if taken is None:
                self.stop.wait(self.poll_seconds)
                continue
            self._handle(*taken)
        log.info("stopped")

    # data/inputs/current's rasters, as runs read them; other versions are cached on first use
    def _preload(self) -> None:
        pointer = self.inputs_root / "current"
        if not pointer.is_file():
            log.warning("%s not found; rasters are cached on first use", pointer)
            return
        inputs = self.inputs_root / pointer.read_text().strip()
        paths = [
            inputs / role / f"{key}.tif" for role in ("baseline", "legalmax") for key in INPUT_KEYS
        ]
        paths += [inputs / "topo" / name for name in ("aspect.tif", "slope_degrees.tif")]
        start = time.monotonic()
        nbytes = rasters.preload(paths)
        log.info(
            "cached %d rasters of inputs %s: %.1f GiB in %.0fs",
            len(paths),
            inputs.name,
            nbytes / 2**30,
            time.monotonic() - start,
        )

    # active claims past expires_at -> failed; then the oldest queued claim, DL or PT (the claim's
    # BURN_EMULATOR_BACKEND), taken queued -> running, with its queue window in seconds
    def _take_next(self) -> tuple[Blob, int | None] | None:
        now = int(time.time())
        queued = []
        for b in storage_client().list_blobs(self.bucket, prefix=CLAIM_PREFIX):
            status = (b.metadata or {}).get("status")
            if status not in EXPIRED_ERRORS:
                continue
            expires_at = _meta_int(b, "expires_at")
            if expires_at is not None and expires_at <= now:
                self._expire(b, status)
            elif status == STATUS_QUEUED:
                queued.append(b)
        queued.sort(key=lambda b: _meta_int(b, "updated_at") or 0)
        for blob in queued:
            if self.stop.is_set():
                return None
            window = _window(blob)
            now = int(time.time())
            if self._update(
                blob,
                status=STATUS_RUNNING,
                updated_at=str(now),
                expires_at=str(now + int(self.run_timeout) + REPORT_GRACE_SECONDS),
                runner=self.runner_id,
            ):
                return blob, window
        return None

    # active -> failed, with GetJob's error for its status
    def _expire(self, blob: Blob, status: str) -> None:
        error = EXPIRED_ERRORS[status].format(minutes=(_window(blob) or 0) / 60)
        if self._update(
            blob,
            status=STATUS_FAILED,
            error=error,
            updated_at=str(int(time.time())),
            expires_at=None,
        ):
            log.info("claim %s expired while %s", blob.name.removeprefix(CLAIM_PREFIX), status)

    # metadata patch under the claim's generation and metageneration; False if either moved
    def _update(self, blob: Blob, **metadata: str | None) -> bool:
        gen, mgen = blob.generation, blob.metageneration
        blob.metadata = metadata
        try:
            blob.patch(if_generation_match=gen, if_metageneration_match=mgen)
        except (PreconditionFailed, NotFound):
            return False
        return True

    # queue_window: the claim's expires_at - updated_at when it was taken, reused on a requeue
    def _handle(self, claim: Blob, queue_window: int | None) -> None:
        key = claim.name.removeprefix(CLAIM_PREFIX)
        claim_generation = str(claim.generation)
        report_path = ""
        timed_out = threading.Event()
        self.cancel = cancel = threading.Event()
        if self.stop.is_set():
            cancel.set()

        def on_timeout() -> None:
            timed_out.set()
            cancel.set()

        timer = threading.Timer(self.run_timeout, on_timeout)
        timer.daemon = True
        try:
            env = json.loads(claim.download_as_text(if_generation_match=claim.generation))
            report_path = env.get("BURN_EMULATOR_REPORT_PATH", "")
            env |= self._paths(env)
            log.info("claim %s taken", key)
            timer.start()
            with tempfile.TemporaryDirectory() as local_dir:
                out = run_job(env, local_dir, cancel)
                timer.cancel()
                publish(out)
            log.info("run done %s output_path=%s", key, out.output_path)
            write_report(report_path, claim_generation, REPORT_COMPLETED)
        except RunCancelled as e:
            now = int(time.time())
            if timed_out.is_set():
                log.error("run timed out %s: %s", key, e)
                error = f"run exceeded {self.run_timeout:g}s: {e}"
                write_report(report_path, claim_generation, REPORT_FAILED, error)
            elif self._update(
                claim,
                status=STATUS_QUEUED,
                updated_at=str(now),
                expires_at=str(now + queue_window) if queue_window else None,
                runner=None,
            ):
                log.info("claim %s requeued: %s", key, e)
            else:
                log.warning("claim %s changed while cancelling; leaving it", key)
        except Exception as e:
            log.exception("run failed %s", key)
            write_report(report_path, claim_generation, REPORT_FAILED, f"{type(e).__name__}: {e}")
        finally:
            timer.cancel()

    # the paths dispatch.runnerEnv leaves out of a metal claim, in its layout
    def _paths(self, env: dict[str, str]) -> dict[str, str]:
        inputs = self.inputs_root / env["BURN_EMULATOR_INPUTS_VERSION"]
        paths = {
            "BURN_EMULATOR_BASELINE_FUELS": os.fspath(inputs / "baseline"),
            "BURN_EMULATOR_LEGALMAX_FUELS": os.fspath(inputs / "legalmax"),
            "BURN_EMULATOR_TOPO_PATH": os.fspath(inputs / "topo"),
            "BURN_EMULATOR_FBFM_MAP_PATH": os.fspath(
                inputs / "fbfm" / "fbfm_behavior_adjectives.csv"
            ),
        }
        if model_version := env.get("BURN_EMULATOR_MODEL_VERSION"):
            paths["BURN_EMULATOR_BUNDLE_DIR"] = os.fspath(
                self.bundles_root / env["BURN_EMULATOR_VARLOC"] / model_version
            )
        return paths


def _meta_int(blob: Blob, key: str) -> int | None:
    try:
        return int((blob.metadata or {})[key])
    except (KeyError, ValueError):
        return None


# expires_at - updated_at in seconds; None without both
def _window(blob: Blob) -> int | None:
    expires_at, updated_at = _meta_int(blob, "expires_at"), _meta_int(blob, "updated_at")
    if expires_at is None or updated_at is None:
        return None
    return expires_at - updated_at


def serve() -> None:
    ClaimLoop().serve()
