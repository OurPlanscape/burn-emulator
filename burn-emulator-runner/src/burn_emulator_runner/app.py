import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from copy import deepcopy

from burn_emulator.config import load_treatment_area
from burn_emulator.run import RunCancelled, run
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from burn_emulator_runner.artifacts import bundle_dir, load_spec, read_provenance
from burn_emulator_runner.utils import env_flag, warm_gpu

BASELINE_FUELS = os.environ["BURN_EMULATOR_BASELINE_FUELS"]
LEGALMAX_FUELS = os.environ["BURN_EMULATOR_LEGALMAX_FUELS"]
TOPO_PATH = os.environ["BURN_EMULATOR_TOPO_PATH"]
GPU_SLOTS = int(os.environ.get("BURN_EMULATOR_GPU_SLOTS", "4"))
DEBUG = env_flag("BURN_EMULATOR_DEBUG")
DISCONNECT_POLL_INTERVAL = 1.0

log = logging.getLogger("uvicorn.error").getChild("runner")

# how many run() calls may execute on this instance's GPU at once.
_slots = asyncio.Semaphore(GPU_SLOTS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    warm_gpu()
    yield


app = FastAPI(title="burn-emulator-runner", lifespan=lifespan)


class InferRequest(BaseModel):
    varloc: str
    version: str
    treatment_area: str
    treatment_area_crs: str
    ignition_density: float | None = None
    hash: str
    output_path: str


class InferResponse(BaseModel):
    status: str
    output_path: str
    timing: dict | None = None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest, request: Request) -> InferResponse:
    try:
        bundle = await asyncio.to_thread(bundle_dir, req.varloc, req.version)
        spec = await asyncio.to_thread(load_spec, bundle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"loading model bundle: {e}") from e

    try:
        meta, runner_code_sha = await asyncio.to_thread(read_provenance, bundle)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"reading bundle provenance: {e}") from e

    if meta is None:
        log.warning(
            "bundle %s/%s has no bundle_meta.json; cannot check model architecture code",
            req.varloc,
            req.version,
        )
    elif runner_code_sha is None:
        log.warning(
            "bundle %s/%s names architecture %s, absent from this image; cannot check code",
            req.varloc,
            req.version,
            meta.get("model_class_path"),
        )
    elif meta.get("model_code_sha256") != runner_code_sha:
        log.warning(
            "model architecture code mismatch varloc=%s version=%s: "
            "bundle repo_sha=%s code_sha=%s, runner code_sha=%s",
            req.varloc,
            req.version,
            meta.get("model_repo_sha"),
            meta.get("model_code_sha256"),
            runner_code_sha,
        )

    try:
        cfg = await asyncio.to_thread(_run_config, spec, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"building run config: {e}") from e

    async with _slots:
        log.info("run start varloc=%s version=%s hash=%s", req.varloc, req.version, req.hash)
        cancel = threading.Event()
        run_task = asyncio.create_task(asyncio.to_thread(run, cancel=cancel, **cfg))
        while not run_task.done():
            if await request.is_disconnected():
                cancel.set()
                break
            await asyncio.sleep(DISCONNECT_POLL_INTERVAL)
        try:
            timing = await run_task
        except RunCancelled:
            log.info("run cancelled hash=%s (client disconnected)", req.hash)
            raise HTTPException(status_code=499, detail="client disconnected") from None
        except Exception as e:
            log.exception("run failed hash=%s", req.hash)
            raise HTTPException(status_code=500, detail=f"run failed: {e}") from e
        log.info("run done hash=%s output_path=%s", req.hash, req.output_path)

    return InferResponse(status="completed", output_path=req.output_path, timing=timing)


def _run_config(spec: dict, req: InferRequest) -> dict:
    cfg = deepcopy(spec)
    init = cfg.setdefault("dataset", {}).setdefault("init_args", {})
    # TODO: warn user if treatment area does not align with varloc
    init["treatment_area"] = load_treatment_area(req.treatment_area, req.treatment_area_crs)
    init["fuels_paths"] = {"baseline": BASELINE_FUELS, "treatment": LEGALMAX_FUELS}
    init["topo_path"] = TOPO_PATH
    init["ignitions_path"] = None
    if req.ignition_density is not None:
        if req.ignition_density <= 0:
            raise ValueError("ignition_density must be > 0")
        init["ignition_density"] = req.ignition_density
    cfg["out_path"] = f"{req.output_path.rstrip('/')}/{cfg['model_name']}_run.tif"
    cfg["debug"] = DEBUG
    return cfg
