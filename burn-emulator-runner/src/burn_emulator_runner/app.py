import asyncio
import logging
import os
from contextlib import asynccontextmanager
from copy import deepcopy

from burn_emulator.config import load_treatment_area
from burn_emulator.run import run
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from burn_emulator_runner.artifacts import bundle_dir, load_spec
from burn_emulator_runner.utils import env_flag, warm_gpu


BASELINE_FUELS = os.environ["BURN_EMULATOR_BASELINE_FUELS"]
LEGALMAX_FUELS = os.environ["BURN_EMULATOR_LEGALMAX_FUELS"]
TOPO_PATH = os.environ["BURN_EMULATOR_TOPO_PATH"]
GPU_SLOTS = int(os.environ.get("BURN_EMULATOR_GPU_SLOTS", "1"))
DEBUG = env_flag("BURN_EMULATOR_DEBUG")

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
async def infer(req: InferRequest) -> InferResponse:
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
        cfg = await asyncio.to_thread(_run_config, spec, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid treatment_area: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"resolving treatment_area: {e}") from e

    async with _slots:
        log.info("run start varloc=%s version=%s hash=%s", req.varloc, req.version, req.hash)
        try:
            timing = await asyncio.to_thread(run, **cfg)
        except Exception as e:
            log.exception("run failed hash=%s", req.hash)
            raise HTTPException(status_code=500, detail=f"run failed: {e}") from e
        log.info("run done hash=%s output_path=%s", req.hash, req.output_path)

    return InferResponse(status="completed", output_path=req.output_path, timing=timing)


def _run_config(spec: dict, req: InferRequest) -> dict:
    cfg = deepcopy(spec)
    init = cfg.setdefault("dataset", {}).setdefault("init_args", {})
    # TODO: warn user if treatment area does not align with varloc
    init["treatment_area"] = load_treatment_area(req.treatment_area)
    init["fuels_paths"] = {"baseline": BASELINE_FUELS, "treatment": LEGALMAX_FUELS}
    init["topo_path"] = TOPO_PATH
    init["ignitions_path"] = None
    cfg["out_path"] = req.output_path
    cfg["debug"] = DEBUG
    return cfg
