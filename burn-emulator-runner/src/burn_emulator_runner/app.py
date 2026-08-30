import asyncio
import os
from contextlib import asynccontextmanager
from copy import deepcopy

from burn_emulator.config import load_treatment_area
from burn_emulator.run import run
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from burn_emulator_runner.artifacts import bundle_dir, load_spec

BASELINE_FUELS = os.environ["BURN_EMULATOR_BASELINE_FUELS"]
LEGALMAX_FUELS = os.environ["BURN_EMULATOR_LEGALMAX_FUELS"]
TOPO_PATH = os.environ["BURN_EMULATOR_TOPO_PATH"]
GPU_SLOTS = int(os.environ.get("BURN_EMULATOR_GPU_SLOTS", "1"))
DEBUG = bool(os.environ.get("BURN_EMULATOR_DEBUG"))

# how many run() calls may execute on this instance's GPU at once.
_slots = asyncio.Semaphore(GPU_SLOTS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warm_gpu()
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
    # bundle lookup + YAML parse are off the GPU path, so they overlap an
    # in-flight run.
    try:
        bundle = await asyncio.to_thread(bundle_dir, req.varloc, req.version)
        spec = await asyncio.to_thread(load_spec, bundle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"loading model bundle: {e}") from e

    cfg = _run_config(spec, req)

    async with _slots:
        try:
            timing = await asyncio.to_thread(run, **cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"run failed: {e}") from e

    return InferResponse(status="completed", output_path=req.output_path, timing=timing)


def _run_config(spec: dict, req: InferRequest) -> dict:
    cfg = deepcopy(spec)
    init = cfg.setdefault("dataset", {}).setdefault("init_args", {})
    init["treatment_area"] = load_treatment_area(req.treatment_area)
    init["fuels_paths"] = {"baseline": BASELINE_FUELS, "treatment": LEGALMAX_FUELS}
    init["topo_path"] = TOPO_PATH
    init["ignitions_path"] = None
    cfg["out_path"] = req.output_path
    cfg["debug"] = DEBUG
    return cfg


def _warm_gpu() -> None:
    import torch
    from burn_emulator.constants import RUN_DEVICE

    if RUN_DEVICE == "cuda" and torch.cuda.is_available():
        torch.zeros(1, device="cuda")  # pay CUDA init once, at startup
