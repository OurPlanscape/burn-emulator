import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

from burn_emulator.config import dynamic_import
from burn_emulator.constants import DEFAULT_DEVICE, DEFAULT_DTYPE, INF_PROFILE, RUN_DEVICE, Path
from burn_emulator.datasets.utils import compute_crop_region
from burn_emulator.run import _center_component, peak_memory_rows, timing_report
from burn_emulator.utils import peak_gpu_gb, resolve_model_checkpoint, timed


def _drain(pending: list[Future], limit: int) -> None:
    while len(pending) >= limit:
        pending.pop(0).result()


def _compute_crop_region_helper(
    b: int, pdiffs: tuple, bounds: tuple
) -> tuple[int, int, int, int, int, int, int, int]:
    ydiff, xdiff = pdiffs
    ymin, ymax, xmin, xmax = bounds
    y0, y1, x0, x1, y_start, x_start = compute_crop_region(
        (ymin[b], ymax[b], xmin[b], xmax[b]), (ydiff[b], xdiff[b])
    )
    return y0, y1, x0, x1, y_start, x_start, y1 - y0, x1 - x0


def _write_batch(
    pred: np.ndarray,
    pdiffs: tuple,
    bounds: tuple,
    indxes: tuple,
    ignitions: Any,
    shape: tuple,
    profile: dict,
    eval_name: str,
    outdir: Path,
) -> None:
    sidx, _ = indxes

    for b in range(pred.shape[0]):
        y0, y1, x0, x1, y_start, x_start, h, w = _compute_crop_region_helper(b, pdiffs, bounds)

        canvas = torch.zeros(shape, dtype=torch.float32)
        canvas[:, y0:y1, x0:x1] = torch.from_numpy(
            pred[b, :, y_start : y_start + h, x_start : x_start + w]
        )

        ignition_number = str(ignitions.iloc[int(sidx[b])]["ignition_number"])
        cbp_burn = str(ignitions.iloc[int(sidx[b])]["cbp_burn"])
        sample_path = outdir / cbp_burn / ignition_number / f"{eval_name}.tif"
        sample_path.parent.mkdir(exist_ok=True, parents=True)

        with rasterio.open(sample_path, "w", **profile) as dst:
            dst.write(canvas.numpy())


def evaluate_model(
    eval_name: str,
    model: torch.nn.Module,
    eval_loader: DataLoader,
    activation: torch.nn.Module,
    out_channels: int,
    num_sims: int,
    outdir: Path,
    max_write_workers: int,
    timings: dict[str, float] | None = None,
) -> int:
    profile = eval_loader.dataset.profile | INF_PROFILE
    shape = (out_channels, profile["height"], profile["width"])
    profile.update({"count": out_channels})

    pending: list[Future] = []
    n_batches = 0

    eval_start_time = time.perf_counter()
    with torch.no_grad(), ThreadPoolExecutor(max_workers=max_write_workers) as pool:
        for _ in range(num_sims):
            for sample in eval_loader:
                with timed(timings, "data_movement"):
                    X = sample["x"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                    W = sample["wind"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                    M = sample["mask"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                pdiffs = sample["pdiffs"]
                bounds = sample["bounds"]
                indxes = sample["indxes"]

                with timed(timings, "model forward"):
                    pred = (activation(model(X, W)) * M).to(torch.float32)
                    burned = _center_component(pred.argmax(dim=1) != 0)
                    pred = pred * burned.unsqueeze(1).to(torch.float32)
                    pred[:, 0] += (~burned).to(torch.float32)

                with timed(timings, "write drain"):
                    _drain(pending, limit=max_write_workers)

                # TODO: evaluation metrics; predictions are only written to disk
                pending.append(
                    pool.submit(
                        _write_batch,
                        pred.cpu().numpy(),
                        pdiffs,
                        bounds,
                        indxes,
                        eval_loader.dataset.ignitions,
                        shape,
                        profile.copy(),
                        eval_name,
                        outdir,
                    )
                )
                n_batches += 1
        with timed(timings, "write drain"):
            _drain(pending, limit=1)
    eval_perf_time = time.perf_counter() - eval_start_time

    peak_gpu = peak_gpu_gb()
    tp = {
        "model": eval_name,
        "num_batches": len(eval_loader),
        "batch_size": eval_loader.batch_size,
        "max_memory_alloc": None if peak_gpu is None else round(peak_gpu, 2),
        "eval_perf_time": round(eval_perf_time, 2),
    }
    df = pd.DataFrame([tp])
    header = not (outdir / "throughput.csv").exists()
    df.to_csv(outdir / "throughput.csv", mode="a", index=False, header=header)
    return n_batches


def _load_model(
    model: dict, experiment_dir: Path, ckpt_path: str | None, timings: dict | None
) -> tuple[torch.nn.Module, Path]:
    with timed(timings, "model load"):
        model = dynamic_import(model)
        ckpt_path = resolve_model_checkpoint(experiment_dir, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=DEFAULT_DEVICE)
        if next(iter(ckpt.keys())).startswith("_orig_mod"):
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

        model.load_state_dict(ckpt)
        model.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        model.eval()
        model = torch.compile(model)
    return model, ckpt_path


def evaluate_pt(
    eval_name: str,
    dataset: Any,
    out_channels: int,
    outdir: Path,
    pt_workers: int | None,
    timings: dict[str, float] | None = None,
) -> dict:
    # optional [data] extra
    from burn_emulator import pt

    # VarLoc without burn_paths serves the first fuels role
    fkey = list(dataset.fuels_paths)[0]
    profile = dataset.profile | INF_PROFILE
    profile.update({"count": out_channels})

    with timed(timings, "pt inputs"):
        raw = pt.read_raw_inputs(dataset.fuels_paths[fkey], dataset.topo_path, dataset.profile)
        state = {"inputs": pt.pt_inputs(raw), "profile": profile}

    ignitions = dataset.ignitions
    out_name = f"{eval_name}.tif"
    sim_runtime, stops = 0.0, Counter()
    eval_start_time = time.perf_counter()
    pt_workers = pt.resolve_workers(profile["height"] * profile["width"], pt_workers)
    with pt.pool(state, pt_workers) as pool:
        futures = [
            pool.submit(
                pt.fire_type_task,
                (int(ignition["row"]), int(ignition["col"])),
                float(dataset.wind_angles[fkey].iloc[i]),
                outdir / str(ignition["cbp_burn"]) / str(ignition["ignition_number"]) / out_name,
            )
            for i, (_, ignition) in enumerate(ignitions.iterrows())
        ]
        with timed(timings, "pt simulate + write"):
            for future in as_completed(futures):
                result = future.result()
                sim_runtime += result["runtime"]
                stops.update(result["stop_conditions"])
    eval_perf_time = time.perf_counter() - eval_start_time

    tp = {
        "model": eval_name,
        "num_batches": len(futures),
        "batch_size": 1,
        "max_memory_alloc": None,
        "eval_perf_time": round(eval_perf_time, 2),
    }
    df = pd.DataFrame([tp])
    header = not (outdir / "throughput.csv").exists()
    df.to_csv(outdir / "throughput.csv", mode="a", index=False, header=header)

    n_sims = len(futures)
    return {
        "n_sims": n_sims,
        "rows": {
            "workers": pt_workers,
            "sim cpu": f"{sim_runtime:7.2f}s",
            "sim mean": f"{1000 * sim_runtime / max(n_sims, 1):7.2f}ms ({n_sims} sims)",
            **{f"stop: {k}": v for k, v in stops.items()},
        },
    }


def evaluate(
    eval_name: str,
    model_name: str,
    experiment_dir: str | Path,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    out_channels: int,
    max_write_workers: int = 4,
    num_sims: int = 1,
    ckpt_path: str | None = None,
    debug: bool = False,
    pyretechnics: bool = False,
    pt_workers: int | None = None,
    **kwargs: Any,
) -> None:
    timings = {} if debug else None
    t_start = time.perf_counter() if debug else None

    experiment_dir = Path(experiment_dir)
    if not pyretechnics:
        model, ckpt_path = _load_model(model, experiment_dir, ckpt_path, timings)

    outdir = experiment_dir / "inference"

    dataset.setdefault("init_args", {}).setdefault("stats_path", experiment_dir / "stats.yaml")
    with timed(timings, "dataset caching"):
        dataset = dynamic_import(dataset)
        eval_loader = dynamic_import(dataloader, {"dataset": dataset})

    if pyretechnics:
        run_eval_name = f"pt_{eval_name}"
        pt_stats = evaluate_pt(run_eval_name, dataset, out_channels, outdir, pt_workers, timings)
        if debug:
            total = time.perf_counter() - t_start
            timings["other"] = max(total - sum(timings.values()), 0.0)
            timing_report(
                "eval",
                f"pyretechnics  {run_eval_name}  ({pt_stats['n_sims']} ignitions)",
                timings,
                total,
                {**pt_stats["rows"], **peak_memory_rows()},
            )
        return

    activation = dynamic_import(activation)
    run_eval_name = f"{model_name}_{eval_name}"
    with torch.no_grad():
        n_batches = evaluate_model(
            eval_name=run_eval_name,
            model=model,
            eval_loader=eval_loader,
            activation=activation,
            out_channels=out_channels,
            num_sims=num_sims,
            outdir=outdir,
            max_write_workers=max_write_workers,
            timings=timings,
        )

    if debug:
        total = time.perf_counter() - t_start
        timings["other"] = max(total - sum(timings.values()), 0.0)
        print(f"[eval] used checkpoint: {ckpt_path}", flush=True)
        timing_report(
            "eval",
            f"device={RUN_DEVICE}  {run_eval_name}  ({n_batches} batches)",
            timings,
            total,
            peak_memory_rows(),
        )
