import copy
import resource
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

from burn_emulator.config import dynamic_import
from burn_emulator.constants import DEFAULT_DEVICE, DEFAULT_DTYPE, INF_PROFILE, RUN_DEVICE, Path
from burn_emulator.datasets.utils import compute_crop_region
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
    # how fragile things can be...
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
                    Mx = M.unsqueeze(1) if X.dim() != M.dim() else M
                pdiffs = sample["pdiffs"]
                bounds = sample["bounds"]
                indxes = sample["indxes"]

                with timed(timings, "model forward"):
                    pred = (activation(model(X * Mx, W)) * M).to(torch.float32)

                with timed(timings, "write drain"):
                    _drain(pending, limit=max_write_workers)

                # TODO: add in actual evaluation metrics here with
                #       precomputed burns spreads rasters i.e dataloaders
                #       for now, just only writing out the predictions to
                #       disk for post-hoc evaluation
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


def _run_single_evaluation(
    eval_name: str,
    model_name: str,
    experiment_dir: str | Path,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    out_channels: int,
    iteration: int | None = None,
    scenario: int | None = None,
    scenarios_path: Path | None = None,
    num_sims: int = 1,
    max_write_workers: int = 4,
    debug: bool = False,
    **kwargs: Any,
) -> tuple[int | None, int | None]:
    timings = {} if debug else None
    t_start = time.perf_counter() if debug else None

    experiment_dir = Path(experiment_dir)
    with timed(timings, "model load"):
        model = dynamic_import(model)
        ckpt_path = resolve_model_checkpoint(experiment_dir, kwargs.get("ckpt_path"))

        ckpt = torch.load(ckpt_path, map_location=DEFAULT_DEVICE)
        if next(iter(ckpt.keys())).startswith("_orig_mod"):
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

        model.load_state_dict(ckpt)
        model.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        model.eval()
        model = torch.compile(model)

    outdir = experiment_dir / "inference"
    if iteration is not None and scenario is not None:
        spath = scenarios_path / f"iteration_{iteration}" / f"{scenario}_{eval_name}"
        fpath = dataset.get("init_args", {}).get("fuels_paths")
        outdir /= f"iteration_{iteration}"
        init_args = {
            "ignitions_path": spath / f"{scenario}_ignitions_locations.csv",
            "fuels_paths": {"baseline": spath} if fpath is None else fpath,
        }
        ds_kwargs = copy.deepcopy(dataset)
        ds_kwargs["init_args"] = {**ds_kwargs.get("init_args", {}), **init_args}
    else:
        ds_kwargs = dataset

    ds_kwargs.setdefault("init_args", {}).setdefault(
        "stats_path", experiment_dir / "stats.yaml"
    )
    with timed(timings, "dataset caching"):
        dataset = dynamic_import(ds_kwargs)
        eval_loader = dynamic_import(dataloader, {"dataset": dataset})
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
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        peak_gpu = peak_gpu_gb()
        rows = "\n".join(f"  {label:<18}: {sec:7.2f}s" for label, sec in timings.items())
        gpu_row = "" if peak_gpu is None else f"\n  {'peak gpu mem':<18}: {peak_gpu:7.2f}GB"
        print(
            f"[eval] used checkpoint: {ckpt_path}\n"
            f"[eval] timing  device={RUN_DEVICE}  {run_eval_name}  ({n_batches} batches)\n"
            f"{rows}\n"
            f"  {'total':<18}: {total:7.2f}s\n"
            f"  {'peak cpu mem':<18}: {peak_rss_gb:7.2f}GB"
            f"{gpu_row}",
            flush=True,
        )

    return iteration, scenario


def evaluate(
    eval_name: str,
    model_name: str,
    experiment_dir: str | Path,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    max_write_workers: int,
    out_channels: int,
    debug: bool = False,
    **kwargs: Any,
) -> None:
    _run_single_evaluation(
        eval_name=eval_name,
        model_name=model_name,
        experiment_dir=experiment_dir,
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        activation=activation,
        max_write_workers=max_write_workers,
        out_channels=out_channels,
        debug=debug,
    )


def evaluate_iterations(
    eval_name: str,
    model_name: str,
    experiment_dir: str | Path,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    out_channels: int,
    num_iterations: int,
    num_scenarios: int,
    scenarios_path: str,
    max_workers: int = 4,
    num_sims: int = 1,
    max_write_workers: int = 4,
    debug: bool = False,
    **kwargs: Any,
) -> None:
    tasks = [(i, s) for i in range(num_iterations) for s in range(1, num_scenarios + 1)]
    scenarios_path = Path(scenarios_path)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_evaluation,
                eval_name,
                model_name,
                experiment_dir,
                model,
                dataset,
                dataloader,
                activation,
                out_channels,
                i,
                s,
                scenarios_path,
                num_sims,
                max_write_workers,
                debug,
            ): (i, s)
            for i, s in tasks
        }
        for future in as_completed(futures):
            i, s = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"iteration_{i} scenario_{s} failed: {e}")
