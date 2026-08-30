import copy
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

from burn_emulator.config import dynamic_import
from burn_emulator.constants import DEFAULT_DEVICE, DEFAULT_DTYPE, INF_PROFILE, Path
from burn_emulator.datasets.utils import crop_region
from burn_emulator.utils import experiment_dir, resolve_checkpoint


def _drain(pending: list[Future], limit: int) -> None:
    while len(pending) >= limit:
        pending.pop(0).result()


def _compute_crop_region(
    b: int, pdiffs: tuple, bounds: tuple
) -> tuple[int, int, int, int, int, int, int, int]:
    ydiff, xdiff = pdiffs
    ymin, ymax, xmin, xmax = bounds
    y0, y1, x0, x1, y_start, x_start = crop_region(
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
    test_name: str,
    outdir: Path,
) -> None:
    sidx, _ = indxes

    for b in range(pred.shape[0]):
        y0, y1, x0, x1, y_start, x_start, h, w = _compute_crop_region(b, pdiffs, bounds)

        canvas = torch.zeros(shape, dtype=torch.float32)
        canvas[:, y0:y1, x0:x1] = torch.from_numpy(
            pred[b, :, y_start : y_start + h, x_start : x_start + w]
        )

        ignition_number = str(ignitions.iloc[int(sidx[b])]["ignition_number"])
        cbp_burn = str(ignitions.iloc[int(sidx[b])]["cbp_burn"])
        sample_path = outdir / cbp_burn / ignition_number / f"{test_name}.tif"
        sample_path.parent.mkdir(exist_ok=True, parents=True)

        with rasterio.open(sample_path, "w", **profile) as dst:
            dst.write(canvas.numpy())


def test_model(
    test_name: str,
    model: torch.nn.Module,
    test_loader: DataLoader,
    activation: torch.nn.Module,
    out_channels: int,
    num_sims: int,
    outdir: Path,
    max_write_workers: int,
) -> None:
    # how fragile things can be...
    profile = test_loader.dataset.profile | INF_PROFILE
    shape = (out_channels, profile["height"], profile["width"])
    profile.update({"count": out_channels})

    pending: list[Future] = []
    sim_perf_times = []
    sam_perf_times = []  # does not account for partial batches
    drn_perf_times = []

    test_start_time = time.perf_counter()
    with torch.no_grad(), ThreadPoolExecutor(max_workers=max_write_workers) as pool:
        for _ in range(num_sims):
            sim_start_time = time.perf_counter()
            for sample in test_loader:
                X = sample["x"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                W = sample["wind"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                M = sample["mask"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                pdiffs = sample["pdiffs"]
                bounds = sample["bounds"]
                indxes = sample["indxes"]

                sam_start_time = time.perf_counter()
                Mx = M.unsqueeze(1) if X.dim() != M.dim() else M
                pred = (activation(model(X * Mx, W)) * M).to(torch.float32)
                sam_end_time = time.perf_counter()
                sam_perf_times.append(sam_end_time - sam_start_time)

                drn_start_time = time.perf_counter()
                _drain(pending, limit=max_write_workers)
                drn_end_time = time.perf_counter()
                drn_perf_times.append(drn_end_time - drn_start_time)

                pending.append(
                    pool.submit(
                        _write_batch,
                        pred.cpu().numpy(),
                        pdiffs,
                        bounds,
                        indxes,
                        test_loader.dataset.ignitions,
                        shape,
                        profile.copy(),
                        test_name,
                        outdir,
                    )
                )
            sim_end_time = time.perf_counter()
            sim_perf_times.append(sim_end_time - sim_start_time)
        _drain(pending, limit=1)
    test_end_time = time.perf_counter()
    test_perf_time = test_end_time - test_start_time
    tp = {
        "model": test_name,
        "num_batches": len(test_loader),
        "batch_size": test_loader.batch_size,
        "max_memory_alloc": np.round(
            torch.cuda.max_memory_allocated(DEFAULT_DEVICE) / 1024**3, decimals=2
        ),
        "test_perf_time": np.round(test_perf_time, decimals=2),
        "sim_perf_time_mu": np.round(np.mean(sim_perf_times), decimals=2).item(),
        "sam_perf_time_mu": np.round(np.mean(sam_perf_times), decimals=2).item(),
        "drn_perf_time_mu": np.round(np.mean(drn_perf_times), decimals=2).item(),
    }
    df = pd.DataFrame([tp])
    header = not (outdir / "throughput.csv").exists()
    df.to_csv(outdir / "throughput.csv", mode="a", index=False, header=header)


def _run_single_test(
    test_name: str,
    model_name: str,
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
    **kwargs: Any,
) -> tuple[int | None, int | None]:
    model = dynamic_import(model)
    ckpt_path = resolve_checkpoint(model_name, kwargs.get("ckpt_path"))

    ckpt = torch.load(ckpt_path, map_location=DEFAULT_DEVICE)
    if next(iter(ckpt.keys())).startswith("_orig_mod"):
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt)
    model.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    model.eval()
    model = torch.compile(model)

    experiment_path = experiment_dir(model_name)
    outdir = experiment_path / "inference"
    if iteration is not None and scenario is not None:
        spath = scenarios_path / f"iteration_{iteration}" / f"{scenario}_{test_name}"
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

    dataset = dynamic_import(ds_kwargs, {"stats_path": experiment_path / "stats.yaml"})
    test_loader = dynamic_import(dataloader, {"dataset": dataset})
    activation = dynamic_import(activation)

    run_test_name = f"{model_name}_{test_name}"
    with torch.no_grad():
        test_model(
            test_name=run_test_name,
            model=model,
            test_loader=test_loader,
            activation=activation,
            out_channels=out_channels,
            num_sims=num_sims,
            outdir=outdir,
            max_write_workers=max_write_workers,
        )
    return iteration, scenario


def test(
    test_name: str,
    model_name: str,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    max_write_workers: int,
    out_channels: int,
    **kwargs: Any,
) -> None:
    _run_single_test(
        test_name=test_name,
        model_name=model_name,
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        activation=activation,
        max_write_workers=max_write_workers,
        out_channels=out_channels,
    )


def test_iterations(
    test_name: str,
    model_name: str,
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
    **kwargs: Any,
) -> None:
    tasks = [(i, s) for i in range(num_iterations) for s in range(1, num_scenarios + 1)]
    scenarios_path = Path(scenarios_path)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_test,
                test_name,
                model_name,
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
            ): (i, s)
            for i, s in tasks
        }
        for future in as_completed(futures):
            i, s = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"iteration_{i} scenario_{s} failed: {e}")
