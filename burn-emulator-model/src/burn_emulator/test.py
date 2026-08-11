import copy
import time
import numpy as np
import pandas as pd
import rasterio
import torch

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
from typing import Any

from burn_emulator.constants import DEFAULT_DEVICE, DEFAULT_DTYPE, INF_PROFILE, OUTDIR, Path
from burn_emulator.utils import dynamic_import


torch.set_float32_matmul_precision('high')


def _drain(pending: list[Future], limit: int) -> None:
    while len(pending) >= limit:
        pending.pop(0).result()


def _compute_crop_region(b: int, pdiffs: tuple, bounds: tuple) -> tuple[int, int, int, int, int, int, int, int]:
    ydiff, xdiff = pdiffs
    ymin, ymax, xmin, xmax = bounds
    yd, xd = int(ydiff[b]), int(xdiff[b])
    y0, y1 = int(ymin[b]), int(ymax[b])
    x0, x1 = int(xmin[b]), int(xmax[b])
    h, w = y1 - y0, x1 - x0
    y_start = yd if (yd > 0 and y0 == 0) else 0
    x_start = xd if (xd > 0 and x0 == 0) else 0
    return y0, y1, x0, x1, y_start, x_start, h, w


def _accumulate_batch(
    pred: torch.Tensor,
    pdiffs: tuple,
    bounds: tuple,
    n_count: torch.Tensor,
    mean_acc: torch.Tensor,
    M2_acc: torch.Tensor,
    log_prod_acc: torch.Tensor,
    log1m_prod_acc: torch.Tensor,
) -> None:
    for b in range(pred.shape[0]):
        y0, y1, x0, x1, y_start, x_start, h, w = _compute_crop_region(b, pdiffs, bounds)

        x = pred[b, :, y_start:y_start + h, x_start:x_start + w].double()
        roi = (slice(None), slice(y0, y1), slice(x0, x1))

        n_count[roi] += 1
        delta = x - mean_acc[roi]
        mean_acc[roi] += delta / n_count[roi]
        delta2 = x - mean_acc[roi]
        M2_acc[roi] += delta * delta2

        eps = 1e-7
        log_prod_acc[roi] += torch.log(x.clamp(min=eps))
        log1m_prod_acc[roi] += torch.log((1.0 - x).clamp(min=eps))


def _write_monte_carlo(
    n_count: torch.Tensor,
    mean_acc: torch.Tensor,
    M2_acc: torch.Tensor,
    log_prod_acc: torch.Tensor,
    log1m_prod_acc: torch.Tensor,
    mask: torch.Tensor,
    test_name: str,
    outdir: Path,
    profile: dict,
) -> None:
    mean_out = mean_acc.float() * mask
    std_out = (M2_acc / n_count.clamp(min=1)).sqrt().float() * mask
    entropy_out = (
        -(log_prod_acc / n_count.clamp(min=1))
        - (log1m_prod_acc / n_count.clamp(min=1))
    ).float() * mask
    n_out = n_count.float() * mask
    with rasterio.open(outdir / f'{test_name}_entropy.tif', 'w', **profile) as dst:
        dst.write(entropy_out.cpu().numpy())
    with rasterio.open(outdir / f'{test_name}_mean.tif', 'w', **profile) as dst:
        dst.write(mean_out.cpu().numpy())
    with rasterio.open(outdir / f'{test_name}_stdv.tif', 'w', **profile) as dst:
        dst.write(std_out.cpu().numpy())
    with rasterio.open(outdir / f'{test_name}_count.tif', 'w', **profile) as dst:
        dst.write(n_out.cpu().numpy())


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
            pred[b, :, y_start:y_start + h, x_start:x_start + w])

        ignition_number = str(ignitions.iloc[int(sidx[b])]['ignition_number'])
        cbp_burn = str(ignitions.iloc[int(sidx[b])]['cbp_burn'])
        sample_path = outdir / cbp_burn / ignition_number / f'{test_name}.tif'
        sample_path.parent.mkdir(exist_ok=True, parents=True)

        with rasterio.open(sample_path, 'w', **profile) as dst:
            dst.write(canvas.numpy())


def test_model(
    test_name: str,
    model: torch.nn.Module,
    test_loader: DataLoader,
    activation: torch.nn.Module,
    num_sims: int,
    monte_carlo: bool,
    outdir: Path,
    max_write_workers: int,
) -> None:
    # how fragile things can be...
    profile = test_loader.dataset.profile | INF_PROFILE
    count = 4
    shape = (count, profile["height"], profile["width"])
    profile.update({"count": count})

    if monte_carlo:
        odtype = torch.float64
        n_count = torch.zeros([*shape], dtype=odtype, device=DEFAULT_DEVICE)
        mean_acc = torch.zeros([*shape], dtype=odtype, device=DEFAULT_DEVICE)
        M2_acc = torch.zeros([*shape], dtype=odtype, device=DEFAULT_DEVICE)
        log_prod_acc = torch.zeros([*shape], dtype=odtype, device=DEFAULT_DEVICE)
        log1m_prod_acc = torch.zeros([*shape], dtype=odtype, device=DEFAULT_DEVICE)
    else:
        odtype = torch.float32

    pending: list[Future] = []
    sim_perf_times = []
    sam_perf_times = [] # does not account for partial batches
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
                pred = (activation(model(X, W)) * M).to(odtype)
                sam_end_time = time.perf_counter()
                sam_perf_times.append(sam_end_time-sam_start_time)
                
                if monte_carlo:
                    _accumulate_batch(
                        pred, pdiffs, bounds,
                        n_count, mean_acc, M2_acc,
                        log_prod_acc, log1m_prod_acc,
                    )
                else:
                    drn_start_time = time.perf_counter()
                    _drain(pending, limit=max_write_workers)
                    drn_end_time = time.perf_counter()
                    drn_perf_times.append(drn_end_time-drn_start_time)
                    
                    pending.append(pool.submit(
                        _write_batch,
                        pred.cpu().numpy(),
                        pdiffs, bounds, indxes,
                        test_loader.dataset.ignitions,
                        shape, profile.copy(),
                        test_name, outdir,
                    ))
            sim_end_time = time.perf_counter()
            sim_perf_times.append(sim_end_time-sim_start_time)
        _drain(pending, limit=1)
    if monte_carlo:
        mask = test_loader.dataset.masks.values().__next__().to(DEFAULT_DEVICE)
        _write_monte_carlo(
            n_count, mean_acc, M2_acc,
            log_prod_acc, log1m_prod_acc,
            mask, test_name, outdir, profile,
        )
    test_end_time = time.perf_counter()
    test_perf_time = test_end_time-test_start_time
    tp = {"model": test_name,
          "num_batches": len(test_loader),
          "batch_size": test_loader.batch_size,
          "max_memory_alloc": np.round(torch.cuda.max_memory_allocated(DEFAULT_DEVICE) / 
                                       1024**3, decimals=2),
          "test_perf_time": np.round(test_perf_time, decimals=2),
          "sim_perf_time_mu": np.round(np.mean(sim_perf_times), decimals=2).item(),
          "sam_perf_time_mu": np.round(np.mean(sam_perf_times), decimals=2).item(),
          "drn_perf_time_mu": np.round(np.mean(drn_perf_times), decimals=2).item()}
    df = pd.DataFrame([tp])
    header = not (outdir / 'throughput.csv').exists()
    df.to_csv(outdir / 'throughput.csv', mode='a', index=False, header=header)


def _run_single_test(test_name: str,
                     model_name: str, 
                     model: dict,
                     dataset: dict,
                     dataloader: dict,
                     activation: dict,
                     iteration: int | None = None,
                     scenario: int | None = None,
                     scenario_path: Path | None = None,
                     num_sims: int=1,
                     monte_carlo: bool=False,
                     max_write_workers: int=4,
                     **kwargs: Any):
    model = dynamic_import(model)
    ckpt_path = kwargs.get("ckpt_path")
    if ckpt_path is None:
        ckpt_dir = OUTDIR / model_name / "checkpoints"
        ckpt_path = Path(sorted(ckpt_dir.glob("*.pt"))[0])
    else:
        ckpt_path = Path(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEFAULT_DEVICE)
    if next(iter(ckpt.keys())).startswith("_orig_mod"):
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    
    model.load_state_dict(ckpt)
    model.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    model.eval()
    model = torch.compile(model)

    outdir = OUTDIR / "inference"
    if iteration is not None and scenario is not None:
        spath = scenarios_path / f"iteration_{iteration}" / f"{scenario}_{test_name}"
        fpath = dataset.get("init_args", {}).get("fuels_paths")
        outdir /= f"iteration_{iteration}"
        init_args = {
            "ignitions_path": spath / f"{scenario}_ignitions_locations.csv",
            "fuels_paths": [spath] if fpath is None else fpath,
        }
        ds_kwargs = copy.deepcopy(dataset)
        ds_kwargs["init_args"] = {**ds_kwargs.get("init_args", {}), **init_args}
    else:
        ds_kwargs = dataset

    dataset = dynamic_import(ds_kwargs)
    test_loader = dynamic_import(dataloader, {"dataset": dataset})
    activation = dynamic_import(activation)
    
    run_test_name = f"{model_name}_{test_name}"
    with torch.no_grad():
        test_model(
            model=model,
            test_loader=test_loader,
            activation=activation,
            test_name=run_test_name,
            num_sims=num_sims,
            monte_carlo=monte_carlo,
            outdir=outdir,
            max_write_workers=max_write_workers
        )
    return iteration, scenario


def test(test_name: str,
         model_name: str, 
         model: dict,
         dataset: dict,
         dataloader: dict,
         activation: dict,
         max_write_workers: int,
         **kwargs: Any) -> None:
    _run_single_test(test_name=test_name, 
                     model_name=model_name,
                     model=model,
                     dataset=dataset,
                     dataloader=dataloader,
                     activation=activation,
                     max_write_workers=max_write_workers)


def test_iterations(test_name: str,
                    model_name: str, 
                    model: dict,
                    dataset: dict,
                    dataloader: dict,
                    activation: dict,
                    num_iterations: int, 
                    num_scenarios: int,
                    scenarios_path: str,
                    max_workers: int=4,
                    num_sims: int=1,
                    monte_carlo: bool=False,
                    max_write_workers: int=4,
                    **kwargs: Any) -> None:
    tasks = [(i, s) for i in range(num_iterations) for s in range(1, num_scenarios+1)]
    scenarios_path = Path(scenarios_path)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_test,test_name, model_name, model, dataset,
                dataloader, activation, i, s, num_sims, monte_carlo,
                scenarios_path, max_write_workers
            ): (i, s)
            for i, s in tasks
        }
        for future in as_completed(futures):
            i, s = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"iteration_{i} scenario_{s} failed: {e}")