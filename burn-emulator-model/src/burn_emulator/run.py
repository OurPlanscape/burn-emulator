import resource
import threading
import time
from collections import Counter
from concurrent.futures import as_completed
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.features import geometry_mask

from burn_emulator.config import dynamic_import
from burn_emulator.constants import BACKENDS, INF_PROFILE, RUN_BACKEND, RUN_DEVICE, RUN_DTYPE, Path
from burn_emulator.datasets.utils import compute_crop_region
from burn_emulator.utils import batched_agg, peak_gpu_gb, resolve_model_checkpoint, timed


class RunCancelled(Exception):
    pass


def _fire_touches(
    fire: torch.Tensor,
    region_mask: torch.Tensor,
    bounds_b: tuple[int, int, int, int],
    diffs_b: tuple[int, int],
) -> bool:
    y0, y1, x0, x1, ys, xs = compute_crop_region(bounds_b, diffs_b)
    fire = fire[ys : ys + (y1 - y0), xs : xs + (x1 - x0)]
    return bool((fire & region_mask[y0:y1, x0:x1]).any())


def _center_component(mask: torch.Tensor) -> torch.Tensor:
    _, h, w = mask.shape
    cy, cx = h // 2, w // 2
    seed = torch.zeros_like(mask)
    # seeded from the 3x3 center
    seed[:, max(cy - 1, 0) : cy + 2, max(cx - 1, 0) : cx + 2] = True
    seed = seed & mask
    if seed.sum() == 0:
        return seed
    
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    for _ in range(max(h, w)*3):
        grown = (torch.conv2d(seed.float().unsqueeze(1), kernel, padding=1).squeeze(1) > 0) & mask
        if grown.sum() == seed.sum():
            break
        seed = grown
    return seed


def timing_report(tag: str, header: str, timings: dict, total: float, rows: dict) -> None:
    timing_rows = "\n".join(f"  {label:<18}: {sec:7.2f}s" for label, sec in timings.items())
    extra_rows = "".join(f"\n  {label:<18}: {value}" for label, value in rows.items())
    print(
        f"[{tag}] timing  {header}\n"
        f"{timing_rows}\n"
        f"  {'total':<18}: {total:7.2f}s"
        f"{extra_rows}",
        flush=True,
    )


def peak_memory_rows() -> dict:
    # ru_maxrss is KiB on Linux
    rows = {
        "peak cpu mem": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2):7.2f}GB"
    }
    peak_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024**2)
    if peak_child > 0:
        rows["peak worker mem"] = f"{peak_child:7.2f}GB"
    peak_gpu = peak_gpu_gb()
    if peak_gpu is not None:
        rows["peak gpu mem"] = f"{peak_gpu:7.2f}GB"
    return rows


def pt_model_name(model_name: str) -> str:
    # model_name is {VARLOC}_{architecture}_{data_version}, see config.resolve_model_name
    varloc, _, data_version = model_name.split("_", 2)
    return f"{varloc}_pt_{data_version}"


def pt_out_path(out_path: str | Path | None, experiment_dir: Path, model_name: str) -> str | Path:
    pt_name = pt_model_name(model_name)
    if out_path is None:
        return experiment_dir / f"{pt_name}.tif"
    # string ops keep gs:// URIs intact
    out_path = str(out_path)
    head, _, name = out_path.rpartition("/")
    name = name.replace(model_name, pt_name) if model_name in name else f"{pt_name}_{name}"
    return f"{head}/{name}" if head else name


def _run_pt(
    ds: Any,
    region: Any,
    out_path: str | Path,
    pt_workers: int | None,
    timings: dict | None,
    debug: bool,
    cancel: threading.Event | None,
) -> dict:
    # deferred: pyretechnics is an optional dep (pyproject.toml [data] extra)
    from burn_emulator import pt

    profile = ds.profile | INF_PROFILE
    shape = (profile["height"], profile["width"])
    n_change = 3  # 0 no change | 1 non-crown -> crown | 2 crown -> non-crown
    profile.update({"count": n_change})

    with timed(timings, "pt inputs"):
        baseline = pt.read_raw_inputs(ds.fuels_paths["baseline"], ds.topo_path, ds.profile)
        treatment = pt.read_raw_inputs(ds.fuels_paths["treatment"], ds.topo_path, ds.profile)
        treatment = pt.collate_treatment(baseline, treatment, region, ds.profile)
        state = {
            "inputs": {"baseline": pt.pt_inputs(baseline), "treatment": pt.pt_inputs(treatment)},
            "keep_mask": geometry_mask(
                [region], out_shape=shape, transform=profile["transform"], invert=True
            ),
        }

    to_crown = np.zeros(shape, dtype=np.float32)
    from_crown = np.zeros(shape, dtype=np.float32)
    n_kept, sim_runtime, stops = 0, 0.0, Counter()
    points = zip(ds.ignitions["row"], ds.ignitions["col"], strict=True)
    pt_workers = pt.resolve_workers(shape[0] * shape[1], pt_workers)
    with pt.pool(state, pt_workers) as pool:
        futures = [
            pool.submit(
                pt.change_task,
                (int(row), int(col)),
                {role: float(ds.wind_angles[role].iloc[i]) for role in state["inputs"]},
            )
            for i, (row, col) in enumerate(points)
        ]
        completed = as_completed(futures)
        while True:
            with timed(timings, "pt simulate"):
                future = next(completed, None)
                result = None if future is None else future.result()
            if result is None:
                break
            if cancel is not None and cancel.is_set():
                for f in futures:
                    f.cancel()
                raise RunCancelled(f"cancelled after {n_kept}/{len(futures)} ignitions")
            sim_runtime += result["runtime"]
            stops.update(result["stop_conditions"])
            # fires that never reach the region are dropped
            if not result["touches"]:
                continue
            with timed(timings, "aggregate"):
                y0, x0 = result["origin"]
                h, w = result["to_crown"].shape
                to_crown[y0 : y0 + h, x0 : x0 + w] += result["to_crown"]
                from_crown[y0 : y0 + h, x0 : x0 + w] += result["from_crown"]
            n_kept += 1

    # untouched pixels count as no change
    agg = np.stack([n_kept - to_crown - from_crown, to_crown, from_crown])
    agg /= max(n_kept, 1)

    if debug:
        print(f"[run] kept {n_kept}/{len(futures)} fires touching the region", flush=True)

    with timed(timings, "write"), rasterio.open(out_path, "w", **profile) as dst:
        dst.write(agg)

    n_sims = 2 * len(futures)
    return {
        "n_kept": n_kept,
        "rows": {
            "workers": pt_workers,
            "sim cpu": f"{sim_runtime:7.2f}s",
            "sim mean": f"{1000 * sim_runtime / max(n_sims, 1):7.2f}ms ({n_sims} sims)",
            **{f"stop: {k}": v for k, v in stops.items()},
        },
    }


def _run_emulator(
    ds: Any,
    loader: Any,
    region: Any,
    model: dict,
    activation: dict,
    experiment_dir: Path,
    ckpt_path: str | None,
    out_path: str | Path,
    timings: dict | None,
    debug: bool,
    cancel: threading.Event | None,
) -> dict:
    n_ignitions = len(ds)
    ckpt_path = resolve_model_checkpoint(experiment_dir, ckpt_path)
    model = dynamic_import(model)
    activation = dynamic_import(activation)
    if debug:
        print(f"[run] loading ckpt_path: {ckpt_path}", flush=True)

    with timed(timings, "model load"):
        ckpt = torch.load(ckpt_path, map_location=RUN_DEVICE, weights_only=True)
        if next(iter(ckpt.keys())).startswith("_orig_mod"):
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        model.to(RUN_DEVICE, dtype=RUN_DTYPE)
        model.eval()

    profile = ds.profile | INF_PROFILE
    shape = (profile["height"], profile["width"])
    n_change = 3  # 0 no change | 1 non-crown -> crown | 2 crown -> non-crown
    profile.update({"count": n_change})

    keep_mask = torch.from_numpy(
        geometry_mask([region], out_shape=shape, transform=profile["transform"], invert=True)
    ).to(RUN_DEVICE)

    # fp32 accumulator
    agg = torch.zeros([n_change, *shape], dtype=torch.float32, device=RUN_DEVICE)
    n_kept = 0
    with torch.no_grad():
        for sample in loader:
            if cancel is not None and cancel.is_set():
                raise RunCancelled(f"cancelled after {n_kept}/{n_ignitions} ignitions")

            n = sample["x"].shape[0]
            ydiff, xdiff = sample["pdiffs"]
            ymin, ymax, xmin, xmax = sample["bounds"]

            with timed(timings, "data_movement"):
                x = sample["x"]  # (B, 2, C, H, W): layer 0 baseline, layer 1 collated treatment
                X = torch.cat([x[:, 0], x[:, 1]]).to(RUN_DEVICE, dtype=RUN_DTYPE)
                W = torch.cat([sample["wind"], sample["wind"]]).to(RUN_DEVICE, dtype=RUN_DTYPE)
                M = torch.cat([sample["mask"], sample["mask"]]).to(RUN_DEVICE, dtype=RUN_DTYPE)
            
            with timed(timings, "model forward"):
                pred = activation(model(X, W)) * M
                pred = pred.argmax(dim=1)

            # keeps only the component connected to the window center
            with timed(timings, "center component"):
                burned = _center_component(pred != 0)
                burned = burned[:n] | burned[n:]

            # fire_type: 0 unburned | 1 surface | 2 passive crown | 3 active crown
            baseline_crowned = pred[:n] >= 2
            treatment_crowned = pred[n:] >= 2

            # 0 (no change) | 1 (non-crown to crown) | 2 (crown to non-crown)
            to_crown = (~baseline_crowned & treatment_crowned)
            from_crown = (baseline_crowned & ~treatment_crowned)
            no_change = ~(to_crown | from_crown)

            change = torch.stack([no_change, to_crown, from_crown], dim=1).float()

            # fires that never reach the region are dropped
            with timed(timings, "fire touches"):
                keep = [
                    _fire_touches(
                        burned[b], keep_mask,
                        (ymin[b], ymax[b], xmin[b], xmax[b]), (ydiff[b], xdiff[b]),
                    )
                    for b in range(n)
                ]
            if not any(keep):
                continue
            sel = torch.tensor(keep).nonzero(as_tuple=True)[0]
            change = change[sel]
            ydiff, xdiff = ydiff[sel], xdiff[sel]
            ymin, ymax, xmin, xmax = ymin[sel], ymax[sel], xmin[sel], xmax[sel]

            # untouched pixels count as no change
            with timed(timings, "aggregate"):
                agg += batched_agg(
                    change, (ydiff, xdiff), (ymin, ymax, xmin, xmax), shape, bg_channel=0
                )
            n_kept += change.shape[0]
        agg /= max(n_kept, 1)

    if debug:
        print(f"[run] kept {n_kept}/{n_ignitions} fires touching the region", flush=True)

    with timed(timings, "write"):
        agg = agg.cpu().numpy()
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(agg)

    return {"n_kept": n_kept, "rows": {}}


def run(
    model_name: str,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    experiment_dir: str | Path,
    ckpt_path: str | None = None,
    out_path: str | Path | None = None,
    debug: bool = False,
    cancel: threading.Event | None = None,
    backend: str = RUN_BACKEND,
    pt_workers: int | None = None,
    **kwargs: Any,
) -> dict:
    backend = backend.upper()
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} must be one of {BACKENDS}")

    timings = {} if debug else None
    t_start = time.perf_counter() if debug else None

    experiment_dir = Path(experiment_dir)

    base_init = dataset.setdefault("init_args", {})
    base_init.setdefault("ignitions_path", None)  # sampled from treatment_area
    base_init.setdefault("stats_path", experiment_dir / "stats.yaml")
    region = base_init.get("treatment_area")
    assert region is not None, "run needs dataset.init_args.treatment_area"
    assert len(base_init.get("fuels_paths", [])) == 2, "run needs 2 fuels_paths"

    with timed(timings, "dataset caching"):
        ds = dynamic_import(dataset)
        loader = dynamic_import(dataloader, {"dataset": ds})

    n_ignitions = len(ds)

    if backend == "PT":
        out_name = pt_model_name(model_name)
        out_path = pt_out_path(out_path, experiment_dir, model_name)
        label = "pyretechnics"
    else:
        out_name = model_name
        out_path = out_path or experiment_dir / f"{model_name}.tif"
        label = f"device={RUN_DEVICE}"
    if debug:
        print(f"[run] {label}: {n_ignitions} ignitions -> {out_path}", flush=True)

    if backend == "PT":
        stats = _run_pt(ds, region, out_path, pt_workers, timings, debug, cancel)
    else:
        stats = _run_emulator(
            ds, loader, region, model, activation, experiment_dir, ckpt_path, out_path,
            timings, debug, cancel,
        )

    result = {"model_name": out_name, "out_path": out_path, "timings": None}
    if debug:
        total = time.perf_counter() - t_start
        timings["other"] = max(total - sum(timings.values()), 0.0)
        timing_report(
            "run",
            f"{label}  ({n_ignitions} ignitions, {stats['n_kept']} kept)",
            timings,
            total,
            {**stats["rows"], **peak_memory_rows()},
        )
        result["timings"] = {**timings, "total": total}

    return result
