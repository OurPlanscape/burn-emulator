import resource
import threading
import time
from typing import Any

import rasterio
import torch
from rasterio.features import geometry_mask

from burn_emulator.config import dynamic_import
from burn_emulator.constants import INF_PROFILE, RUN_DEVICE, RUN_DTYPE, Path
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
    seed = torch.zeros_like(mask)
    seed[:, h // 2, w // 2] = True
    seed = seed & mask  # center not burned -> empty component
    if seed.sum() == 0:
        return seed
    
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    # capped at the window diameter*3 as buffer
    # but very likely won't reach that kind of shape
    for _ in range(max(h, w)*3):
        grown = (torch.conv2d(seed.float().unsqueeze(1), kernel, padding=1).squeeze(1) > 0) & mask
        if grown.sum() == seed.sum():
            break
        seed = grown
    return seed


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
    **kwargs: Any,
) -> dict | None:
    timings = {} if debug else None
    t_start = time.perf_counter() if debug else None

    experiment_dir = Path(experiment_dir)
    ckpt_path = resolve_model_checkpoint(experiment_dir, ckpt_path)

    model = dynamic_import(model)
    activation = dynamic_import(activation)

    base_init = dataset.setdefault("init_args", {})
    base_init.setdefault("ignitions_path", None)  # sampled from treatment_area
    base_init.setdefault("stats_path", experiment_dir / "stats.yaml")
    region = base_init.get("treatment_area")
    assert region is not None, "run needs dataset.init_args.treatment_area"
    assert len(base_init.get("fuels_paths", [])) == 2, "run needs 2 fuels_paths"

    with timed(timings, "dataset caching"):
        ds = dynamic_import(dataset)
        loader = dynamic_import(dataloader, {"dataset": ds})

    n_ignitions = len(loader.dataset)
    if debug:
        print(f"[run] loading ckpt_path: {ckpt_path}", flush=True)
        print(f"[run] {model_name}: {n_ignitions} ignitions", flush=True)

    # out_path may be a gs:// URI - rasterio writes it through GDAL's /vsigs/
    if out_path is None:
        out_path = experiment_dir / f"{model_name}_run.tif"

    with timed(timings, "model load"):
        ckpt = torch.load(ckpt_path, map_location=RUN_DEVICE, weights_only=True)
        if next(iter(ckpt.keys())).startswith("_orig_mod"):
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        model.to(RUN_DEVICE, dtype=RUN_DTYPE)
        model.eval()

    profile = loader.dataset.profile | INF_PROFILE
    shape = (profile["height"], profile["width"])
    n_change = 3  # 0 no change | 1 non-crown -> crown | 2 crown -> non-crown
    profile.update({"count": n_change})

    # raster of the region; a fire is kept when its predicted burn overlaps it
    keep_mask = torch.from_numpy(
        geometry_mask([region], out_shape=shape, transform=profile["transform"], invert=True)
    ).to(RUN_DEVICE)

    # accumulate in fp32: RUN_DTYPE (bf16) loses 1/len increments once agg approaches 1
    agg = torch.zeros([n_change, *shape], dtype=torch.float32, device=RUN_DEVICE)
    n_kept = 0
    with torch.no_grad():
        for sample in loader:
            # for use in the runner when the api disconnects from the job
            # NOTE: probably a good idea to not run 'cancelled' jobs
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

            # restrict to the fire spreading from the window center;
            # removing i.e disconnected blobs from NN outputs
            with timed(timings, "center component"):
                burned = _center_component(pred != 0)
                # or operator since extention is still a signal to be captured
                burned = burned[:n] | burned[n:]

            # fire_type classes: 0 unburned | 1 surface | 2 passive crown | 3 active crown
            # crowned (passive or active) is class >= 2
            baseline_crowned = pred[:n] >= 2
            treatment_crowned = pred[n:] >= 2

            # 0 (no change) | 1 (non-crown to crown) | 2 (crown to non-crown)
            to_crown = (~baseline_crowned & treatment_crowned)
            from_crown = (baseline_crowned & ~treatment_crowned)
            no_change = ~(to_crown | from_crown)

            change = torch.stack([no_change, to_crown, from_crown], dim=1).float()

            # keep a fire when its central burn reaches the region
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

            # bg_channel=0: pixels a window never reaches count as "no change"
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

    if debug:
        total = time.perf_counter() - t_start
        timings["other"] = max(total - sum(timings.values()), 0.0)
        # ru_maxrss is this process's peak resident set size (Linux reports KiB)
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        peak_gpu = peak_gpu_gb()
        rows = "\n".join(f"  {label:<18}: {sec:7.2f}s" for label, sec in timings.items())
        gpu_row = "" if peak_gpu is None else f"\n  {'peak gpu mem':<18}: {peak_gpu:7.2f}GB"
        print(
            f"[run] timing  device={RUN_DEVICE}  ({n_ignitions} ignitions, {n_kept} kept)\n"
            f"{rows}\n"
            f"  {'total':<18}: {total:7.2f}s\n"
            f"  {'peak cpu mem':<18}: {peak_rss_gb:7.2f}GB"
            f"{gpu_row}",
            flush=True,
        )
        return {**timings, "total": total}

    return None
