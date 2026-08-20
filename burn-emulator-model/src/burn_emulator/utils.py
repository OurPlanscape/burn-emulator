import heapq
import re

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
from rasterio.windows import Window
from shapely.geometry import Polygon
from torch.optim import Optimizer

from burn_emulator.constants import (
    DEFAULT_DTYPE,
    FBFM_OH_MAP,
    INPUT_KEYS,
    NO_DATA,
    USE_CLOUD_PATHS,
    Path,
)

CKPT_META_RE = re.compile(r"epoch-(\d+)_step-(\d+)")


def to_flow(aspect_raw: torch.Tensor, slope_deg: torch.Tensor):
    missing_mask = (aspect_raw < 0) | (slope_deg < 0)
    flat_mask = slope_deg <= 0
    zero_mask = missing_mask | flat_mask

    aspect_deg = aspect_raw.clamp(0, 255) * (360.0 / 256.0)
    slope_rad = torch.deg2rad(slope_deg.clamp(0.0, 47.0))
    aspect_rad = torch.deg2rad(aspect_deg)
    magnitude = torch.sin(slope_rad)
    flow_x = magnitude * torch.sin(aspect_rad)
    flow_y = -magnitude * torch.cos(aspect_rad)
    flow_x = flow_x.masked_fill(zero_mask, 0.0)
    flow_y = flow_y.masked_fill(zero_mask, 0.0)
    return flow_x, flow_y


def cache_bfuels_inputs(
    fuels_paths: list[Path],
    topo_path: list[Path],
    stats_path: str,
    flow: bool = True,
    window_geo: Polygon = None,
) -> tuple[dict, dict, dict]:
    inputs = {}
    topos = {}
    masks = {}

    for fuels_path in fuels_paths:
        fkey = fuels_path.stem

        inputs[fkey] = {}
        # TODO: convert to zarr/icechunk inputs
        fuels_files = {file.stem.rsplit("_", 1)[1]: file for file in fuels_path.glob("*.tif")}
        for name in INPUT_KEYS:
            file = fuels_files.get(name)
            if file is None:
                continue
            with rasterio.open(file) as src:
                window = (
                    Window.from_bounds(*window_geo.bounds, transform=src.transform)
                    if window_geo
                    else None
                )
                dat = torch.tensor(src.read(window=window))
                if name == "fbfm":
                    profile = src.profile
                    masks[fkey] = dat != src.nodata
                    for k, v in FBFM_OH_MAP.items():
                        dat[dat == k] = v
                    dat = F.one_hot(
                        dat.long(), num_classes=len(np.unique(list(FBFM_OH_MAP.values())))
                    )
                    dat = dat.squeeze(0).permute(2, 0, 1)
                    dat = dat[1:].to(DEFAULT_DTYPE)  # remove the 0th channel (no fuel data)
                else:
                    dat = dat.to(DEFAULT_DTYPE)
                    dat[dat == src.nodata] = np.nan
                    dat[dat < 0] = 0
            inputs[fkey][name] = dat
    stats_data = stats_path.exists()
    if stats_data:
        with stats_path.open() as f:
            stats = yaml.safe_load(f)
    else:
        stats = {}
    for key in INPUT_KEYS:
        if key != "fbfm":
            arrs = []
            if stats_data:
                mean = stats[key]["mean"]
                stdv = stats[key]["stdv"]
            else:
                for fuels_path in fuels_paths:
                    fkey = fuels_path.stem
                    arrs.append(inputs[fkey][key])
                arrs = torch.concat(arrs)
                mean = torch.nanmean(arrs).item()
                stdv = torch.sqrt(torch.nanmean((arrs - mean) ** 2)).item()
                stats[key] = {"mean": mean, "stdv": stdv}

            for fuels_path in fuels_paths:
                fkey = fuels_path.stem
                inputs[fkey][key] = (inputs[fkey][key] - mean) / stdv
                inputs[fkey][key][torch.isnan(inputs[fkey][key])] = NO_DATA
    if not stats_data and not USE_CLOUD_PATHS:
        with stats_path.open("w") as file:
            yaml.dump(stats, file, sort_keys=False)

    with rasterio.open(topo_path / "aspect.tif") as src:
        aspect = torch.tensor(src.read(window=window)).to(DEFAULT_DTYPE)
    with rasterio.open(topo_path / "slope_degrees.tif") as src:
        slope = torch.tensor(src.read(window=window)).to(DEFAULT_DTYPE)

    if flow:
        flow_x, flow_y = to_flow(aspect, slope)
        topos["flow_x"] = flow_x
        topos["flow_y"] = flow_y
    else:
        topos["aspect"] = aspect
        topos["slope"] = slope

    return inputs, topos, masks, profile


def batched_agg(pred, diffs, slices, out_shape):
    device = pred.device
    B, C, Hp, Wp = pred.shape
    H, W = out_shape

    ydiff, xdiff = diffs
    ymin, ymax, xmin, xmax = slices

    ydiff = ydiff.to(device).long()
    xdiff = xdiff.to(device).long()
    y0, y1 = ymin.to(device).long(), ymax.to(device).long()
    x0, x1 = xmin.to(device).long(), xmax.to(device).long()

    y_start = torch.where((ydiff > 0) & (y0 == 0), ydiff, torch.zeros_like(ydiff))
    x_start = torch.where((xdiff > 0) & (x0 == 0), xdiff, torch.zeros_like(xdiff))

    y = torch.arange(H, device=device).view(1, H, 1)  # (1, H, 1)
    x = torch.arange(W, device=device).view(1, 1, W)  # (1, 1, W)

    y0_ = y0.view(B, 1, 1)
    y1_ = y1.view(B, 1, 1)
    x0_ = x0.view(B, 1, 1)
    x1_ = x1.view(B, 1, 1)
    y_start_ = y_start.view(B, 1, 1)
    x_start_ = x_start.view(B, 1, 1)

    mask = (y >= y0_) & (y < y1_) & (x >= x0_) & (x < x1_)
    src_y = (y - y0_ + y_start_).clamp(0, Hp - 1)
    src_x = (x - x0_ + x_start_).clamp(0, Wp - 1)

    src_y = src_y.unsqueeze(1).expand(B, C, H, W)
    src_x = src_x.unsqueeze(1).expand(B, C, H, W)
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, C, H, W)
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(B, C, H, W)

    gathered = pred[b_idx, c_idx, src_y, src_x]  # (B, C, H, W)
    canvas = gathered * mask.unsqueeze(1).to(gathered.dtype)

    return torch.sum(canvas, dim=0)


def circle_mask(window_size: int):
    h = w = window_size
    # assume an odd window size for
    # radius 1 less than window size // 2
    cy, cx = (h - 1) / 2, (w - 1) / 2
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist < min(cx, cy)


def save_checkpoint(
    model: torch.nn.Module,
    tag: str,
    epoch: int,
    step: int,
    loss: float,
    heap: list,
    out_path: Path,
    optimizer: Optimizer | None = None,
) -> None:
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = f"{tag}_loss-{loss:.4f}_epoch-{epoch:04d}_step-{step:06d}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        torch.save(model._orig_mod.state_dict(), ckpt_path)
    else:
        torch.save(model.state_dict(), ckpt_path)

    if optimizer is not None:
        optim_dir = ckpt_dir / "optim"
        optim_dir.mkdir(exist_ok=True)
        torch.save(optimizer.state_dict(), optim_dir / ckpt_path.name)

    heapq.heappush(heap, (-loss, epoch, step, ckpt_path.stem))

    if len(heap) > 3:
        _, _, _, worst_path = heapq.heappop(heap)
        (ckpt_dir / f"{worst_path}.pt").unlink()
        worst_optim_path = ckpt_dir / "optim" / f"{worst_path}.pt"
        if worst_optim_path.exists():
            worst_optim_path.unlink()


def parse_checkpoint_meta(ckpt_path: Path) -> tuple[int, int] | None:
    match = CKPT_META_RE.search(Path(ckpt_path).stem)
    return (int(match.group(1)), int(match.group(2))) if match else None


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = [p for p in ckpt_dir.glob("*.pt") if parse_checkpoint_meta(p) is not None]
    if not ckpts:
        return None
    return max(ckpts, key=parse_checkpoint_meta)


def optimizer_checkpoint_path(ckpt_path: Path) -> Path:
    ckpt_path = Path(ckpt_path)
    return ckpt_path.parent / "optim" / ckpt_path.name
