import heapq
import re
import time

import torch
from torch.optim import Optimizer

from burn_emulator.constants import OUTDIR, RUN_DEVICE, Path
from burn_emulator.datasets.utils import crop_region

_CKPT_META_RE = re.compile(r"epoch-(\d+)_step-(\d+)")
_CKPT_LOSS_RE = re.compile(r"loss-([\d.]+)")


class timed:
    """Accumulate wall time into timings[label]. No-op (and no cuda sync) when timings is None."""

    def __init__(self, timings: dict[str, float] | None, label: str) -> None:
        self.timings = timings
        self.label = label

    def __enter__(self) -> "timed":
        if self.timings is not None:
            if RUN_DEVICE == "cuda":
                torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.timings is None:
            return
        if RUN_DEVICE == "cuda":
            torch.cuda.synchronize()
        self.timings[self.label] = self.timings.get(self.label, 0.0) + (
            time.perf_counter() - self.t0
        )


def peak_gpu_gb() -> float | None:
    dev = RUN_DEVICE.lower()
    try:
        if dev == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        if dev == "xla":
            import torch_xla.core.xla_model as xm

            info = xm.get_memory_info(xm.xla_device())
            peak = info.get("peak_bytes_used", info.get("bytes_used"))
            return None if peak is None else peak / (1024**3)
    except Exception:
        return None
    return None


def to_flow(aspect_raw: torch.Tensor, slope_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def batched_agg(
    pred: torch.Tensor,
    diffs: tuple[torch.Tensor, torch.Tensor],
    slices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    out_shape: tuple[int, int],
    bg_channel: int | None = None,
) -> torch.Tensor:
    B, C, Hp, Wp = pred.shape
    H, W = out_shape

    ydiff, xdiff = diffs
    ymin, ymax, xmin, xmax = slices

    out = torch.zeros(C, H, W, dtype=pred.dtype, device=pred.device)
    if bg_channel is not None:
        # every sample votes into bg_channel everywhere its window doesn't reach
        # NOTE: this is quite fragile if the aggregation later in run.py doesn't fit
        out[bg_channel] += B

    for b in range(B):
        y0, y1, x0, x1, ys, xs = crop_region(
            (ymin[b], ymax[b], xmin[b], xmax[b]), (ydiff[b], xdiff[b])
        )
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            continue
        out[:, y0:y1, x0:x1] += pred[b, :, ys : ys + h, xs : xs + w]
        if bg_channel is not None:
            out[bg_channel, y0:y1, x0:x1] -= 1  # ...except inside its own window

    return out


def circle_mask(window_size: int) -> torch.Tensor:
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
    heap: list[tuple],
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


def parse_checkpoint_meta(ckpt_path: str | Path) -> tuple[int, int] | None:
    match = _CKPT_META_RE.search(Path(ckpt_path).stem)
    return (int(match.group(1)), int(match.group(2))) if match else None


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = [p for p in ckpt_dir.glob("*.pt") if parse_checkpoint_meta(p) is not None]
    if not ckpts:
        return None
    return max(ckpts, key=parse_checkpoint_meta)


def find_best_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    scored = [
        (float(m.group(1)), p)
        for p in ckpt_dir.glob("*.pt")
        if (m := _CKPT_LOSS_RE.search(p.stem))
    ]
    if scored:
        return min(scored)[1]
    return find_latest_checkpoint(ckpt_dir)


def optimizer_checkpoint_path(ckpt_path: str | Path) -> Path:
    ckpt_path = Path(ckpt_path)
    return ckpt_path.parent / "optim" / ckpt_path.name


def experiment_dir(model_name: str, override: str | Path | None = None) -> Path:
    return Path(override) if override else OUTDIR / model_name


def resolve_checkpoint(
    model_name: str,
    ckpt_path: str | Path | None = None,
    training_dir: str | Path | None = None,
) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    ckpt_dir = experiment_dir(model_name, training_dir) / "checkpoints"
    best = find_best_checkpoint(ckpt_dir)
    if best is None:
        raise FileNotFoundError(f"no checkpoint for {model_name} in {ckpt_dir}")
    return best
