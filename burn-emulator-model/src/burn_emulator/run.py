from typing import Any

import rasterio
import torch

from burn_emulator.config import dynamic_import
from burn_emulator.constants import INF_PROFILE, RUN_DEVICE, RUN_DTYPE, Path
from burn_emulator.utils import batched_agg


def run(
    run_name: str,
    model_name: str,
    model: dict,
    ckpt_path: str,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    out_path: Path,
    **kwargs: Any,
) -> None:
    model = dynamic_import(model)
    activation = dynamic_import(activation)

    dataset = dynamic_import(dataset)
    dataloader = dynamic_import(dataloader, {"dataset": dataset})
    out_path = Path(out_path) / f"{run_name}_{model_name}.tif"

    ckpt = torch.load(ckpt_path, map_location=RUN_DEVICE)
    if next(iter(ckpt.keys())).startswith("_orig_mod"):
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.to(RUN_DEVICE, dtype=RUN_DTYPE)
    model.eval()

    profile = dataloader.dataset.profile | INF_PROFILE
    bts = dataloader.dataset.burn_times
    count = len(bts) if bts else 1
    shape = (profile["height"], profile["width"])
    profile.update({"count": count})

    agg = torch.zeros([count, *shape], dtype=RUN_DTYPE, device=RUN_DEVICE)
    with torch.no_grad():
        for sample in dataloader:
            X = sample["x"].to(RUN_DEVICE, dtype=RUN_DTYPE)
            W = sample["wind"].to(RUN_DEVICE, dtype=RUN_DTYPE)
            M = sample["mask"].to(RUN_DEVICE, dtype=RUN_DTYPE)
            diffs = sample["pdiffs"]
            slices = sample["bounds"]
            Mx = M.unsqueeze(1) if X.dim() != M.dim() else M
            pred = activation(model(X * Mx, W)) * M
            # TODO: add masking
            agg += batched_agg(pred, diffs, slices, shape)
        agg /= len(dataloader.dataset)

    agg = agg.to_numpy()

    # NOTE: precomputed CBP should be already on disk somwhere
    # discuss whether change should be computed here or elsewhere
    with rasterio.open(out_path, **profile) as dst:
        dst.write(agg)
