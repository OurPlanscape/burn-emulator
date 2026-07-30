from typing import Any

import rasterio
import torch

from burn_emulator.constants import INF_PROFILE, RUN_DEVICE, RUN_DTYPE, Path
from burn_emulator.utils import batched_agg, dynamic_import


def run(
    run_name: str,
    model_name: str,
    model: dict,
    ckpt_path: str,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    out_path: Path,
    max_write_workers: int,
    **kwargs: Any,
) -> None:
    model = dynamic_import(model)
    activation = dynamic_import(activation)

    dataset = dynamic_import(dataset)
    dataloader = dynamic_import(dataloader, {"dataset": dataset})
    out_path = Path(out_path) / f"{run_name}_{model_name}.tif"

    ckpt = torch.load(ckpt_path, map_location=RUN_DEVICE)
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
        for X, M, diffs, slices, _ in dataloader:
            X = X.to(RUN_DEVICE, dtype=RUN_DTYPE)
            M = M.to(RUN_DEVICE, dtype=RUN_DTYPE)
            pred = activation(model(X)) * M
            agg += batched_agg(pred, diffs, slices, shape)
        agg /= len(dataloader.dataset)

    agg = agg.to_numpy()
    with rasterio.open(out_path, **profile) as dst:
        dst.write(agg)
