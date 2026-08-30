import time
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from burn_emulator.config import dynamic_import
from burn_emulator.constants import DEFAULT_DEVICE, DEFAULT_DTYPE, Path
from burn_emulator.utils import (
    experiment_dir,
    find_latest_checkpoint,
    optimizer_checkpoint_path,
    parse_checkpoint_meta,
    save_checkpoint,
)


def train_model(
    model: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    model_name: str,
    optimizer: Optimizer,
    criterion: nn.Module,
    out_path: Path,
    start_epoch: int = 0,
    start_step: int = 0,
) -> None:
    train_top3 = []
    step = start_step

    model.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    model.train()
    model = torch.compile(model)
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.perf_counter()
        train_loss_acc = 0
        train_loss_avg = None
        log = []
        for sample in train_loader:
            sample_start_time = time.perf_counter()
            X = sample["x"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            Y = sample["y"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            W = sample["wind"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            M = sample["mask"].to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

            # no OOD validation loop in favor of post-training
            # evaluation of approximation accuracy
            optimizer.zero_grad()
            Mx = M.unsqueeze(1) if X.dim() != M.dim() else M
            Y_hat = model(X * Mx, W)
            train_loss = criterion(Y_hat * M, Y * M)
            train_loss.backward()
            optimizer.step()

            step += 1
            train_loss_val = train_loss.item()
            train_loss_acc += train_loss_val
            log.append(
                {
                    "train_loss_avg": None,
                    "train_loss_val": train_loss_val,
                    "epoch": epoch,
                    "step": step,
                    "time": time.perf_counter() - sample_start_time,
                }
            )

        train_loss_avg = train_loss_acc / len(train_loader)

        log.append(
            {
                "train_loss_avg": train_loss_avg,
                "train_loss_val": None,
                "epoch": epoch,
                "step": step,
                "time": time.perf_counter() - epoch_start_time,
            }
        )
        save_checkpoint(
            model,
            f"{model_name}_train",
            epoch,
            step,
            train_loss_avg,
            train_top3,
            out_path,
            optimizer=optimizer,
        )

        header = not (out_path / "train_log.csv").exists()
        pd.DataFrame(log).to_csv(out_path / "train_log.csv", mode="a", index=False, header=header)
    if num_epochs > start_epoch:
        save_checkpoint(
            model,
            f"{model_name}_train",
            epoch,
            step,
            train_loss_avg,
            [],
            out_path,
            optimizer=optimizer,
        )


def train(
    model: dict,
    model_name: str,
    dataset: dict,
    dataloader: dict,
    optimizer: dict,
    criterion: dict,
    num_epochs: int,
    **kwargs: Any,
) -> None:
    experiment_path = experiment_dir(model_name)
    experiment_path.mkdir(exist_ok=True, parents=True)

    model = dynamic_import(model)
    ckpt_path = kwargs.get("ckpt_path")
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(experiment_path / "checkpoints")

    start_epoch, start_step = 0, 0
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=DEFAULT_DEVICE)
        if next(iter(ckpt.keys())).startswith("_orig_mod"):
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        model.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

        meta = parse_checkpoint_meta(ckpt_path)
        if meta is not None:
            start_epoch, start_step = meta[0] + 1, meta[1]

    dataset = dynamic_import(dataset, {"stats_path": experiment_path / "stats.yaml"})
    optimizer = dynamic_import(optimizer, {"params": model.parameters()})
    if ckpt_path is not None:
        optim_ckpt_path = optimizer_checkpoint_path(ckpt_path)
        if optim_ckpt_path.exists():
            optim_state_dict = torch.load(optim_ckpt_path, map_location=DEFAULT_DEVICE)
            optimizer.load_state_dict(optim_state_dict)
    criterion = dynamic_import(criterion)
    train_loader = dynamic_import(dataloader, {"dataset": dataset})

    train_model(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        model_name=model_name,
        optimizer=optimizer,
        criterion=criterion,
        out_path=experiment_path,
        start_epoch=start_epoch,
        start_step=start_step,
    )
