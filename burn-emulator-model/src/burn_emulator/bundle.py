import shutil
from pathlib import Path
from typing import Any

import yaml

from burn_emulator.utils import experiment_dir, resolve_checkpoint

# dataset.init_args keys the runner / deployment fills in - never bundle them.
_RUNTIME_DATASET_KEYS = (
    "treatment_area",
    "fuels_paths",
    "topo_path",
    "ignitions_path",
    "burn_paths",
    "wind_ang_paths",
)


def bundle(
    dest: str | Path,
    model_name: str,
    model: dict,
    dataset: dict,
    dataloader: dict,
    activation: dict,
    ckpt_path: str | None = None,
    out_path: str | Path | None = None,
    **kwargs: Any,
) -> None:
    training_dir = experiment_dir(model_name, out_path)
    ckpt = resolve_checkpoint(model_name, ckpt_path, out_path)
    stats = _resolve_stats(dataset, training_dir)

    dst = Path(dest)
    dst.mkdir(parents=True, exist_ok=True)

    # just to clean it incase of mis entry
    init = dict(dataset.get("init_args") or {})
    for key in _RUNTIME_DATASET_KEYS:
        init.pop(key, None)
    init["stats_path"] = "stat.yaml"

    shutil.copy2(ckpt, dst / "model.pt")
    shutil.copy2(stats, dst / "stat.yaml")

    config = {
        "model_name": model_name,
        "ckpt_path": "model.pt",
        "model": model,
        "activation": activation,
        "dataset": {**dataset, "init_args": init},
        "dataloader": dataloader,
    }
    (dst / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"[bundle] {model_name} ({ckpt.name}) -> {dst}")
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dst)}")


def _resolve_stats(dataset: dict, training_dir: Path) -> Path:
    cfg = (dataset.get("init_args") or {}).get("stats_path")
    if cfg and Path(cfg).is_file():
        return Path(cfg)
    for name in ("stat.yaml", "stats.yaml"):
        if (training_dir / name).is_file():
            return training_dir / name
    raise ValueError(f"no stats file: expected {training_dir / 'stat.yaml'}")
