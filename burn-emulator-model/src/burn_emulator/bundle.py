import json
import shutil
from typing import Any

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from burn_emulator import provenance
from burn_emulator.constants import BUNDLE_DIR, CONFIG_DIR, WIND_DIRECTIONS, Path
from burn_emulator.utils import resolve_model_checkpoint

# dataset.init_args keys the runner / deployment fills in - never bundle them.
_RUNTIME_DATASET_KEYS = (
    "treatment_area",
    "treatment_area_crs",
    "fuels_paths",
    "topo_path",
    "ignitions_path",
    "burn_paths",
    "wind_ang_paths",
)


def bundle(configs: DictConfig, ckpt_path: str | None = None, **kwargs: Any) -> None:
    varloc = OmegaConf.select(configs, "varloc")
    architecture = OmegaConf.select(configs, "architecture")

    model_config = OmegaConf.load(CONFIG_DIR / architecture / "model.yaml")
    configs = OmegaConf.merge(model_config, configs)
    OmegaConf.update(configs, "dataset.init_args.wind_range", _wind_range(varloc))
    config = OmegaConf.to_container(configs, resolve=True)

    model_name = config["model_name"]
    experiment_dir = Path(config["experiment_dir"])
    ckpt = resolve_model_checkpoint(experiment_dir, ckpt_path)
    stats = _resolve_stats(config["dataset"], experiment_dir)
    fbfm_map = _resolve_fbfm_map(config["dataset"])

    dst = BUNDLE_DIR / model_name
    dst.mkdir(parents=True, exist_ok=True)

    init = {
        k: v
        for k, v in (config["dataset"].get("init_args") or {}).items()
        if k not in _RUNTIME_DATASET_KEYS
    }
    init["stats_path"] = "stats.yaml"
    init["fbfm_map_path"] = fbfm_map.name

    shutil.copy2(ckpt, dst / "model.pt")
    shutil.copy2(stats, dst / "stats.yaml")
    shutil.copy2(fbfm_map, dst / fbfm_map.name)

    out_config = {
        "model_name": model_name,
        "ckpt_path": "model.pt",
        "model": config["model"],
        "activation": config["activation"],
        "dataset": {**config["dataset"], "init_args": init},
        "dataloader": config["dataloader"],
    }
    (dst / "config.yaml").write_text(yaml.safe_dump(out_config, sort_keys=False))

    # provenance: which model architecture code this checkpoint was trained
    # against, so the runner can warn when its own code no longer matches.
    # NOTE: preprocessing code isn't included...
    repo_sha, dirty = provenance.git_head()
    class_path = config["model"]["class_path"]
    meta = {
        "model_repo_sha": repo_sha,
        "model_repo_dirty": dirty,
        "model_class_path": class_path,
        "model_code_sha256": provenance.model_code_sha(class_path),
    }
    (dst / "bundle_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[bundle] {model_name} ({ckpt.name}) -> {dst}")
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dst)}")


def _wind_range(varloc: str) -> list[int]:
    df = pd.read_csv(WIND_DIRECTIONS)
    key = varloc.replace("_", "").upper()
    match = df[df["varloc"].str.replace("_", "").str.upper() == key]
    if match.empty:
        raise ValueError(f"no wind_range for {varloc!r} in {WIND_DIRECTIONS}")
    row = match.iloc[0]
    return [int(row["low_dir"]), int(row["high_dir"])]


def _resolve_stats(dataset: dict, experiment_dir: Path) -> Path:
    cfg = (dataset.get("init_args") or {}).get("stats_path")
    if cfg and Path(cfg).is_file():
        return Path(cfg)
    if (experiment_dir / "stats.yaml").is_file():
        return experiment_dir / "stats.yaml"
    raise ValueError(f"no stats file: expected {experiment_dir / 'stats.yaml'}")


def _resolve_fbfm_map(dataset: dict) -> Path:
    cfg = (dataset.get("init_args") or {}).get("fbfm_map_path")
    if cfg and Path(cfg).is_file():
        return Path(cfg)
    raise ValueError(f"no fbfm map file: expected dataset.init_args.fbfm_map_path, got {cfg!r}")
