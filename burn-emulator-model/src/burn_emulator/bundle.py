import json
import shutil
from datetime import UTC, datetime
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from burn_emulator import provenance
from burn_emulator.config import wind_range
from burn_emulator.constants import BUNDLE_DIR, CONFIG_DIR, Path
from burn_emulator.utils import resolve_model_checkpoint

# dataset.init_args keys filled in at run time
_RUNTIME_DATASET_KEYS = (
    "treatment_area",
    "fuels_paths",
    "topo_path",
    "ignitions_path",
    "burn_paths",
    "wind_ang_paths",
    "fbfm_map_path",
)


def bundle(configs: DictConfig, ckpt_path: str | None = None, **kwargs: Any) -> None:
    varloc = OmegaConf.select(configs, "varloc")
    architecture = OmegaConf.select(configs, "architecture")

    model_config = OmegaConf.load(CONFIG_DIR / architecture / "model.yaml")
    configs = OmegaConf.merge(model_config, configs)
    OmegaConf.update(configs, "dataset.init_args.wind_range", wind_range(varloc))
    config = OmegaConf.to_container(configs, resolve=True)

    model_name = config["model_name"]
    experiment_dir = Path(config["experiment_dir"])
    ckpt = resolve_model_checkpoint(experiment_dir, ckpt_path)
    stats = _resolve_stats(config["dataset"], experiment_dir)

    repo_sha, dirty = provenance.git_head()
    if repo_sha is None:
        raise ValueError("bundle needs a git checkout: model_version names the commit")
    model_version = _model_version(ckpt, repo_sha, dirty)

    # <VARLOC>/<model_version>/ + <VARLOC>/current, the registry layout publish_model.sh uploads
    root = BUNDLE_DIR / varloc.upper()
    dst = root / model_version
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)

    init = {
        k: v
        for k, v in (config["dataset"].get("init_args") or {}).items()
        if k not in _RUNTIME_DATASET_KEYS
    }
    init["stats_path"] = "stats.yaml"

    shutil.copy2(ckpt, dst / "model.pt")
    shutil.copy2(stats, dst / "stats.yaml")

    out_config = {
        "model_name": model_name,
        "ckpt_path": "model.pt",
        "model": config["model"],
        "activation": config["activation"],
        "dataset": {**config["dataset"], "init_args": init},
        "dataloader": config["dataloader"],
    }
    (dst / "config.yaml").write_text(yaml.safe_dump(out_config, sort_keys=False))

    # architecture code hash; preprocessing code isn't covered
    class_path = config["model"]["class_path"]
    meta = {
        "model_repo_sha": repo_sha,
        "model_repo_dirty": dirty,
        "model_class_path": class_path,
        "model_code_sha256": provenance.model_code_sha(class_path),
    }
    (dst / "bundle_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    pointer = root / "current"
    tmp = root / f".current.{model_version}"
    tmp.write_text(model_version)
    tmp.replace(pointer)

    print(f"[bundle] {model_name} ({ckpt.name}) -> {dst}")
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dst)}")
    print(f"[bundle] {pointer} = {model_version}")


# <checkpoint mtime, UTC>-<7-char sha>[-dirty]: changes only with a new checkpoint or commit
def _model_version(ckpt: Path, repo_sha: str, dirty: bool) -> str:
    stamp = datetime.fromtimestamp(int(ckpt.stat().st_mtime), tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{repo_sha[:7]}{'-dirty' if dirty else ''}"


def _resolve_stats(dataset: dict, experiment_dir: Path) -> Path:
    cfg = (dataset.get("init_args") or {}).get("stats_path")
    if cfg and Path(cfg).is_file():
        return Path(cfg)
    if (experiment_dir / "stats.yaml").is_file():
        return experiment_dir / "stats.yaml"
    raise ValueError(f"no stats file: expected {experiment_dir / 'stats.yaml'}")
