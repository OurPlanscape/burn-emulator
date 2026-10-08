import csv
import json
import os
from functools import cache

from burn_emulator import constants, provenance
from burn_emulator.constants import Path
from omegaconf import OmegaConf

# burn-emulator-model/configs, next to the editable install's src/; burn_emulator has no __init__.py
MODEL_CONFIGS = Path(constants.__file__).resolve().parents[2] / "configs"
WIND_DIRECTIONS_CSV = MODEL_CONFIGS / "wind_directions.csv"
RUN_SMOKE_YAML = MODEL_CONFIGS / "varlocs" / "templates" / "run_smoke.yaml"

_CLOUD_SCHEMES = ("gs://", "s3://", "az://")


def bundle_dir(path: str) -> Path:
    d = Path(path)
    if not (d / "model.pt").exists():
        raise FileNotFoundError(f"no model bundle at {d}")
    return d


def wind_range(varloc: str) -> list[float]:
    with WIND_DIRECTIONS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["varloc"] == varloc:
                return [float(row["low_dir"]), float(row["high_dir"])]
    raise ValueError(f"no wind range for varloc {varloc} in {WIND_DIRECTIONS_CSV}")


# PT without a bundle: run_smoke.yaml defaults, stats computed in memory
def pt_spec(varloc: str, inputs_version: str, work_dir: str) -> dict:
    params = {
        "varloc": varloc,
        "inputs_version": inputs_version,
        "wind_range": wind_range(varloc),
    }
    # params last: run_smoke.yaml's own varloc: ${varloc} would resolve to itself
    merged = OmegaConf.merge(OmegaConf.load(RUN_SMOKE_YAML), OmegaConf.create(params))
    spec = OmegaConf.to_container(merged, resolve=True)
    for k in params:
        spec.pop(k, None)
    spec["model_name"] = f"{varloc}_pt_{inputs_version}"
    spec["model"] = {}
    spec["experiment_dir"] = work_dir
    spec["dataset"]["init_args"]["stats_path"] = os.path.join(work_dir, "stats.yaml")
    return spec


def load_spec(bundle: Path) -> dict:
    config = bundle / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"no config.yaml in model bundle {bundle}")

    spec = OmegaConf.to_container(OmegaConf.load(os.fspath(config)), resolve=True)
    spec["experiment_dir"] = os.fspath(bundle)
    spec["ckpt_path"] = _localize(spec.get("ckpt_path", "model.pt"), bundle)

    init = spec.setdefault("dataset", {}).setdefault("init_args", {})
    if not isinstance(init.get("stats_path"), str):
        raise ValueError(f"model bundle {bundle} config.yaml has no dataset.init_args.stats_path")
    init["stats_path"] = _localize(init["stats_path"], bundle)
    if not init["stats_path"].startswith(_CLOUD_SCHEMES) and not os.path.isfile(init["stats_path"]):
        raise FileNotFoundError(f"model bundle {bundle} is missing {init['stats_path']}")

    return spec


def _localize(p: str, root: Path) -> str:
    if p.startswith(_CLOUD_SCHEMES) or os.path.isabs(p):
        return p
    return os.fspath(root / p)


@cache
def image_model_code_sha(class_path: str) -> str:
    return provenance.model_code_sha(class_path)


def read_provenance(bundle: Path) -> tuple[dict | None, str | None]:
    p = bundle / "bundle_meta.json"
    if not p.is_file():
        return None, None
    meta = json.loads(p.read_text())
    try:
        current = image_model_code_sha(meta["model_class_path"])
    except (KeyError, OSError):
        current = None
    return meta, current
