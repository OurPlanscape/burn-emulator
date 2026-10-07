import csv
import json
import os
import re
from functools import cache

import burn_emulator
from burn_emulator import provenance
from burn_emulator.constants import Path
from omegaconf import OmegaConf

MODELS_DIR = Path(os.environ.get("BURN_EMULATOR_MODELS_DIR", "/models"))

# burn-emulator-model/configs, next to the editable install's src/
WIND_DIRECTIONS_CSV = (
    Path(burn_emulator.__file__).resolve().parents[2] / "configs" / "wind_directions.csv"
)

_CLOUD_SCHEMES = ("gs://", "s3://", "az://")
_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def bundle_dir(varloc: str, model_version: str) -> Path:
    if not _SEGMENT.match(varloc) or not _SEGMENT.match(model_version):
        raise ValueError(f"invalid varloc/model_version: {varloc!r}/{model_version!r}")
    d = MODELS_DIR / varloc / model_version
    if not (d / "model.pt").exists():
        raise FileNotFoundError(f"no model bundle at {d}")
    return d


def wind_range(varloc: str) -> list[float]:
    with WIND_DIRECTIONS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["varloc"] == varloc:
                return [float(row["low_dir"]), float(row["high_dir"])]
    raise ValueError(f"no wind range for varloc {varloc} in {WIND_DIRECTIONS_CSV}")


# PT runs without a bundle: run_smoke.yaml dataset defaults, stats computed in memory
def pt_spec(varloc: str, inputs_version: str, work_dir: str) -> dict:
    return {
        "model_name": f"{varloc}_pt_{inputs_version}",
        "model": {},
        "activation": {},
        "experiment_dir": work_dir,
        "backend": "PT",
        "dataset": {
            "class_path": "burn_emulator.datasets.varloc.VarLoc",
            "init_args": {
                "treatment_buff": 960,
                "ignition_density": 20,
                "ignition_method": "uniform",
                "treatment_seed": 42,
                "wind_range": wind_range(varloc),
                "wind_seed": 42,
                "window_size": 129,
                "stats_path": os.path.join(work_dir, "stats.yaml"),
            },
        },
        "dataloader": {
            "class_path": "torch.utils.data.DataLoader",
            "init_args": {"batch_size": 1, "shuffle": False},
        },
    }


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
