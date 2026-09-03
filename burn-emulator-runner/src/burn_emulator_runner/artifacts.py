import json
import os
import re
from functools import cache

from burn_emulator import provenance
from burn_emulator.constants import Path
from omegaconf import OmegaConf

MODELS_DIR = Path(os.environ.get("BURN_EMULATOR_MODELS_DIR", "/models"))

_CLOUD_SCHEMES = ("gs://", "s3://", "az://")
_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def bundle_dir(varloc: str, version: str) -> Path:
    if not _SEGMENT.match(varloc) or not _SEGMENT.match(version):
        raise ValueError(f"invalid varloc/version: {varloc!r}/{version!r}")
    d = MODELS_DIR / varloc / version
    if not (d / "model.pt").exists():
        raise FileNotFoundError(f"no model bundle at {d}")
    return d


def load_spec(bundle: Path) -> dict:
    yamls = sorted(bundle.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"no *.yaml in model bundle {bundle}")

    merged = OmegaConf.merge(*(OmegaConf.load(os.fspath(p)) for p in yamls))
    spec = OmegaConf.to_container(merged, resolve=True)

    spec["ckpt_path"] = _localize(spec.get("ckpt_path", "model.pt"), bundle)

    init = spec.setdefault("dataset", {}).setdefault("init_args", {})
    if isinstance(init.get("stats_path"), str):
        init["stats_path"] = _localize(init["stats_path"], bundle)
    if isinstance(init.get("fbfm_map_path"), str):
        init["fbfm_map_path"] = _localize(init["fbfm_map_path"], bundle)

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
