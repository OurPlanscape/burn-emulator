import os
import re

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

    return spec


def _localize(p: str, root: Path) -> str:
    if p.startswith(_CLOUD_SCHEMES) or os.path.isabs(p):
        return p
    return os.fspath(root / p)
