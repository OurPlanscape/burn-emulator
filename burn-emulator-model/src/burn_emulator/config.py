import argparse
import importlib
import json
import os
import re
from datetime import datetime
from typing import Any

import geopandas as gpd
from omegaconf import DictConfig, OmegaConf
from shapely.geometry.base import BaseGeometry

from burn_emulator.constants import INF_PROFILE, OUTDIR, Path

_MODEL_NAME_FLAGS = {"varloc": "-vl", "architecture": "-a", "data_version": "-dv"}
# bare ${name} interpolations that resolve nowhere fall back to the environment,
# then to null (so optional slots can be left unset, e.g. an unexported role).
_INTERP_RE = re.compile(r"\$\{(\w+)\}")


def load_configs(config_dir: str | None, config_paths: list[str] | None) -> DictConfig:
    config_files = []
    if config_dir:
        config_files.extend(sorted(Path(config_dir).glob("*.yaml")))
    if config_paths:
        config_files.extend(Path(p) for p in config_paths)

    loaded = []
    for config_path in config_files:
        with Path(config_path).open() as f:
            loaded.append(OmegaConf.load(f))

    merged = OmegaConf.merge(*loaded) if loaded else OmegaConf.create()

    fill = {}
    for name in set(_INTERP_RE.findall(OmegaConf.to_yaml(merged))):
        try:
            if OmegaConf.select(merged, name) is not None:
                continue
        except Exception:
            pass
        fill[name] = os.environ.get(name)
    if fill:
        merged = OmegaConf.merge(OmegaConf.create(fill), merged)

    return merged


def _iso_data_version(value: str) -> str:
    value = value.strip()
    for fmt in ("%d%b%Y", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError(
        f"data_version {value!r} is not a recognised date "
        "(expected DDMonYYYY like 28Aug2026, YYYYMMDD, or YYYY-MM-DD)"
    )


def resolve_model_name(
    configs: DictConfig,
    varloc: str | None = None,
    architecture: str | None = None,
    data_version: str | None = None,
) -> None:
    for key, value in (
        ("varloc", varloc),
        ("architecture", architecture),
        ("data_version", data_version),
    ):
        if value is not None:
            OmegaConf.update(configs, key, value, merge=True)

    parts = {k: OmegaConf.select(configs, k) for k in _MODEL_NAME_FLAGS}
    missing = [f"{_MODEL_NAME_FLAGS[k]}/{k}" for k, v in parts.items() if v is None]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} required: pass the flag or set it in a config file"
        )

    parts["data_version"] = _iso_data_version(str(parts["data_version"]))
    model_name = "{varloc}_{architecture}_{data_version}".format(**parts)
    OmegaConf.update(configs, "model_name", model_name, merge=True)
    OmegaConf.update(configs, "experiment_dir", str(OUTDIR / model_name), merge=True)


def apply_overrides(configs: DictConfig, args: argparse.Namespace) -> dict:
    if args.ckpt_path is not None:
        OmegaConf.update(configs, "ckpt_path", args.ckpt_path, merge=True)

    dataset_overrides = {
        key: value
        for key, value in (
            ("treatment_area", args.treatment_area),
            ("treatment_buff", args.treatment_buff),
            ("treatment_seed", args.treatment_seed),
            ("ignition_density", args.ignition_density),
            ("wind_seed", args.wind_seed)
        )
        if value is not None
    }

    if args.treatment_area_crs is not None:
        dataset_overrides["treatment_area_crs"] = args.treatment_area_crs

    if dataset_overrides:
        if OmegaConf.select(configs, "dataset.init_args") is None:
            raise ValueError(
                "dataset overrides were given on the CLI, but no 'dataset.init_args' section "
                "was found in the loaded config files"
            )
        for key, value in dataset_overrides.items():
            OmegaConf.update(configs, f"dataset.init_args.{key}", value, merge=True)

    configs = OmegaConf.to_container(configs, resolve=True)

    init_args = configs.get("dataset", {}).get("init_args") or {}
    treatment_area_crs = init_args.pop("treatment_area_crs", None)

    if args.fbfm_map_path:
        init_args["fbfm_map_path"] = args.fbfm_map_path

    fuels_paths = dict(init_args.get("fuels_paths") or {})
    if args.baseline_fuels:
        fuels_paths["baseline"] = args.baseline_fuels
    if args.legalmax_fuels:
        fuels_paths["treatment"] = args.legalmax_fuels
    if fuels_paths:
        init_args["fuels_paths"] = fuels_paths

    if args.topo_path:
        init_args["topo_path"] = args.topo_path

    if init_args.get("treatment_area") is not None:
        init_args["treatment_area"] = load_treatment_area(
            init_args["treatment_area"], treatment_area_crs
        )
    elif init_args.get("treatment_buff") is not None:
        raise ValueError("treatment_buff requires treatment_area")

    return configs


def dynamic_import(loader: dict, kwargs: dict | None = None) -> Any:
    class_path = loader.get("class_path")
    if not class_path or "." not in class_path:
        raise ValueError(f"loader needs a dotted 'class_path', got {class_path!r}")

    # build a fresh dict so the caller's loader config is never mutated
    init_args = {**(loader.get("init_args") or {}), **(kwargs or {})}

    module_path, class_name = class_path.rsplit(".", 1)
    loader_cls = getattr(importlib.import_module(module_path), class_name)

    return loader_cls(**init_args)


# raster CRS everything is reprojected into before use.
TARGET_CRS = INF_PROFILE["crs"]


def load_treatment_area(
    value: str | dict | BaseGeometry | None, crs: str | None = None
) -> BaseGeometry | None:
    if value is None:
        return None

    from shapely.geometry import shape

    if isinstance(value, BaseGeometry):
        if not crs:
            return value  # caller's contract: already in TARGET_CRS
        geoms = [value]
    else:
        obj = value if isinstance(value, dict) else None
        if obj is None and isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            obj = json.loads(value)

        if obj is None:
            geom = gpd.read_file(value).to_crs(TARGET_CRS).union_all()
            _assert_plausible_bounds(geom)
            return geom

        if not crs:
            raise ValueError("treatment_area is inline geometry but no crs was given")
        kind = obj.get("type")
        if kind == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in obj["features"]]
        elif kind == "Feature":
            geoms = [shape(obj["geometry"])]
        else:
            geoms = [shape(obj)]

    try:
        geom = gpd.GeoSeries(geoms, crs=crs).to_crs(TARGET_CRS).union_all()
    except Exception as e:
        raise ValueError(f"invalid treatment_area_crs {crs!r}: {e}") from e

    _assert_plausible_bounds(geom)
    return geom


def _assert_plausible_bounds(geom: BaseGeometry) -> None:
    minx, miny, maxx, maxy = geom.bounds
    if not (-3e6 < minx < 3e6 and -3e6 < miny < 3e6):
        raise ValueError(
            f"treatment_area implausible after reprojection to {TARGET_CRS}: "
            f"{geom.bounds}; check treatment_area_crs"
        )
