import importlib
import json
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def dynamic_import(loader: dict, kwargs: dict | None = None) -> Any:
    class_path = loader.get("class_path")
    init_args = loader.get("init_args", {})
    if kwargs is not None:
        init_args |= kwargs

    loader_path = class_path.rsplit(".", 1)
    module_path, class_name = loader_path
    loader_cls = getattr(importlib.import_module(module_path), class_name)

    return loader_cls(**init_args)


def load_treatment_area(value: str | dict | BaseGeometry | None) -> BaseGeometry | None:
    if value is None:
        return None

    from shapely import union_all
    from shapely.geometry import shape

    if isinstance(value, BaseGeometry):
        return value

    obj = value if isinstance(value, dict) else None
    if obj is None and isinstance(value, str) and value.lstrip()[:1] in "{[":
        obj = json.loads(value)

    if obj is not None:
        kind = obj.get("type")
        if kind == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in obj["features"]]
        elif kind == "Feature":
            geoms = [shape(obj["geometry"])]
        else:
            geoms = [shape(obj)]
        return geoms[0] if len(geoms) == 1 else union_all(geoms)

    return gpd.read_file(value).union_all()
