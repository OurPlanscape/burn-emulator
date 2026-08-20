import importlib


def dynamic_import(loader: dict, kwargs: dict | None = None):
    class_path = loader.get("class_path")
    init_args = loader.get("init_args", {})
    if kwargs is not None:
        init_args |= kwargs

    loader_path = class_path.rsplit(".", 1)
    module_path, class_name = loader_path
    loader_cls = getattr(importlib.import_module(module_path), class_name)

    return loader_cls(**init_args)
