import argparse

from omegaconf import DictConfig, OmegaConf

from burn_emulator.bundle import bundle
from burn_emulator.config import load_treatment_area
from burn_emulator.constants import BUNDLE_DIR, METHODS, Path
from burn_emulator.run import run
from burn_emulator.test import test, test_iterations
from burn_emulator.train import train


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

    return OmegaConf.merge(*loaded) if loaded else OmegaConf.create()


def apply_overrides(configs: DictConfig, args: argparse.Namespace) -> dict:
    for key in ("ckpt_path", "out_path", "model_name"):
        value = getattr(args, key)
        if value is not None:
            OmegaConf.update(configs, key, value, merge=True)

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
        init_args["treatment_area"] = load_treatment_area(init_args["treatment_area"])
    elif init_args.get("treatment_buff") is not None:
        raise ValueError("treatment_buff requires treatment_area")

    return configs


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--method", default="train", choices=METHODS)
    parser.add_argument("-C", "--config_dir", action="store")
    parser.add_argument("-c", "--config", action="append")
    parser.add_argument("-bf", "--baseline_fuels", action="store")
    parser.add_argument("-lf", "--legalmax_fuels", action="store")
    parser.add_argument("-tp", "--topo_path", action="store")
    parser.add_argument("-ta", "--treatment_area", action="store")
    parser.add_argument("-tb", "--treatment_buff", action="store", type=float)
    parser.add_argument("-ts", "--treatment_seed", action="store", type=float)
    parser.add_argument("-id", "--ignition_density", action="store", type=float)
    parser.add_argument("-ws", "--wind_seed", action="store", type=int)
    parser.add_argument("-p", "--ckpt_path", action="store")
    parser.add_argument("-o", "--out_path", action="store")
    parser.add_argument("-mn", "--model_name", action="store")
    parser.add_argument("-vl", "--varloc", action="store")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    configs = load_configs(args.config_dir, args.config)

    if args.method == "bundle":
        configs = OmegaConf.to_container(configs, resolve=True)
        for key in ("ckpt_path", "model_name"):
            if getattr(args, key) is not None:
                configs[key] = getattr(args, key)
    else:
        configs = apply_overrides(configs, args)
        configs["debug"] = args.debug

    match args.method:
        case "train":
            train(**configs)
        case "test":
            test(**configs)
        case "test_iterations":
            test_iterations(**configs)
        case "run":
            run(**configs)
        case "bundle":
            if args.out_path:
                dest = args.out_path
            elif args.varloc:
                dest = str(BUNDLE_DIR / args.varloc)
            else:
                parser.error("bundle needs -o, -vl, or -C")
            bundle(dest=dest, **configs)


if __name__ == "__main__":
    main()
