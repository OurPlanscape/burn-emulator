import argparse
import json

from omegaconf import DictConfig, OmegaConf
from shapely.geometry import shape

from burn_emulator.constants import METHODS, Path
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
    for key in ("ckpt_path", "out_path", "run_name", "model_name"):
        value = getattr(args, key)
        if value is not None:
            OmegaConf.update(configs, key, value, merge=True)

    scalar_dataset_overrides = {
        key: value
        for key, value in (
            ("fuels_path", args.fuels_path),
            ("treatment_seed", args.treatment_seed),
            ("ignition_density", args.ignition_density),
        )
        if value
    }

    geometry_overrides = {}
    geometry_overrides["treatment_area"] = shape(json.loads(args.treatment_area)) if args.treatment_area else None
    if args.treatment_buff:
        if treatment_area is None:
            raise ValueError(
                "--treatment_buff requires --treatment_area (on the CLI or already in the "
                "loaded config's dataset.init_args)"
            )
        geometry_overrides["treatment_buff"] = area_for_buff.envelope.buffer(args.treatment_buff)

    if scalar_dataset_overrides:
        if OmegaConf.select(configs, "dataset.init_args") is None:
            raise ValueError(
                "dataset overrides were given on the CLI, but no 'dataset.init_args' section "
                "was found in the loaded config files"
            )
        for key, value in scalar_dataset_overrides.items():
            OmegaConf.update(configs, f"dataset.init_args.{key}", value, merge=True)
    configs = OmegaConf.to_container(configs, resolve=True)

    if geometry_overrides:
        configs["dataset"]["init_args"].update(geometry_overrides)

    return configs


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--method", default="train", choices=METHODS)
    parser.add_argument("-C", "--config_dir", action="store")
    parser.add_argument("-c", "--config", action="append")
    parser.add_argument("-f", "--fuels_path", action="append")
    parser.add_argument("-ta", "--treatment_area", action="store")
    parser.add_argument("-tb", "--treatment_buff", action="store")
    parser.add_argument("-ts", "--treatment_seed", action="store", type=int, default=42)
    parser.add_argument("-id", "--ignition_density", action="store", type=float)
    parser.add_argument("-p", "--ckpt_path", action="store")
    parser.add_argument("-o", "--out_path", action="store")
    parser.add_argument("-n", "--run_name", action="store")
    parser.add_argument("-mn", "--model_name", action="store")
    args = parser.parse_args()

    configs = load_configs(args.config_dir, args.config)
    configs = apply_overrides(configs, args)

    match args.method:
        case "train":
            train(**configs)
        case "test":
            test(**configs)
        case "test_iterations":
            test_iterations(**configs)
        case "run":
            run(**configs)


if __name__ == "__main__":
    main()
