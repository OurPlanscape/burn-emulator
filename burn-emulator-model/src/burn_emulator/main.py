import argparse

import yaml

from burn_emulator.constants import METHODS, Path
from burn_emulator.run import run
from burn_emulator.test import test, test_iterations
from burn_emulator.train import train


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--method", default="train", choices=METHODS)
    parser.add_argument("-C", "--config_dir", action="store", required=False)
    parser.add_argument("-c", "--config", action="append", required=False, default=[])
    parser.add_argument("-g", "--gonfig", action="append", required=False, default=[])
    parser.add_argument("-p", "--ckpt_path", action="store", required=False)
    parser.add_argument("-f", "--fuels_path", action="append", default="")
    parser.add_argument("-o", "--out_path", action="store", default="")
    parser.add_argument("-n", "--run_name", action="store", default="")
    args = parser.parse_args()

    configs = {"ckpt_path": args.ckpt_path,
               "out_path": args.out_path,
               "run_name": args.run_name}
    config_files = []
    if args.config_dir:
        config_dir = Path(args.config_dir)
        config_files.extend(config_dir.glob(".yaml"))
        
    if args.config:
        config_files.extend(args.config)
    
    for config_path in config_files:
        with Path(config_path).open() as f:
            # NOTE: this is not recursive
            configs |= yaml.safe_load(f)

    for config_path in args.gonfig:
        with Path(config_path).open() as f:
            # NOTE: this is not recursive
            configs |= yaml.safe_load(f)
    
    if args.fuels_path:
        # TODO: update this very fragile config
        configs["dataset"]["init_args"]["fuels_path"] = args.fuels_path
    
    
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