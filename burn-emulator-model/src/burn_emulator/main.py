import argparse

from burn_emulator.bundle import bundle
from burn_emulator.config import apply_overrides, load_configs, resolve_model_name
from burn_emulator.constants import METHODS
from burn_emulator.evaluate import evaluate, evaluate_iterations
from burn_emulator.run import run
from burn_emulator.train import train


def main():
    parser = argparse.ArgumentParser(description="")

    # all methods
    parser.add_argument("-m", "--method", default="train", choices=METHODS)
    parser.add_argument("-C", "--config_dir", action="store")
    parser.add_argument("-c", "--config", action="append")

    # model identity: -vl/-dv used by every method, -a by all but ignite
    parser.add_argument("-vl", "--varloc", action="store")
    parser.add_argument("-a", "--architecture", action="store")
    parser.add_argument("-dv", "--data_version", action="store")

    # dataset overrides: train / evaluate / evaluate_iterations / run only
    parser.add_argument("-mp", "--fbfm_map_path", action="store")
    parser.add_argument("-bf", "--baseline_fuels", action="store")
    parser.add_argument("-lf", "--legalmax_fuels", action="store")
    parser.add_argument("-tp", "--topo_path", action="store")
    parser.add_argument("-ta", "--treatment_area", action="store")
    parser.add_argument("-tc", "--treatment_area_crs", action="store")
    parser.add_argument("-tb", "--treatment_buff", action="store", type=float)
    parser.add_argument("-ts", "--treatment_seed", action="store", type=float)
    parser.add_argument("-id", "--ignition_density", action="store", type=float)
    parser.add_argument("-wr", "--wind_range", action="store", nargs=2, type=float)
    parser.add_argument("-ws", "--wind_seed", action="store", type=int)
    parser.add_argument("-d", "--debug", action="store_true")

    # checkpoint: train / evaluate / evaluate_iterations / run / bundle
    parser.add_argument("-cp", "--ckpt_path", action="store")

    # run only
    parser.add_argument("-o", "--out_path", action="store")

    # ignite only
    parser.add_argument("-ni", "--num_ignitions", action="store", type=int)
    parser.add_argument("-ci", "--collate_ignitions", action="store_true")

    args = parser.parse_args()

    configs = load_configs(args.config_dir, args.config)

    # ignite generates training data from a varloc string, not a model/dataset config
    if args.method != "ignite":
        resolve_model_name(configs, args.varloc, args.architecture, args.data_version)

    # bundle assembles its own config from the DictConfig; ignite has no config at all
    if args.method not in ("bundle", "ignite"):
        configs = apply_overrides(configs, args)
        configs["debug"] = args.debug

    match args.method:
        case "train":
            train(**configs)
        case "evaluate":
            evaluate(**configs)
        case "evaluate_iterations":
            evaluate_iterations(**configs)
        case "run":
            run(**configs)
        case "bundle":
            bundle(configs, ckpt_path=args.ckpt_path)
        case "ignite":
            # deferred: pyretechnics/ray are optional deps (pyproject.toml [data] extra)
            from burn_emulator.ignite import ignite

            ignite_kwargs = {
                "varloc": args.varloc,
                "data_version": args.data_version,
                "collate_ignitions": args.collate_ignitions,
            }
            if args.num_ignitions is not None:
                ignite_kwargs["num_ignitions"] = args.num_ignitions
            ignite(**ignite_kwargs)


if __name__ == "__main__":
    main()
