import argparse

from burn_emulator.bundle import bundle
from burn_emulator.config import (
    apply_overrides,
    current_data_version,
    load_configs,
    resolve_model_name,
)
from burn_emulator.constants import METHODS
from burn_emulator.evaluate import evaluate
from burn_emulator.run import run
from burn_emulator.train import train


def _str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


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

    # dataset overrides: train / evaluate / run only
    parser.add_argument("-mp", "--fbfm_map_path", action="store")
    parser.add_argument("-bf", "--baseline_fuels", action="store")
    parser.add_argument("-lf", "--legalmax_fuels", action="store")
    parser.add_argument("-tp", "--topo_path", action="store")
    parser.add_argument("-ta", "--treatment_area", action="store")
    parser.add_argument("-tb", "--treatment_buff", action="store", type=float)
    parser.add_argument("-ts", "--treatment_seed", action="store", type=float)
    parser.add_argument("-id", "--ignition_density", action="store", type=float)
    parser.add_argument("-wr", "--wind_range", action="store", nargs=2, type=float)
    parser.add_argument("-ws", "--wind_seed", action="store", type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    # cache burn windows in memory (training); `-cb false` with jitter
    parser.add_argument("-cb", "--cache_burns", action="store", type=_str2bool, default=True)

    # checkpoint: train / evaluate / run / bundle
    parser.add_argument("-cp", "--ckpt_path", action="store")

    # run only
    parser.add_argument("-o", "--out_path", action="store")

    # run / evaluate: pyretechnics instead of the emulator
    parser.add_argument("-pt", "--pyretechnics", action="store_true")

    # ignite only
    parser.add_argument("-ni", "--num_ignitions", action="store", type=int)
    parser.add_argument("-ci", "--collate_ignitions", action="store_true")
    parser.add_argument("-ow", "--overwrite", action="store_true")

    args = parser.parse_args()

    configs = load_configs(args.config_dir, args.config)

    if args.method != "ignite":
        resolve_model_name(configs, args.varloc, args.architecture, args.data_version)

    if args.method not in ("bundle", "ignite"):
        configs = apply_overrides(configs, args)
        configs["debug"] = args.debug
        configs["pyretechnics"] = args.pyretechnics
        if args.pyretechnics:
            configs["backend"] = "PT"

    match args.method:
        case "train":
            train(**configs)
        case "evaluate":
            evaluate(**configs)
        case "run":
            run(**configs)
        case "bundle":
            bundle(configs, ckpt_path=args.ckpt_path)
        case "ignite":
            # optional [data] extra
            from burn_emulator.ignite import ignite

            ignite_kwargs = {
                "varloc": args.varloc,
                "data_version": args.data_version or current_data_version(),
                "collate_ignitions": args.collate_ignitions,
                "overwrite": args.overwrite,
            }
            if args.num_ignitions is not None:
                ignite_kwargs["num_ignitions"] = args.num_ignitions
            ignite(**ignite_kwargs)


if __name__ == "__main__":
    main()
