# burn-emulator-model

Model library + CLI. `burn_emulator.run.run()` is the inference entry point that [`burn-emulator-runner`](../burn-emulator-runner) imports.

## Modelling

The default model is an archetypal CNN with one major adjustment: [circular kernels](https://arxiv.org/pdf/2107.02451). The ignition point is fixed at the center pixel of the CNN context window, so the model only has to learn the mapping from local input connectivities (fuels, etc.) to a spread shape from that point (i.e it doesn't need to know *where* a burn will occur, just a fuzzy approximation of what it will look like). That approximation is then stamped across the landscape in parallel to compute conditional burn probability.

## Install

```bash
uv sync && uv pip install -e burn-emulator-model
uv sync --extra data     # for scripts/ (training-data generation from Pyretechnics)
```

`burn-emulator-model/`  scripts are intended to be run within the repo.

## Train

All configs are composable, with priority going to CLI flags then the latest config entered.
```bash
burn_emulator -m train -c <model.yaml> -c <train.yaml> -c <data.yaml>
```

Output: `data/outputs/<model_name>/` ; `checkpoints/`, `stats.yaml`, `train_log.csv`.

To train every varloc in a batch, run `scripts/train_varlocs.sh` (or `slurm/train_varlocs.slurm` on the cluster). It reads `configs/varlocs/varlocs.txt` ; one varloc name per line, where each name maps to `data/training_data/<varloc>/<data_version>/`. The active architecture and data version come from `configs/varlocs/current.yaml`.

## Evaluate

```bash
# evaluate will only run inference on a set of ignitions
# since evaluation evolves constantly in various scripts
# those tasks are not implemented and left for the DS to do
burn_emulator -m evaluate \
              -a "$ARCHITECTURE" \
              -vl "$VARLOC" \
              -dv "$DATA_VERSION" \
              -c <model.yaml> -c <eval_data.yaml>
```

Output: `data/outputs/<model_name>/inference/` ; per-ignition prediction GeoTIFFs, `throughput.csv`.

To evaluate every ignition scenario (baseline + legalmax) under a directory:
```bash
scripts/ignite_inference.sh <varloc> <outputs_root>
```
`<outputs_root>` holds one subdirectory per scenario, each with `<N>_baseline` / `<N>_legalmax` ignition sets. Resolves `$ARCHITECTURE` / `$DATA_VERSION` from `configs/varlocs/current.yaml`.

## Run

```bash
burn_emulator -m run \
              -c <model.yaml> \
              -c <data.yaml> \
              -bf <baseline_fuels> \
              -lf <legalmax_fuels> \
              -tp <topo> \
              -ta <treatment_area> \
              -o <output_path>
```

```
1. merge -c YAMLs + CLI overrides
2. build the VarLoc dataset: sample seeded ignitions across the buffered treatment_area,
   one window per ignition (baseline + collated-treatment fuels, wind, circular mask)
3. load the checkpoint if exists (-p, else lowest-loss in <dir>/checkpoints)
4. per batch: forward baseline + treatment -> activation -> argmax -> fire_type per pixel
5. per fire:
    keep the center-connected burn (NN outputs have minor speckling)
    classify crown change, drop fires that
    miss the treatment region
6. aggregate kept fires onto the full raster (fp32) -> per-pixel change probabilities
7. write <dir>/<model_name>.tif   (3-band float32 GeoTIFF)
```

`<dir>` is `-o` if given, else `data/outputs/<model_name>`.

Env: `RUN_DEVICE` (`cuda` | `XLA` | `cpu`), `RUN_DTYPE` (`bfloat16`), `USE_CLOUD_PATHS` (`1` for `gs://`).

## Inputs

```
ignitions_path/        # ignition points (or sampled from treatment_area)
topo_path/             # tifs: aspect / slope
baseline_fuels_path/   # tifs: baseline fuels ("cbd", "cbh", "cc", "fbfm", "th")
legalmax_fuels_path/   # tifs: treatment fuels ("cbd", "cbh", "cc", "fbfm", "th")
burn_paths/{ignition_number}/fire_type.tif # only for training
```

## Datasets

`VarLoc` (`burn_emulator.datasets.varloc`) is the only dataset. It reads the fuel layers listed above (`"cbd", "cbh", "cc", "fbfm", "th"`), windows one context per ignition, and yields a dict keyed by the model contract:

| key | meaning |
| --- | --- |
| `x` | stacked input channels (topo + fuels + FBFM one-hots) |
| `y` | per-pixel `fire_type` target ; train only |
| `wind` | ignition wind direction (degrees) |
| `mask` | burnable / circular-window mask |

Inference samples additionally carry `pdiffs` / `bounds` / `indxes` for stamping predictions back onto the full raster.

If a new fuel product ships different layers (renamed, added/dropped, or different semantics/resolution), add a new `Dataset` rather than messing with `VarLoc`. Keep the same output contract [`x`, `y`, `wind`, `mask`] so `train` / `evaluate` / `run` and `model.forward(x, wind)` work unchanged.

## Publish a model

```bash
burn_emulator -m bundle -c configs/varlocs/current.yaml -vl <varloc>
scripts/publish_model.sh <varloc> <bundle_dir> <models_uri>
```

`-m bundle` resolves `<varloc>` against `configs/varlocs/current.yaml`'s architecture/data_version and writes the bundle to `data/bundles/<model_name>/`. `publish_model.sh` uploads that bundle to `<models_uri>` and repoints `current`; `<models_uri>` can also come from `BURN_EMULATOR_MODELS_URI` instead of the third argument.

## Publish inputs (fuels + topo)

```bash
scripts/publish_inputs.sh <data_version> <fuels_dir> <topo_dir> [inputs_uri]
# <data_version>  DDMonYYYY (28Aug2026) or YYYYMMDD, stored as YYYYMMDD
# <fuels_dir>     both baseline_*.tif and legalmax_*.tif, e.g. data/training_data/West_Fuels_DN_24Aug2026
# <topo_dir>      all topo tifs, uploaded as-is
```

Fuels and topo are published together under one `data_version`, matching how the training data pairs them: `publish_inputs.sh` splits `<fuels_dir>` by filename into `baseline/` and `legalmax/`, uploads `<topo_dir>` wholesale to `topo/`, all under `${inputs_uri}/<data_version>/`, and only then repoints `${inputs_uri}/current` (a failed upload leaves `current` on the previous version). Re-running skips a layer that's already published unless `FORCE=1`. `<inputs_uri>` can also come from `BURN_EMULATOR_INPUTS_URI` instead of the fourth argument; the script aborts if neither is set.

Re-publishing an existing `data_version` with `FORCE=1` does not invalidate outputs already cached under it; publish changed inputs under a new `data_version`.

`-m bundle` writes `data/bundles/<model_name>/`:

| file | from |
| --- | --- |
| `model.pt` | `-p`, else the lowest-loss checkpoint under `data/outputs/<model_name>/checkpoints` |
| `stats.yaml` | the same training dir |
| `fbfm_behavior_adjectives.csv` | `dataset.init_args.fbfm_map_path` |
| `config.yaml` | the `-c` config with runtime-injected fields removed |
| `bundle_meta.json` | `model_repo_sha` (+ dirty flag), `model_class_path`, and `model_code_sha256` (sha256 of that architecture module `.py`) |

`bundle_meta.json` lets the runner warn when its architecture code no longer matches what this checkpoint was trained on but currently doesn't do anything YET! See [`burn-emulator-runner`](../burn-emulator-runner). `publish_model.sh` refuses a bundle that lacks it.

`publish_model.sh` uploads it to `gs://<models>/<varloc>/<model_version>/` (`<model_version>` = `<model.pt mtime>-<git sha>`) and repoints `current`. The runner injects `treatment_area` / `fuels_paths` / `topo_path` / `ignitions_path` at request time.
