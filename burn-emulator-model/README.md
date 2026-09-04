# burn-emulator-model

Model library + CLI. `burn_emulator.run.run()` is the inference entry point that [`burn-emulator-runner`](../burn-emulator-runner) imports.

## Modelling

The default model is an archetypal CNN with one major adjustment: [circular kernels](https://arxiv.org/pdf/2107.02451). The ignition point is fixed at the center pixel of the CNN context window, so the model only has to learn the mapping from local input connectivities (fuels, etc.) to a spread shape from that point (i.e it doesn't need to know *where* a burn will occur, just a fuzzy approximation of what it will look like). That approximation is then stamped across the landscape in parallel to compute conditional burn probability.

## Install

```bash
uv sync && uv pip install -e .
uv sync --extra data     # for scripts/ (training-data generation from Pyretechnics)
```

## Train

All configs are composable, with priority going to CLI flags then the latest config entered.
```bash
burn_emulator -m train -c <model.yaml> -c <train.yaml> -c <data.yaml>
```

Output: `data/outputs/<model_name>/` ; `checkpoints/`, `stats.yaml`, `train_log.csv`.

To train every varloc in a batch, use `scripts/train_varlocs.sh` (or `slurm/train_varlocs.slurm`). It reads `configs/varlocs/varlocs.txt` ; one varloc name per line, where each name maps to `data/training_data/<varloc>/<data_version>/`. The active architecture and data version come from `configs/varlocs/current.yaml`.

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
7. write <dir>/<model_name>_run.tif   (3-band float32 GeoTIFF)
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
# make sure here to match model name and varloc
burn_emulator -m bundle -c configs/circle_net_bundle_template.yaml -mn <model_name> -vl <varloc>
make publish-model VARLOC=<varloc>
```

## Publish fuels

```bash
make publish-fuels
```

Both publish targets need a `gs://` destination root, taken from the environment
(or an explicit make var):

| var | used by | make override |
| --- | --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `make publish-model` (model registry root) | `MODELS_URI=` |
| `BURN_EMULATOR_FUELS_URI` | `make publish-fuels` (published fuel/topo layers root) | `FUELS_URI=` |

The scripts abort if neither the env var nor the make var is set.

`-m bundle` writes `data/bundles/<model_name>/`:

| file | from |
| --- | --- |
| `model.pt` | `-p`, else the lowest-loss checkpoint under `data/outputs/<model_name>/checkpoints` |
| `stats.yaml` | the same training dir |
| `fbfm_behavior_adjectives.csv` | `dataset.init_args.fbfm_map_path` |
| `config.yaml` | the `-c` config with runtime-injected fields removed |
| `bundle_meta.json` | `model_repo_sha` (+ dirty flag), `model_class_path`, and `model_code_sha256` (sha256 of that architecture module `.py`) |

`bundle_meta.json` lets the runner warn when its architecture code no longer matches what this checkpoint was trained on but currently doesn't do anything YET! See [`burn-emulator-runner`](../burn-emulator-runner). `publish_model.sh` refuses a bundle that lacks it.

`make publish-model` uploads it to `gs://<models>/<varloc>/<version>/` and repoints `current`. The runner injects `treatment_area` / `fuels_paths` / `topo_path` / `ignitions_path` at request time.
