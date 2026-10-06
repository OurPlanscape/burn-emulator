# burn-emulator-model

Model library + CLI. `burn_emulator.run.run()` is the inference entry point that [`burn-emulator-runner`](../burn-emulator-runner) imports.

## Modelling

The default model is a CNN with [circular kernels](https://arxiv.org/pdf/2107.02451). The ignition point is fixed at the center pixel of the context window, so the model learns the spread shape from local inputs (fuels, topo, wind) around that point, not where a burn occurs. At inference that prediction is stamped across the landscape in parallel to compute conditional burn probability.

## Install

```bash
make shell   # from the repo root: syncs burn-emulator-model/.venv-<arch> (with the data extra) and activates it
```

or, from `burn-emulator-model/`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-$(uname -m) uv sync              # library + CLI
UV_PROJECT_ENVIRONMENT=.venv-$(uname -m) uv sync --extra data # + pyretechnics/ray for -m ignite and PT runs
```

Run `scripts/` from `burn-emulator-model/`.

## CLI

`burn_emulator -m <train|evaluate|run|bundle|ignite>`. `-c` configs merge in order (later wins) and CLI flags override them. `-a` / `-vl` / `-dv` (or `architecture` / `varloc` / `data_version` in a config) set `model_name` = `<VARLOC>_<architecture>_<data_version>` and the experiment dir `data/outputs/<model_name>/`. `configs/varlocs/current.yaml` holds the current `architecture`, `inputs_version` and `ignitions_version`; `data_version` = `<inputs_version>_<ignitions_version>`.

## Training data

```bash
burn_emulator -m ignite -vl <varloc> -dv <data_version> [-ni <num_ignitions>] [-ow]
```

Clips fuels and topo for the varloc and simulates burns with pyretechnics into `data/training_data/<varloc>/<data_version>/`. A varloc is complete once `legalmax/outputs_table.csv` exists. See `make training-data` / `training-data-all`.

## Train

```bash
burn_emulator -m train -a <architecture> -vl <varloc> -dv <data_version> \
              -c configs/<architecture>/model.yaml -c configs/<architecture>/train.yaml -c configs/varlocs/templates/train.yaml
```

Output: `data/outputs/<model_name>/`: `checkpoints/`, `stats.yaml`, `train_log.csv`. Resumes from the latest checkpoint if one exists. See `make train-all`.

## Evaluate

```bash
burn_emulator -m evaluate -a <architecture> -vl <varloc> -dv <data_version> -c <model.yaml> -c <eval_data.yaml>
```

Runs inference on a set of ignitions; it does not compute metrics. Output: `data/outputs/<model_name>/inference/`: per-ignition prediction GeoTIFFs and `throughput.csv`.

To evaluate every ignition scenario (baseline + legalmax) under a directory:

```bash
scripts/ignite_inference.sh <varloc> <outputs_root>
```

`<outputs_root>` holds one subdirectory per scenario, each with `<N>_baseline` / `<N>_legalmax` ignition sets. Architecture and data version come from `configs/varlocs/current.yaml`.

## Run

```bash
burn_emulator -m run -vl <varloc> \
              -c configs/varlocs/current.yaml -c configs/<architecture>/model.yaml -c <data.yaml> \
              -bf <baseline_fuels> -lf <legalmax_fuels> -tp <topo> -mp <fbfm_map> \
              -ta <treatment_area> [-tc <crs>] [-id <ignition_density>] [-cp <checkpoint>] [-o <out.tif>] [-pt]
```

```
1. merge -c configs + CLI overrides
2. build the VarLoc dataset: sample seeded ignitions over the buffered treatment_area,
   one window per ignition (baseline + collated-treatment fuels, wind, circular mask)
3. load the checkpoint (-cp, else the lowest-loss one in <experiment_dir>/checkpoints)
4. per batch: forward baseline + treatment -> activation -> argmax -> fire_type per pixel
5. per fire: keep the center-connected burn, classify crown change,
   drop fires that don't reach the treatment area
6. average kept fires onto the full raster (fp32) -> per-pixel change probabilities
7. write a 3-band float32 GeoTIFF (no change | to crown | from crown)
```

Output: `-o` if given, else `data/outputs/<model_name>/<model_name>.tif`. `make smoke` runs this against a sample treatment area. A run is capped at 2**16 ignitions.

Env: `RUN_DEVICE` (`cuda` default, `XLA`, `cpu`), `RUN_DTYPE` (default `bfloat16`), `USE_CLOUD_PATHS` (set for `gs://` paths), `BURN_EMULATOR_BACKEND` (`DL` default, `PT`). `PT` (or `-pt`) runs pyretechnics instead of the emulator and writes `model_<VARLOC>_pt_<data_version>.tif`.

## Inputs

```
topo_path/             # tifs: aspect / slope
baseline_fuels_path/   # tifs: cbd, cbh, cc, fbfm, th
legalmax_fuels_path/   # tifs: cbd, cbh, cc, fbfm, th
fbfm_map_path          # FBFM code -> behaviour lookup csv
ignitions_path         # ignition points; null samples them from treatment_area
burn_paths/{ignition_number}/fire_type.tif # training only
```

## Datasets

`VarLoc` (`burn_emulator.datasets.varloc`) is the dataset all configs use. It reads the fuel layers above, windows one context per ignition and yields:

| key | meaning |
| --- | --- |
| `x` | stacked input channels (topo + fuels + FBFM one-hots) |
| `y` | per-pixel `fire_type` target; train only |
| `wind` | ignition wind direction (degrees) |
| `mask` | burnable / circular-window mask |

Inference samples also carry `pdiffs` / `bounds` / `indxes` for stamping predictions back onto the full raster.

A fuel product with different layers gets a new `Dataset` class with the same `x` / `y` / `wind` / `mask` contract, so `train` / `evaluate` / `run` and `model.forward(x, wind)` work unchanged.

## Publish a model

```bash
burn_emulator -m bundle -c configs/varlocs/current.yaml -vl <varloc> [-cp <checkpoint>]   # make model-bundle
scripts/publish_model.sh <varloc> <bundle_dir> [models_uri]                               # make model-release
```

`-m bundle` writes `data/bundles/<model_name>/`:

| file | from |
| --- | --- |
| `model.pt` | `-cp`, else the lowest-loss checkpoint in `data/outputs/<model_name>/checkpoints` |
| `stats.yaml` | the same experiment dir |
| `config.yaml` | the merged config without runtime fields (treatment area, fuels, topo, fbfm map, ignitions, burns) |
| `bundle_meta.json` | `model_repo_sha`, `model_repo_dirty`, `model_class_path`, `model_code_sha256` (sha256 of the architecture module) |

`publish_model.sh` checks that all four files exist and that `model_name` matches `<varloc>`, then uploads to `<models_uri>/<varloc>/<model_version>/` and repoints `current`. `<model_version>` = `<model.pt mtime, YYYYMMDDTHHMMSSZ>-<7-char model_repo_sha>[-dirty]`. If that version is already published, a matching bundle only repoints `current`; a different one is refused unless `FORCE=1`. Forcing does not invalidate outputs cached under that version. `models_uri` defaults to `BURN_EMULATOR_MODELS_URI`.

The runner uses `bundle_meta.json` to warn when its architecture code differs from the bundle's; see [`burn-emulator-runner`](../burn-emulator-runner).

## Publish inputs (fuels + topo + varlocs + fbfm map)

```bash
scripts/publish_inputs.sh <data_version> <fuels_dir> <topo_dir> <varlocs_gpkg> <varlocs_txt> <fbfm_map> [inputs_uri]   # make inputs-release
# <data_version>  fuels date, YYYYMMDD (make default: current.yaml inputs_version)
# <fuels_dir>     {baseline,legalmax}/{cbd,cbh,cc,fbfm,th}.tif, e.g. data/training_data/fuels_20260824
# <topo_dir>      topo tifs
# <varlocs_gpkg>  varloc polygons, e.g. data/outputs/valid_varlocs_5070.gpkg (make valid-varlocs)
# <varlocs_txt>   the api's varloc allow-list, configs/varlocs/varlocs.txt
# <fbfm_map>      configs/fbfm_behavior_adjectives.csv
```

Uploads to `<inputs_uri>/<data_version>/`: `baseline/`, `legalmax/`, `topo/`, `varlocs/` (gpkg + txt) and `fbfm/`, then repoints `<inputs_uri>/current`. A failed upload leaves `current` unchanged. Layers already published are skipped unless `FORCE=1`. Republishing with `FORCE=1` does not invalidate cached outputs; publish changed inputs under a new `data_version`. `inputs_uri` defaults to `BURN_EMULATOR_INPUTS_URI`.

To update only the varlocs layer (e.g. after training a new varloc):

```bash
scripts/publish_varlocs.sh <varlocs_txt> <varlocs_gpkg> [data_version] [inputs_uri]   # make varlocs-release
```

Overwrites `varlocs/varlocs.txt` and the gpkg under a published `data_version` (default: `current`) and leaves `current` alone. The local gpkg name must match the published one.
