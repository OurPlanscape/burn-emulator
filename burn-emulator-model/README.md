# burn-emulator-model

Model library + CLI. [`burn-emulator-runner`](../burn-emulator-runner) calls `burn_emulator.run.run()` for inference.

## Modelling

The default model is a CNN with [circular kernels](https://arxiv.org/pdf/2107.02451). The ignition point is the center pixel of the context window; the model predicts the spread shape from the local fuels, topo and wind. At inference the prediction is stamped across the landscape in parallel to compute conditional burn probability.

## Install

```bash
make shell   # from the repo root: syncs .venv-<arch> with the data extra and activates it
```

or, from `burn-emulator-model/`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-$(uname -m) uv sync              # library + CLI
UV_PROJECT_ENVIRONMENT=.venv-$(uname -m) uv sync --extra data # + pyretechnics/ray for -m ignite and PT
```

`scripts/` run from `burn-emulator-model/`.

## CLI

`burn_emulator -m <train|evaluate|run|bundle|ignite>`. `-c` configs merge in order (later wins); CLI flags override them. `-a` / `-vl` / `-dv` (or `architecture` / `varloc` / `data_version`) set `model_name` = `<VARLOC>_<architecture>_<data_version>` and the experiment dir `data/outputs/<model_name>/`. `configs/varlocs/current.yaml` holds the current `architecture`, `inputs_version` and `ignitions_version`; `data_version` = `<inputs_version>_<ignitions_version>`.

## Training data

```bash
burn_emulator -m ignite -vl <varloc> -dv <data_version> [-ni <num_ignitions>] [-ow]
```

Clips fuels and topo for the varloc (polygon from `data/training_data/varlocs.gpkg`) and simulates burns with pyretechnics into `data/training_data/<varloc>/<data_version>/`. Complete once `legalmax/outputs_table.csv` exists.

## Train

```bash
burn_emulator -m train -a <architecture> -vl <varloc> -dv <data_version> \
              -c configs/<architecture>/model.yaml -c configs/<architecture>/train.yaml -c configs/varlocs/templates/train.yaml
```

Writes `data/outputs/<model_name>/`: `checkpoints/` (`<model_name>_loss-<loss>_epoch-<NNNN>_step-<NNNNNN>.pt`, epochs from 0), `stats.yaml`, `train_log.csv`. Resumes from the latest checkpoint.

## Evaluate

```bash
burn_emulator -m evaluate -a <architecture> -vl <varloc> -dv <data_version> -c <model.yaml> -c <eval_data.yaml>
```

Writes per-ignition prediction GeoTIFFs and `throughput.csv` to `data/outputs/<model_name>/inference/`; no metrics.

```bash
scripts/ignite_inference.sh <varloc> <outputs_root>
```

Evaluates every scenario under `<outputs_root>` (one subdirectory per scenario, with `<N>_baseline` / `<N>_legalmax` ignition sets), using `configs/varlocs/current.yaml`.

## Run

```bash
burn_emulator -m run -vl <varloc> \
              -c configs/varlocs/current.yaml -c configs/<architecture>/model.yaml -c <data.yaml> \
              -bf <baseline_fuels> -lf <legalmax_fuels> -tp <topo> -mp <fbfm_map> \
              -ta <treatment_area> [-id <ignition_density>] [-cp <checkpoint>] [-o <out.tif>] [-pt]
```

```
1. merge -c configs + CLI overrides
2. VarLoc dataset: seeded ignitions over the buffered treatment_area, one window per ignition
   (baseline + collated-treatment fuels, wind, circular mask)
3. load the checkpoint (-cp, else the lowest-loss one in <experiment_dir>/checkpoints)
4. per batch: baseline + treatment forward -> activation -> argmax -> fire_type per pixel
5. per fire: keep the center-connected burn, classify crown change, drop fires outside the treatment area
6. average kept fires onto the full raster (fp32)
7. write a 3-band float32 GeoTIFF (no change | to crown | from crown)
```

Writes `-o`, else `data/outputs/<model_name>/<model_name>.tif`. Capped at 2**16 ignitions. `make smoke` runs it on a sample treatment area.

Env: `RUN_DEVICE` (`cuda` default, `XLA`, `cpu`), `RUN_DTYPE` (default `bfloat16`), `USE_CLOUD_PATHS` (for `gs://` paths), `BURN_EMULATOR_BACKEND` (`DL` default, `PT`). `PT` / `-pt` runs pyretechnics and writes `<VARLOC>_pt_<data_version>.tif`.

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

`VarLoc` (`burn_emulator.datasets.varloc`) windows one context per ignition and yields:

| key | meaning |
| --- | --- |
| `x` | input channels (topo + fuels + FBFM one-hots) |
| `y` | per-pixel `fire_type`; training only |
| `wind` | wind direction (degrees) |
| `mask` | burnable / circular-window mask |

Inference samples also carry `pdiffs` / `bounds` / `indxes` for stamping predictions onto the full raster. Another fuel product needs a `Dataset` with the same `x` / `y` / `wind` / `mask` keys.

## Publish a model

```bash
burn_emulator -m bundle -c configs/varlocs/current.yaml -vl <varloc> [-cp <checkpoint>]   # make model-bundle
scripts/publish_model.sh <varloc>=<bundle_dir> [<varloc>=<bundle_dir> ...]                # make model-release(-all)
```

`-m bundle` writes `data/bundles/<model_name>/`:

| file | from |
| --- | --- |
| `model.pt` | `-cp`, else the lowest-loss checkpoint |
| `stats.yaml` | the experiment dir |
| `config.yaml` | the merged config without run-time fields (treatment area, fuels, topo, fbfm map, ignitions, burns) |
| `bundle_meta.json` | `model_repo_sha`, `model_repo_dirty`, `model_class_path`, `model_code_sha256` (architecture module) |

`publish_model.sh` (needs `BURN_EMULATOR_MODELS_URI`) checks every bundle (four files, `model_name` matches `<varloc>`), uploads each to `<models_uri>/<varloc>/<model_version>/`, then repoints each `<varloc>/current`. `<model_version>` = `<model.pt mtime, YYYYMMDDTHHMMSSZ>-<7-char model_repo_sha>[-dirty]`. An already published version with a matching bundle is not re-uploaded; a different bundle is refused unless `FORCE=1`, which keeps outputs cached under that version.

## Publish inputs

```bash
scripts/publish_inputs.sh <data_version> <fuels_dir> <topo_dir> <valid_varlocs.gpkg> <all_varlocs.gpkg> <fbfm_map> [inputs_uri]   # make inputs-release
# <data_version>        fuels date, YYYYMMDD
# <fuels_dir>           {baseline,legalmax}/{cbd,cbh,cc,fbfm,th}.tif
# <topo_dir>            topo tifs
# <valid_varlocs.gpkg>  data/outputs/valid_varlocs.gpkg (make valid-varlocs)
# <all_varlocs.gpkg>    data/outputs/all_varlocs.gpkg (make all-varlocs)
# <fbfm_map>            configs/fbfm_behavior_adjectives.csv
```

Uploads `baseline/`, `legalmax/`, `topo/`, `varlocs/` (both gpkgs; the api reads them by name) and `fbfm/` to `<inputs_uri>/<data_version>/`, then repoints `<inputs_uri>/current`. A failed upload leaves `current` unchanged. Published layers are skipped unless `FORCE=1`, which keeps cached outputs; changed inputs go under a new `data_version`. `inputs_uri` defaults to `BURN_EMULATOR_INPUTS_URI`.

```bash
scripts/publish_varlocs.sh <valid_varlocs.gpkg> <all_varlocs.gpkg> [data_version] [inputs_uri]   # make varlocs-release
```

Replaces both gpkgs under a published `data_version` (default `current`); `current` is not repointed.
