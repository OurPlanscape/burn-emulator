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

`burn_emulator -m <train|evaluate|run|bundle|ignite>`. `-c` configs merge in order (later wins); CLI flags override them. `-a` / `-vl` / `-dv` (or `architecture` / `varloc` / `data_version`) set `model_name` = `<VARLOC>_<architecture>_<data_version>` and the experiment dir `data/outputs/<VARLOC>/<data_version>_<architecture>/` (`<experiment_dir>` below). `configs/varlocs/current.yaml` holds the current `architecture`, `inputs_version` and `ignitions_version`; `data_version` = `<inputs_version>_<ignitions_version>`.

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
8. compute the indirect (outside) and direct (inside) benefit of the treatment area
```

Writes `-o`, else `<experiment_dir>/<model_name>.tif`. Capped at 2**16 ignitions. `run()` returns `out_path` plus `project_area_acres`, `net_indirect_crown_acres_avoided`, `net_indirect_benefit_pct`, `significant_indirect_acres_reduced`, `significant_indirect_benefit_pct` and the `direct` equivalents (inside the treatment area) (defined in [meta.geojson](../burn-emulator-runner/README.md#metageojson)), unrounded. `-d` prints them to 3 significant figures with the timings. `make smoke` runs it on a sample treatment area.

`run.enable_caching(max_models)` makes later runs in the same process read each raster whole once and slice windows from memory (`burn_emulator.rasters.open_raster`), and keep up to `max_models` `(ckpt_path, model)` pairs loaded on `RUN_DEVICE`, least recently used evicted first. The runner's claim loop turns it on.

Env: `RUN_DEVICE` (`cuda` default, `XLA`, `cpu`), `RUN_DTYPE` (default `bfloat16`), `USE_CLOUD_PATHS` (for `gs://` paths), `BURN_EMULATOR_BACKEND` (`DL` default, `PT`). `PT` / `-pt` runs pyretechnics and writes `<VARLOC>_pt_<data_version>.tif`.

## Training data

```bash
burn_emulator -m ignite -vl <varloc> -dv <data_version> [-ni <num_ignitions>] [-ow]
```

Clips fuels and topo for the varloc (polygon from `data/inputs/<inputs_version>/varlocs/all_varlocs.gpkg`) and simulates burns with pyretechnics into `data/training_data/<varloc>/<data_version>/`. Complete once `legalmax/outputs_table.csv` exists.

## Train

```bash
burn_emulator -m train -a <architecture> -vl <varloc> -dv <data_version> \
              -c configs/<architecture>/model.yaml -c configs/<architecture>/train.yaml -c configs/varlocs/templates/train.yaml
```

Writes `<experiment_dir>/`: `checkpoints/` (`<model_name>_loss-<loss>_epoch-<NNNN>_step-<NNNNNN>.pt`, epochs from 0), `stats.yaml`, `train_log.csv`. Resumes from the latest checkpoint.

## Evaluate

```bash
burn_emulator -m evaluate -a <architecture> -vl <varloc> -dv <data_version> -c <model.yaml> -c <eval_data.yaml>
```

Writes per-ignition prediction GeoTIFFs and `throughput.csv` to `<experiment_dir>/inference/`.

```bash
scripts/ignite_inference.sh <varloc> <outputs_root>
```

Evaluates every scenario under `<outputs_root>` (one subdirectory per scenario, with `<N>_baseline` / `<N>_legalmax` ignition sets), using `configs/varlocs/current.yaml`.

## Publish a model

```bash
burn_emulator -m bundle -c configs/varlocs/current.yaml -vl <varloc> [-cp <checkpoint>]   # make model-bundle
scripts/publish_model.sh <varloc> [<varloc> ...]                                           # make model-release(-all)
```

`-m bundle` writes `data/bundles/<VARLOC>/<model_version>/` and points `data/bundles/<VARLOC>/current` at it, the models bucket's layout:

| file | from |
| --- | --- |
| `model.pt` | `-cp`, else the lowest-loss checkpoint |
| `stats.yaml` | the experiment dir |
| `config.yaml` | the merged config without run-time fields (treatment area, fuels, topo, fbfm map, ignitions, burns) |
| `bundle_meta.json` | `model_repo_sha`, `model_repo_dirty`, `model_class_path`, `model_code_sha256` (architecture module) |

`<model_version>` = `<checkpoint mtime, UTC YYYYMMDDTHHMMSSZ>-<7-char git sha>[-dirty]`.

`publish_model.sh` reads each `<VARLOC>/current` under `BURN_EMULATOR_BUNDLES_ROOT` (default `data/bundles`), checks the bundle, uploads it to `<BURN_EMULATOR_MODELS_URI>/<VARLOC>/<model_version>/` and repoints the bucket's `<VARLOC>/current`. `FORCE=1` overwrites a published version. `POINTER_ONLY=1` checks the local bundle but only repoints `<VARLOC>/current`: enough for metal DL, which reads bundles on prem, but Cloud Run PT runs on that varloc need the bundle in the bucket.

The runner's claim loop runs bundles from `data/bundles`.

## Publish inputs

```bash
scripts/set_inputs_current.sh <inputs_root> <data_version>                                   # make inputs-current
scripts/publish_inputs.sh <inputs_root> [inputs_uri]                                         # make inputs-release
scripts/publish_varlocs.sh <inputs_root> <valid_varlocs.gpkg> <all_varlocs.gpkg> [inputs_uri]   # make varlocs-release
# <inputs_root>   data/inputs
# <data_version>  fuels date, YYYYMMDD
```

`data/inputs/<data_version>/` is made by hand in the inputs bucket's layout:

```
baseline/{cbd,cbh,cc,fbfm,th}.tif
legalmax/{cbd,cbh,cc,fbfm,th}.tif
topo/{aspect,slope_degrees}.tif
varlocs/all_varlocs.gpkg              # every varloc polygon
fbfm/fbfm_behavior_adjectives.csv
varlocs/valid_varlocs.gpkg            # make valid-varlocs: all_varlocs filtered to varlocs.txt
```

`set_inputs_current.sh` checks the layout and points `<inputs_root>/current` at the version. `publish_inputs.sh` uploads `current`'s version to `<inputs_uri>/<data_version>/` and repoints `<inputs_uri>/current`. `FORCE=1` overwrites published layers.

`publish_varlocs.sh` uploads the two gpkgs (`make varlocs-release`: `current`'s `varlocs/`, rebuilt by `valid-varlocs`) to the varlocs layer of `current`'s published version. `inputs_uri` defaults to `BURN_EMULATOR_INPUTS_URI`.

## Environment

The publish scripts and slurm jobs source [`scripts/load_env.sh`](scripts/load_env.sh), which exports `~/.config/burn-emulator/env` (`make env-vars` at the repo root; `BURN_EMULATOR_ENV_FILE` to override): `BURN_EMULATOR_*` and `GOOGLE_APPLICATION_CREDENTIALS`.

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

Inference samples also carry `pdiffs` / `bounds` / `indxes` for stamping predictions onto the full raster.
