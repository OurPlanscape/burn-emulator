# burn-emulator-model

Model library + CLI. `burn_emulator.run.run()` is the inference entry point that [`burn-emulator-runner`](../burn-emulator-runner) imports.

## Modelling

The default model is an archetypal CNN with one major adjustment: [circular kernels](https://arxiv.org/pdf/2107.02451). Very few objects in nature are 'square-shaped', so a square kernel either wastes learning capacity or forces the model to learn a mapping from that arbitrary shape to the expected shape of the burn. Burns also spread in a quasi-circular pattern that doesn't align with grid anisotropy. The ignition point is fixed at the center pixel of the CNN context window, so the model only has to learn the mapping from local input connectivities (fuels, etc.) to a spread shape from that point — it doesn't need to know *where* a burn will occur, just a fuzzy approximation of what it will look like. That approximation is then stamped across the landscape in parallel to compute conditional burn probability.

## Install

```bash
uv sync && uv pip install -e .
uv sync --extra data     # for scripts/ (training-data generation from Pyretechnics)
```

## Train

```bash
burn_emulator -m train -c <model.yaml> -c <train.yaml>
burn_emulator -m test  -c <model.yaml> -c <test.yaml>
```

Output: `data/outputs/<model_name>/` — `checkpoints/`, `stat.yaml`, `train_log.csv`.

## Run

```bash
burn_emulator -m run -c <model.yaml> -c <data.yaml> \
  -bf <baseline_fuels> -lf <legalmax_fuels> -tp <topo> -ta <treatment_area>
```

```
1. merge -c YAMLs + CLI overrides
2. build the VarLoc dataset: sample seeded ignitions across the buffered treatment_area,
   one window per ignition (baseline + collated-treatment fuels, wind, circular mask)
3. load the checkpoint (-p, else lowest-loss in <dir>/checkpoints)
4. per batch: forward baseline + treatment -> activation -> argmax -> fire_type per pixel
5. per fire: keep the center-connected burn, classify crown change, drop fires that
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
baseline_fuels_path/   # tifs: baseline fuels
legalmax_fuels_path/   # tifs: treatment fuels
burn_paths/{ignition_number}/fire_type.tif
```

## Publish a model

```bash
# make sure here to match model name and varloc
burn_emulator -m bundle -c configs/circle_net_bundle_template.yaml -mn <model_name> -vl <varloc>
make publish-model VARLOC=<varloc>
```

`-m bundle` writes `data/bundles/<varloc>/`:

| file | from |
| --- | --- |
| `model.pt` | `-p`, else the lowest-loss checkpoint under `data/outputs/<model_name>/checkpoints` |
| `stat.yaml` | the same training dir |
| `config.yaml` | the `-c` config with runtime-injected fields removed |

`make publish-model` uploads it to `gs://<models>/<varloc>/<version>/` and repoints `current`. The runner injects `treatment_area` / `fuels_paths` / `topo_path` / `ignitions_path` at request time.
