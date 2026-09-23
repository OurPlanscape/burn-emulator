# burn-emulator-runner

GPU batch job. Loads a model bundle and runs `burn_emulator.run.run()` once per execution, then exits. Triggered only by [`burn-emulator-api`](../burn-emulator-api) as a Cloud Run Job execution.

## Config

Static, baked into the job template:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_DIR` | model registry mount point (default `/models`, GCS-FUSE) |
| `BURN_EMULATOR_DEBUG` | log + return per-run timings |

Per-execution, set by `burn-emulator-api` as job execution overrides:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_VARLOC` | varloc name |
| `BURN_EMULATOR_VERSION` | model version |
| `BURN_EMULATOR_TREATMENT_AREA` | geojson treatment area |
| `BURN_EMULATOR_TREATMENT_AREA_CRS` | treatment area CRS |
| `BURN_EMULATOR_HASH` | cache key for this request |
| `BURN_EMULATOR_OUTPUT_PATH` | `gs://` output prefix |
| `BURN_EMULATOR_BASELINE_FUELS` | baseline fuels, under the GCS-FUSE-mounted inputs bucket (`/inputs`) |
| `BURN_EMULATOR_LEGALMAX_FUELS` | treatment fuels, under `/inputs` |
| `BURN_EMULATOR_TOPO_PATH` | topo (aspect/slope), under `/inputs` |
| `BURN_EMULATOR_IGNITION_DENSITY` | optional; omit to use the value baked into the model bundle's `config.yaml` |

## Model bundle

```
<MODELS_DIR>/<varloc>/current       # text file: the active version
<MODELS_DIR>/<varloc>/<version>/
├── model.pt                     # checkpoint
├── stats.yaml                   # normalization stats
├── fbfm_behavior_adjectives.csv # FBFM code -> behaviour lookup
├── config.yaml                  # model + activation + dataset + dataloader + model_name
└── bundle_meta.json             # model-repo git sha + model_class_path + model_code_sha256
```

On each execution the runner compares `bundle_meta.json`'s `model_code_sha256` against its own sha256 of the architecture module file named by `model_class_path` (e.g. `burn_emulator/models/circlepp.py`) and **logs a warning** on a mismatch.

## Flow

```
1. read <MODELS_DIR>/<varloc>/<version>/config.yaml (bundle dir = experiment_dir)
2. warn if bundle_meta.json model_code_sha256 != this image's architecture module hash
3. inject treatment_area, fuels_paths, topo_path, out_path into config
4. run(**config) -> writes <output_path>/<model_name>_run.tif
5. exit 0 on success, exit 1 on any error (the execution/task is marked failed)
```

## Build

```bash
# from the repo root
docker build -f burn-emulator-runner/Dockerfile -t burn-emulator-runner .
```
