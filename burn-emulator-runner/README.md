# burn-emulator-runner

Cloud Run Job. Loads a model bundle and runs `burn_emulator.run.run()` once per execution, then exits. Triggered by [`burn-emulator-api`](../burn-emulator-api). One image backs two jobs: `burn-emulator-runner-gpu-<env>` (`DL`) and `burn-emulator-runner-cpu-<env>` (`PT`, `RUN_DEVICE=cpu`).

## Config

Static, baked into the job template:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_DIR` | model registry mount point (default `/models`, GCS-FUSE) |
| `BURN_EMULATOR_DEBUG` | log per-run timings |
| `RUN_DEVICE` | `cuda` (default) or `cpu` (CPU job) |

Per-execution, set by `burn-emulator-api` as job execution overrides:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_VARLOC` | varloc name |
| `BURN_EMULATOR_MODEL_VERSION` | model version |
| `BURN_EMULATOR_TREATMENT_AREA` | geojson treatment area |
| `BURN_EMULATOR_HASH` | cache key for this request |
| `BURN_EMULATOR_BACKEND` | `DL` (deep learning emulator, GPU job) or `PT` (pyretechnics, CPU-only job) |
| `BURN_EMULATOR_OUTPUT_PATH` | `gs://` output prefix |
| `BURN_EMULATOR_REPORT_PATH` | `gs://` path of the `_reports/` report written when the run finishes |
| `BURN_EMULATOR_CLAIM_GENERATION` | api claim generation, echoed back on the report |
| `BURN_EMULATOR_BASELINE_FUELS` | baseline fuels, `/inputs/<data_version>/baseline` on the GCS-FUSE-mounted inputs bucket |
| `BURN_EMULATOR_LEGALMAX_FUELS` | treatment fuels, `/inputs/<data_version>/legalmax` |
| `BURN_EMULATOR_TOPO_PATH` | topo (aspect/slope), `/inputs/<data_version>/topo` |
| `BURN_EMULATOR_FBFM_MAP_PATH` | FBFM code -> behaviour lookup, `/inputs/<data_version>/fbfm/fbfm_behavior_adjectives.csv` |
| `BURN_EMULATOR_IGNITION_DENSITY` | optional; omit to use the bundle's `config.yaml` value |

## Model bundle

```
<MODELS_DIR>/<varloc>/current       # text file: the active model version
<MODELS_DIR>/<varloc>/<model_version>/
├── model.pt                     # checkpoint
├── stats.yaml                   # normalization stats
├── config.yaml                  # model_name + ckpt_path + model + activation + dataset + dataloader
└── bundle_meta.json             # model-repo git sha + model_class_path + model_code_sha256
```

On each execution the runner compares `bundle_meta.json`'s `model_code_sha256` with the sha256 of its own copy of the module named by `model_class_path` (e.g. `burn_emulator/models/circlepp.py`) and logs a warning on a mismatch or a missing `bundle_meta.json`. It does not stop the run.

## Flow

```
1. read <MODELS_DIR>/<varloc>/<model_version>/config.yaml (the bundle dir is the experiment_dir)
2. warn if bundle_meta.json model_code_sha256 != this image's architecture module hash
3. inject treatment_area, fuels_paths, topo_path, fbfm_map_path, ignition_density, out_path into the config
4. run(**config) -> local temp <model_name>.tif (DL) or model_<VARLOC>_pt_<data_version>.tif (PT)
5. upload it to <output_path>/<file name>
6. write the empty <report_path> object with metadata {status: completed|failed, claim_generation, error}
7. exit 0 on success, 1 on any error
```

SIGTERM before `run()` returns becomes a failure, so a report is still written. After `run()` returns, and while reporting a failure, SIGTERM is ignored.

The temp raster lives in `/tmp`, which on Cloud Run is in-memory and counts against the job memory (`burn_emulator_runner_gpu_ram` or `burn_emulator_runner_cpu_ram`).

`burn_emulator_runner_max_retries` must stay 0: each failed attempt writes a `failed` report, and the api marks the claim `failed` while the retry is still running.

## Build

```bash
# from the repo root
docker build -f burn-emulator-runner/Dockerfile -t burn-emulator-runner .
```
