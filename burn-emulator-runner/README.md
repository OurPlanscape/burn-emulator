# burn-emulator-runner

Cloud Run Job. Runs `burn_emulator.run.run()` once per execution for a request picked by [`burn-emulator-api`](../burn-emulator-api), then exits. One image backs `burn-emulator-runner-gpu-<env>` (`DL`) and `burn-emulator-runner-cpu-<env>` (`PT`, `RUN_DEVICE=cpu`).

## Config

Job template:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_DIR` | model registry mount (default `/models`, GCS-FUSE) |
| `BURN_EMULATOR_DEBUG` | log per-run timings |
| `RUN_DEVICE` | `cuda` (default) or `cpu` |

Per execution, set by the api:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_VARLOC` | varloc (requested, or picked by the api) |
| `BURN_EMULATOR_BACKEND` | `DL` or `PT` |
| `BURN_EMULATOR_MODEL_VERSION` | DL only: bundle `<MODELS_DIR>/<varloc>/<model_version>/` |
| `BURN_EMULATOR_INPUTS_VERSION` | PT output name, `<VARLOC>_pt_<inputs_version>.tif` |
| `BURN_EMULATOR_TREATMENT_AREA_PATH` | `<output_path>/treatment_area.geojson` |
| `BURN_EMULATOR_HASH` | cache key |
| `BURN_EMULATOR_OUTPUT_PATH` | `gs://` output prefix |
| `BURN_EMULATOR_REPORT_PATH` | `gs://` `_reports/` object written when the run finishes |
| `BURN_EMULATOR_CLAIM_GENERATION` | echoed on the report |
| `BURN_EMULATOR_BASELINE_FUELS` | `/inputs/<inputs_version>/baseline` (GCS-FUSE) |
| `BURN_EMULATOR_LEGALMAX_FUELS` | `/inputs/<inputs_version>/legalmax` |
| `BURN_EMULATOR_TOPO_PATH` | `/inputs/<inputs_version>/topo` |
| `BURN_EMULATOR_FBFM_MAP_PATH` | `/inputs/<inputs_version>/fbfm/fbfm_behavior_adjectives.csv` |
| `BURN_EMULATOR_IGNITION_DENSITY` | optional; default: the bundle's `config.yaml` value (DL) or 20 (PT) |

## Model bundle

```
<MODELS_DIR>/<varloc>/current            # active model_version (read by the api)
<MODELS_DIR>/<varloc>/<model_version>/
├── model.pt                             # checkpoint
├── stats.yaml                           # normalization stats
├── config.yaml                          # model_name, ckpt_path, model, activation, dataset, dataloader
└── bundle_meta.json                     # model_repo_sha, model_class_path, model_code_sha256
```

A `model_code_sha256` that differs from this image's copy of `model_class_path`, or a missing `bundle_meta.json`, is logged as a warning; the run continues.

## Flow

```
1. download the treatment area
2. DL: load <MODELS_DIR>/<varloc>/<model_version>/config.yaml (the bundle dir is the experiment_dir)
   PT: no bundle; run_smoke.yaml dataset defaults, wind_range from configs/wind_directions.csv
3. inject treatment_area, fuels_paths, topo_path, fbfm_map_path, ignition_density, out_path
4. run(**config) -> /tmp/<model_name>.tif (DL) or /tmp/<VARLOC>_pt_<inputs_version>.tif (PT)
5. upload it to <output_path>/
6. write the empty <report_path> object with metadata {status: completed|failed, claim_generation, error}
7. exit 0, or 1 on any error
```

SIGTERM before `run()` returns is a failure and still writes a report; after that, and while reporting a failure, it is ignored. `/tmp` is in-memory on Cloud Run and counts against the job memory. `burn_emulator_runner_max_retries` must stay 0.

## Build

```bash
docker build -f burn-emulator-runner/Dockerfile -t burn-emulator-runner .   # from the repo root
```
