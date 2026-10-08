# burn-emulator-runner

Cloud Run Job. Runs `burn_emulator.run.run()` once per execution for a request picked by [`burn-emulator-api`](../burn-emulator-api), then exits. One image backs `burn-emulator-runner-gpu-<env>` (`DL`) and `burn-emulator-runner-cpu-<env>` (`PT`, `RUN_DEVICE=cpu`).

## Config

Job template:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_DEBUG` | log per-run timings |
| `RUN_DEVICE` | `cuda` (default) or `cpu` |

Per execution, set by the api:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_VARLOC` | varloc (requested, or picked by the api) |
| `BURN_EMULATOR_BACKEND` | `DL` or `PT` |
| `BURN_EMULATOR_MODEL_VERSION` | the varloc's bundle; always for DL, for PT when the varloc has one |
| `BURN_EMULATOR_BUNDLE_DIR` | with it: `/models/<varloc>/<model_version>` (GCS-FUSE) |
| `BURN_EMULATOR_BUNDLE_URI` | with it: `gs://` of that bundle, recorded in `meta.geojson` |
| `BURN_EMULATOR_INPUTS_VERSION` | recorded in `meta.geojson`; fills `run_smoke.yaml` for PT without a bundle |
| `BURN_EMULATOR_OUTPUT_META` | `<output_dir>/meta.geojson`: the treatment area in, the run meta out |
| `BURN_EMULATOR_HASH` | cache key |
| `BURN_EMULATOR_OUTPUT_PATH` | `<output_dir>/output.tif`, the cache hit |
| `BURN_EMULATOR_REPORT_PATH` | `gs://` `_reports/` object written when the run finishes |
| `BURN_EMULATOR_CLAIM_GENERATION` | echoed on the report |
| `BURN_EMULATOR_BASELINE_FUELS` | `/inputs/<inputs_version>/baseline` (GCS-FUSE) |
| `BURN_EMULATOR_LEGALMAX_FUELS` | `/inputs/<inputs_version>/legalmax` |
| `BURN_EMULATOR_TOPO_PATH` | `/inputs/<inputs_version>/topo` |
| `BURN_EMULATOR_FBFM_MAP_PATH` | `/inputs/<inputs_version>/fbfm/fbfm_behavior_adjectives.csv` |
| `BURN_EMULATOR_IGNITION_DENSITY` | optional; default: the bundle's `config.yaml` value, or `run_smoke.yaml`'s (20) without one |

## Model bundle

```
/models/<varloc>/current                 # active model_version (read by the api)
/models/<varloc>/<model_version>/
├── model.pt                             # checkpoint
├── stats.yaml                           # normalization stats
├── config.yaml                          # model_name, ckpt_path, model, activation, dataset, dataloader
└── bundle_meta.json                     # model_repo_sha, model_repo_dirty, model_class_path, model_code_sha256
```

DL only: a `model_code_sha256` that differs from this image's copy of `model_class_path`, or a missing `bundle_meta.json`, is logged as a warning; the run continues.

## Flow

```
1. download the treatment area from <output_meta>
2. with a bundle (DL, or PT on a trained varloc): load <bundle_dir>/config.yaml
   (the bundle dir is the experiment_dir; PT uses only its dataset config)
   PT without one: configs/varlocs/templates/run_smoke.yaml from the image, wind_range from configs/wind_directions.csv
3. inject treatment_area, fuels_paths, topo_path, fbfm_map_path, ignition_density, out_path
4. run(**config) -> a local tif under /tmp
5. overwrite <output_meta>, then upload the tif to <output_path> (the cache hit)
6. write the empty <report_path> object with metadata {status: completed|failed, claim_generation, error}
7. exit 0, or 1 on any error
```

SIGTERM before `run()` returns is a failure and still writes a report; after that, and while reporting a failure, it is ignored. `/tmp` is in-memory on Cloud Run and counts against the job memory. The runner job's max retries must stay 0.

## Output

```
<output_dir>/
├── meta.geojson             # api: the unioned treatment area in EPSG:5070; runner overwrites with one EPSG:5070 Feature,
│                            # geometry = unioned treatment area, properties = hash, backend, varloc,
│                            # inputs_version, ignition_density, output_path
│                            # + with a bundle: model_version, bundle_uri, every bundle_meta.json field
└── output.tif               # bands: no change, to crown, from crown
```

## Build

```bash
docker build -f burn-emulator-runner/Dockerfile -t burn-emulator-runner .   # from the repo root
```
