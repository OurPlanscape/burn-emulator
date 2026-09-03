# burn-emulator-runner

GPU service. Loads a model bundle per request and runs `burn_emulator.run.run()`, capping concurrent GPU work. Called only by [`burn-emulator-api`](../burn-emulator-api).

## Config

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_DIR` | model registry mount point (default `/models`, GCS-FUSE) |
| `BURN_EMULATOR_BASELINE_FUELS` | `gs://` baseline fuels |
| `BURN_EMULATOR_LEGALMAX_FUELS` | `gs://` treatment fuels |
| `BURN_EMULATOR_TOPO_PATH` | `gs://` topo (aspect/slope) |
| `BURN_EMULATOR_GPU_SLOTS` | concurrent `run()` calls (default `1`) |
| `BURN_EMULATOR_DEBUG` | log + return per-run timings |
| `PORT` | listen port (default `8080`) |

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

On `/infer` the runner compares `bundle_meta.json`'s `model_code_sha256` against its own sha256 of the architecture module file named by `model_class_path` (e.g. `burn_emulator/models/circlepp.py`) and **logs a warning** on a mismatch.

## `POST /infer`

```json
{ "varloc": "WC711", "version": "<version>", "treatment_area": "<geojson>",
  "hash": "1a2b3c4d…", "output_path": "gs://<bucket>/WC711/<version>/<hash>" }
```

-> `200 { "status": "completed", "output_path": "…", "timing": {…}|null }`

```
1. read <MODELS_DIR>/<varloc>/<version>/, merge its *.yaml
2. warn if bundle_meta.json model_code_sha256 != this image's architecture module hash
3. inject treatment_area, fuels_paths, topo_path, out_path into the dataset config
4. acquire a GPU slot (Semaphore(GPU_SLOTS))
5. run(**config) -> writes <output_path>/<model_name>_run.tif
```

## `GET /healthz` — `200`

## Build

```bash
# from the repo root
docker build -f burn-emulator-runner/Dockerfile -t burn-emulator-runner .
```
