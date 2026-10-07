# burn-emulator

A deep-learning emulator of a burn spread model that generates thousands of burns in seconds for conditional burn probability (CBP).

## Components

[View architecture diagram](https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FOurPlanscape%2Fburn-emulator%2Fmain%2Fdiagram.drawio) ([source](diagram.drawio))

| Directory | What it is |
| --- | --- |
| [`burn-emulator-model/`](burn-emulator-model/README.md) | Model library + CLI: training data, train, evaluate, bundle and `run()` inference. |
| [`burn-emulator-api/`](burn-emulator-api/README.md) | Go service, no GPU: validates or picks the varloc, picks the backend, serves the GCS cache, triggers a runner job on a miss. |
| [`burn-emulator-runner/`](burn-emulator-runner/README.md) | Cloud Run Job (GPU for `DL`, CPU for `PT`): runs `burn_emulator.run.run()` once per execution. |

## Makefile targets

Run from the repo root. `make help` lists every target; `[metal]` runs locally, `[slurm]` submits to the cluster, `[metal|slurm]` submits with `SLURM=1`.

Images (need `BURN_EMULATOR_ARTIFACT_STORE`):

| target | what it does |
| --- | --- |
| `api-image` / `api-release` | build / build + push `burn-emulator-api:<git-sha>[-dirty]`; a dirty tree is refused when `BURN_EMULATOR_ENV=production` |
| `runner-image` / `runner-release` | same, for `burn-emulator-runner` |
| `images-release` | `api-release` + `runner-release` under one `VERSION`; deploy both together |

Training data and training:

| target | what it does |
| --- | --- |
| `training-data VARLOC= [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1]` | `burn_emulator -m ignite` into `data/training_data/<varloc>/<inputs_version>_<ignitions_version>/` (versions default to `current.yaml`, `NUM_IGNITIONS` to 5000); `OVERWRITE=1` replaces an existing directory |
| `training-data-all [EXCLUDE=] [...] [SLURM=1]` | `training-data` for every varloc in `data/training_data/varlocs.gpkg`, skipping complete ones (legalmax `outputs_table.csv`) unless `OVERWRITE=1`; exits nonzero listing failures; then runs `all-varlocs`. `SLURM=1` queues them for `slurm/ignitions.slurm` (`EXCLUDE` is slurm only) |
| `all-varlocs` | copies `data/training_data/varlocs.gpkg` to `data/outputs/all_varlocs.gpkg` |
| `train-all [SLURM=1 [EXCLUDE=] [NODES=]]` | trains every varloc with complete training data; adds each to `data/outputs/varlocs.txt` |
| `train-release VARLOC= [NODES=]` | slurm: train -> `model-bundle` -> `model-release` -> `mark_trained.sh` + `varlocs-release` |
| `train-release-all [EXCLUDE=] [NODES=]` | the same for every varloc with complete training data |
| `update-mark-trained` | rewrites `data/outputs/varlocs.txt` to the varlocs with an `epoch-<num_epochs - 1>` checkpoint for the current architecture + data version; errors instead of writing an empty list |

Releases (need `BURN_EMULATOR_MODELS_URI` / `BURN_EMULATOR_INPUTS_URI`):

| target | what it does |
| --- | --- |
| `valid-varlocs` | filters `data/outputs/all_varlocs.gpkg` to `data/outputs/varlocs.txt` -> `data/outputs/valid_varlocs.gpkg` |
| `model-bundle VARLOC=` | `burn_emulator -m bundle` -> `data/bundles/<model_name>/` |
| `model-bundle-all` | `update-mark-trained`, `valid-varlocs`, then `model-bundle` for every varloc in `varlocs.txt` |
| `model-release VARLOC= [BUNDLE_DIR=] [FORCE=1]` | `scripts/publish_model.sh`: uploads the bundle, repoints `<varloc>/current` |
| `model-release-all` | uploads every bundle in `varlocs.txt`, then repoints each `current` |
| `inputs-release [DATA_VERSION=] [FUELS_DIR=] [TOPO_DIR=] [FBFM_MAP=]` | `scripts/publish_inputs.sh`: fuels, topo, both varlocs gpkgs and the fbfm map under one `inputs_version`, then repoints `current`. Defaults follow `inputs_version` in `configs/varlocs/current.yaml` |
| `varlocs-release [DATA_VERSION=]` | `valid-varlocs`, then `scripts/publish_varlocs.sh` replaces both gpkgs under a published `inputs_version` (default `current`) |

Local runs:

| target | what it does |
| --- | --- |
| `smoke VARLOC= [WIND_RANGE="<lo> <hi>"] [TREATMENT_AREA=] [OUT_PATH=] [PT=1]` | `burn_emulator -m run -d` with `configs/varlocs/templates/run_smoke.yaml`; `WIND_RANGE` defaults to `configs/wind_directions.csv`, `TREATMENT_AREA` to `data/training_data/<varloc>/sample_pa.geojson` |
| `inference VARLOC= OUTPUTS_ROOT=` | `scripts/ignite_inference.sh`: evaluates every scenario under `OUTPUTS_ROOT` and saves each ignition's burn prediction |
| `inference-all OUTPUTS_ROOT=` | `inference` for every varloc in `varlocs.txt`, root `OUTPUTS_ROOT/<varloc>` |
| `shell` | syncs `burn-emulator-model/.venv-<arch>` (with the `data` extra) and opens a shell in it |

## Varloc files

Under `burn-emulator-model/`:

| file | contents | written by |
| --- | --- | --- |
| `data/training_data/varlocs.gpkg` | every varloc polygon, EPSG:5070 (source) | external |
| `data/outputs/all_varlocs.gpkg` | copy of the source | `all-varlocs` |
| `data/outputs/varlocs.txt` | trained varlocs | `mark_trained.sh`, `update-mark-trained` |
| `data/outputs/valid_varlocs.gpkg` | `all_varlocs.gpkg` filtered to `varlocs.txt` | `valid-varlocs` |

## Model registry (GCS)

```
gs://<models>/<varloc>/current                  # text file: the active model_version
gs://<models>/<varloc>/<model_version>/         # model.pt, stats.yaml, config.yaml, bundle_meta.json
```

`<model_version>` = `<model.pt mtime, YYYYMMDDTHHMMSSZ>-<7-char git sha>[-dirty]`.

## Request flow

```
caller --POST /v1/jobs {treatment_area [, varloc, job_name, ignition_density, backend]}--> burn-emulator-api
  1. validate the request
  2. inputs_version = gs://<inputs>/current                                   (60s cache)
     varloc         = requested varloc if it intersects treatment_area in
                      <inputs_version>/varlocs/all_varlocs.gpkg (else 400),
                      or the largest overlap there (none -> 400)
     DL if varloc is in valid_varlocs.gpkg and gs://<models>/<varloc>/current exists (60s cache),
     else PT
     hash           = sha256(treatment_area[|ignition_density]) + 0 (DL) | 1 (PT)
     out_path       = gs://<out>/<inputs_version>/<varloc>/<model_version | pt>/<hash>
  3. output exists?                                                      -> 200 cached
     claim running?                                                      -> 202 pending + Location
     else claim it, write <out_path>/treatment_area.geojson, trigger the GPU (DL) or CPU (PT) job
                                                                         -> 202 pending + Location
  4. runner: DL loads <varloc>/<model_version> from the models mount, PT uses no bundle;
     run(), upload the tif to <out_path>/, write gs://<out>/_reports/... {status, claim_generation, error}
  5. _reports/ OBJECT_FINALIZE -> Pub/Sub push -> POST /internal/pubsub/run-reports
     completed -> release the claim; failed -> delete partial output, mark the claim failed
  6. caller polls GET <Location> until cached or failed
```

## Calling service

The api rejects malformed GeoJSON, an unsupported `crs`, an unknown `varloc`, and an area outside the requested varloc or every varloc. The ignition cap (2**16 ignitions: `ignition_density` per km² over the buffered treatment area) is checked by the runner, after a job has started.
