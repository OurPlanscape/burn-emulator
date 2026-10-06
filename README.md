# burn-emulator

A deep-learning emulator of a burn spread model that generates thousands of burns in seconds for conditional burn probability (CBP).

## Components

The api runs without a GPU, serves repeated requests from the GCS cache and only triggers a runner job on a cache miss. The runner is a Cloud Run Job (GPU for `DL`, CPU-only for `PT`), so it has no idle cost.

[View architecture diagram](https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FOurPlanscape%2Fburn-emulator%2Fmain%2Fdiagram.drawio) ([source](diagram.drawio))

| Directory | What it is |
| --- | --- |
| [`burn-emulator-model/`](burn-emulator-model/README.md) | Model library + CLI: training data, train, evaluate, bundle and `run()` inference. |
| [`burn-emulator-api/`](burn-emulator-api/README.md) | Go service: validate -> resolve model + data versions -> GCS cache/claim -> trigger a runner job. |
| [`burn-emulator-runner/`](burn-emulator-runner/README.md) | Cloud Run Job: runs `burn_emulator.run.run()` once per execution. |

## Makefile targets

Run from the repo root. `make help` lists every target; `[metal]` runs locally, `[slurm]` submits to the cluster, `[metal|slurm]` submits with `SLURM=1`.

| target | what it does |
| --- | --- |
| `api-image` / `api-release` | build / build + push `<BURN_EMULATOR_ARTIFACT_STORE>/burn-emulator-api:<git-sha>[-dirty]`; refuses a dirty tree when `BURN_EMULATOR_ENV=production` |
| `runner-image` / `runner-release` | same, for `burn-emulator-runner` |
| `valid-varlocs [VARLOCS_TXT=<txt>] [VARLOCS_GPKG=<gpkg>]` | filters the varlocs gpkg to the varlocs in `configs/varlocs/varlocs.txt`, writes `data/outputs/valid_varlocs.gpkg` |
| `model-bundle VARLOC=<varloc>` | `burn_emulator -m bundle` for one varloc, writes `data/bundles/<model_name>/` |
| `model-release VARLOC=<varloc> [BUNDLE_DIR=<path>] [FORCE=1]` | `scripts/publish_model.sh`: uploads the bundle and repoints `current`; a different bundle under an already published version is refused unless `FORCE=1` |
| `model-bundle-all` / `model-release-all` | the above for every varloc in `configs/varlocs/varlocs.txt`, stopping at the first failure; `model-bundle-all` runs `valid-varlocs` first and errors on an empty list |
| `inputs-release [DATA_VERSION=<YYYYMMDD>] [FUELS_DIR=<dir>] [TOPO_DIR=<dir>] [VARLOCS_GPKG=<gpkg>] [VARLOCS_TXT=<txt>] [FBFM_MAP=<csv>]` | `scripts/publish_inputs.sh`: uploads baseline/legalmax fuels, topo, varlocs gpkg + txt and the fbfm map under one `data_version`, then repoints `current`. Defaults: `DATA_VERSION` = `inputs_version` in `configs/varlocs/current.yaml`, `FUELS_DIR` = `data/training_data/fuels_<DATA_VERSION>`, `TOPO_DIR` = `data/training_data/topo_<DATA_VERSION>`, `FBFM_MAP` = `configs/fbfm_behavior_adjectives.csv` |
| `varlocs-release [VARLOCS_TXT=<txt>] [VARLOCS_GPKG=<gpkg>] [DATA_VERSION=<YYYYMMDD>]` | runs `valid-varlocs`, then `scripts/publish_varlocs.sh` replaces only the `varlocs/` layer of a published `data_version` (default: `current`); skips unchanged files, prints varlocs added/removed, never repoints `current`; the api picks up the txt within 60s |
| `train-all [SLURM=1 [EXCLUDE=<v1>,<v2>] [NODES=<n1>,<n2>]]` | trains every varloc with complete training data for the current `data_version` (`scripts/train_all.sh`, or `slurm/submit_train_all.sh` with `SLURM=1`); adds each to `configs/varlocs/varlocs.txt` once its training succeeds; `EXCLUDE` needs `SLURM=1` |
| `train-release VARLOC=<varloc> [NODES=<n1>,<n2>]` | slurm: train -> `model-bundle` -> `model-release` -> add to `varlocs.txt` + `varlocs-release`, for one varloc |
| `train-release-all [EXCLUDE=<v1>,<v2>] [NODES=<n1>,<n2>]` | the same for every varloc with complete training data |
| `inference VARLOC=<varloc> OUTPUTS_ROOT=<dir>` | `scripts/ignite_inference.sh` |
| `inference-all OUTPUTS_ROOT=<dir>` | `inference` for every varloc in `varlocs.txt`, with `OUTPUTS_ROOT/<varloc>` as each root |
| `smoke VARLOC=<varloc> [WIND_RANGE="<lo> <hi>"] [TREATMENT_AREA=<path>] [OUT_PATH=<tif>] [PT=1]` | `burn_emulator -m run -d` with `configs/varlocs/templates/run_smoke.yaml`; `WIND_RANGE` defaults to the varloc's row in `configs/wind_directions.csv`, `TREATMENT_AREA` to `data/training_data/<varloc>/sample_pa.geojson`; `PT=1` runs pyretechnics |
| `training-data VARLOC=<varloc> [INPUTS_VERSION=<YYYYMMDD>] [IGNITIONS_VERSION=<YYYYMMDD>] [NUM_IGNITIONS=<n>] [OVERWRITE=1] [SLURM=1]` | `burn_emulator -m ignite` into `data/training_data/<varloc>/<INPUTS_VERSION>_<IGNITIONS_VERSION>/` (versions default to `current.yaml`, `NUM_IGNITIONS` to 5000); refuses an existing directory unless `OVERWRITE=1`, which deletes it first |
| `training-data-all [EXCLUDE=<v1>,<v2>] [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1]` | `training-data` for every varloc in the varlocs gpkg, skipping complete ones (legalmax `outputs_table.csv`) unless `OVERWRITE=1`; continues past failures and exits nonzero listing them. With `SLURM=1`, `slurm/submit_ignitions_all.sh` queues them for one `slurm/ignitions.slurm` worker on dragon03 (`EXCLUDE` is slurm only) |
| `shell` | syncs the model venv (`burn-emulator-model/.venv-<arch>`, with the `data` extra) and opens a shell with it activated |

`*-image` / `*-release` need `BURN_EMULATOR_ARTIFACT_STORE` / `BURN_EMULATOR_MODELS_URI` / `BURN_EMULATOR_INPUTS_URI` exported.

## Model registry (GCS)

```
gs://<models>/<varloc>/current                  # text file: the active model_version
gs://<models>/<varloc>/<model_version>/         # model.pt, stats.yaml, config.yaml, bundle_meta.json
```

`<model_version>` = `<model.pt mtime, YYYYMMDDTHHMMSSZ>-<7-char git sha>[-dirty]`.

## Request flow

```
caller --POST /v1/jobs {varloc, treatment_area, job_name [, ignition_density, backend]}--> burn-emulator-api
  1. validate the request
  2. data_version  = read gs://<inputs>/current                                  (60s cache)
     varloc in gs://<inputs>/<data_version>/varlocs/varlocs.txt, else 400        (60s cache)
     model_version = read gs://<models>/<varloc>/current                         (60s cache)
     hash          = sha256(varloc|treatment_area[|ignition_density]) + 0 (DL) | 1 (PT)
     out_path      = gs://<out>/<varloc>/<model_version>/<data_version>/<hash>
  3. output exists?                                                     -> 200 cached
     _claims/<varloc>/<model_version>/<data_version>/<hash> running?    -> 202 pending + Location
     else claim it (or reclaim a failed/stale one), trigger a runner job execution
       (GPU job for DL, CPU job for PT) with the request + input paths as env overrides -> 202 pending + Location
  4. runner: load the bundle from the GCS-FUSE models mount, run(), upload the tif to <out_path>/,
     write gs://<out>/_reports/<varloc>/<model_version>/<data_version>/<hash> {status, claim_generation, error}
  5. GCS OBJECT_FINALIZE on _reports/ -> Pub/Sub push -> POST /internal/pubsub/run-reports
     completed -> release the claim; failed -> delete partial output, mark the claim failed
  6. caller polls GET <Location> until cached or failed
```

## Notes for the calling service

The api only validates request shape. Treatment area and ignition density are validated inside the runner, after a job has started. Validate them upstream to avoid paying for a run that fails: a run is capped at 2**16 ignitions (`ignition_density` per km² over the buffered treatment area).
