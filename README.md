# burn-emulator

A deep-learning emulator of a burn spread model to generate thousands of burns in seconds for conditional burn probability (CBP).

## Components

3 components in order to separate resource requirements and limit costs. The api requires no GPU resources and can serve identical requests straight from cache, only triggering the GPU runner job when necessary. The runner is a Cloud Run Job, so it incurs no idle cost between executions. See individual sub-repos for details.

[View architecture diagram](https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FOurPlanscape%2Fburn-emulator%2Fmain%2Fdiagram.drawio) ([source](diagram.drawio))

| Directory | What it is |
| --- | --- |
| [`burn-emulator-model/`](burn-emulator-model/README.md) | Model library + CLI: train, evaluate, and `run()` inference. |
| [`burn-emulator-api/`](burn-emulator-api/README.md) | Go service: validate -> resolve model + data versions -> GCS cache/dedupe -> trigger the runner job. |
| [`burn-emulator-runner/`](burn-emulator-runner/README.md) | GPU Cloud Run Job: runs `burn_emulator.run.run()` once per execution. |

## Makefile targets

Run from the repo root.

| target | what it does |
| --- | --- |
| `build-api` / `push-api` | build/push the api image, tagged `<BURN_EMULATOR_ARTIFACT_STORE>/burn-emulator-api:<git-sha>[-dirty]` |
| `build-runner` / `push-runner` | same, for the runner image |
| `model-bundle VARLOC=<varloc>` | wraps `burn_emulator -m bundle` for one varloc (in `burn-emulator-model/`) |
| `publish-model VARLOC=<varloc> [BUNDLE_DIR=<path>] [FORCE=1]` | uploads a bundle and repoints `current`; refuses to overwrite a published version with a different bundle unless `FORCE=1` |
| `model-bundle-all` / `publish-model-all` | same, looped over every varloc in `configs/varlocs/varlocs.txt`, stopping at the first failure; `publish-model-all` doesn't accept `BUNDLE_DIR` |
| `publish-inputs [DATA_VERSION=<YYYYMMDD>] [FUELS_DIR=<dir>] [TOPO_DIR=<dir>] [VARLOCS_GPKG=<gpkg>] [VARLOCS_TXT=<txt>]` | `DATA_VERSION` defaults to `inputs_version` in `configs/varlocs/current.yaml`, `FUELS_DIR` to `data/training_data/fuels_<DATA_VERSION>`, `TOPO_DIR` to `data/training_data/topo_<DATA_VERSION>` (what `-m ignite` clips from); uploads baseline/legalmax fuel tifs, topo tifs, the varlocs gpkg (default: `valid-varlocs` output) and the api's varloc allow-list (default: `configs/varlocs/varlocs.txt`) under one `data_version`, then repoints `current` |
| `publish-varlocs [VARLOCS_TXT=<txt>] [VARLOCS_GPKG=<gpkg>] [DATA_VERSION=<version>]` | runs `valid-varlocs` to rebuild the gpkg from the txt, then replaces only the `varlocs/` layer (txt default: `configs/varlocs/varlocs.txt`, gpkg default: `valid-varlocs` output) under an already published `data_version` (default: the one `current` points at); skips unchanged files, prints the varlocs added/removed, never repoints `current`; the api picks up the txt within 60s |
| `train-all [SLURM=1 [NODES=<n1>,<n2>]]` | wraps `burn-emulator-model/scripts/train_all.sh` (with `SLURM=1`, `burn-emulator-model/slurm/submit_train_all.sh` instead, spread across the GPU nodes): trains every varloc with complete training data for the current `data_version` and adds each to `configs/varlocs/varlocs.txt` only once its training succeeds |
| `inference VARLOC=<varloc> OUTPUTS_ROOT=<dir>` | wraps `burn-emulator-model/scripts/ignite_inference.sh` |
| `inference-all OUTPUTS_ROOT=<dir>` | `inference` looped over every varloc, using `OUTPUTS_ROOT/<varloc>` as each varloc's root |
| `training-data VARLOC=<varloc> [INPUTS_VERSION=<YYYYMMDD>] [IGNITIONS_VERSION=<YYYYMMDD>] [NUM_IGNITIONS=<n>] [OVERWRITE=1] [SLURM=1]` | wraps `burn_emulator -m ignite` to generate training data under `data_version` = `<INPUTS_VERSION>_<IGNITIONS_VERSION>`, each defaulting to `configs/varlocs/current.yaml`; `NUM_IGNITIONS` defaults to 5000 in `ignite.py`; refuses to run if `data/training_data/<varloc>/<data_version>` exists unless `OVERWRITE=1`, which deletes it first |
| `training-data-all [INPUTS_VERSION=<YYYYMMDD>] [IGNITIONS_VERSION=<YYYYMMDD>] [NUM_IGNITIONS=<n>] [OVERWRITE=1] [SLURM=1]` | with `SLURM=1` (also on `training-data`), `burn-emulator-model/slurm/submit_ignitions_all.sh` applies the same skip and submits a `slurm/ignitions.slurm` job array (exclusive on dragon03, calls `burn_emulator -m ignite` directly), one task per varloc named `ignitions_<varloc>` to run the same target on the cluster; `training-data` looped over every varloc in the varlocs gpkg, skipping ones whose training data for the current `data_version` is complete unless `OVERWRITE=1`, stopping at the first failure |
| `shell` | activates the model repo's venv and cds into it |

`build-*`/`push-*`/`publish-*` need `BURN_EMULATOR_ARTIFACT_STORE` / `BURN_EMULATOR_MODELS_URI` / `BURN_EMULATOR_INPUTS_URI` exported by the caller.

## Model registry (GCS)

```
gs://<bucket_name>/<models>/<varloc>/current       # text file: the active model version
gs://<bucket_name>/<models>/<varloc>/<model_version>/    # model.pt, stats.yaml, config.yaml
```

`<model_version>` = `<model.pt mtime>-<git sha>`. `model-bundle` builds a bundle, `publish-model` uploads it and repoints `current`.

## Request flow

```
caller --POST /v1/jobs {varloc, treatment_area, treatment_area_crs, job_name}--> burn-emulator-api
  1. validate job_name
  2. data_version  = read gs://<inputs>/current            (60s cache; fuels + topo + varlocs)
     varloc in gs://<inputs>/<data_version>/varlocs/varlocs.txt, else 400  (60s cache)
     model_version = read gs://<models>/<varloc>/current  (60s cache)
     hash          = sha256(varloc + "|" + treatment_area + "|" + treatment_area_crs [+ "|" + ignition_density])
     out_path      = gs://<out>/<varloc>/<model_version>/<data_version>/<hash>
  3. out_path exists?                                             -> 200 cached
     _claims/<varloc>/<model_version>/<data_version>/<hash> running?  -> 202 pending + Location
     else claim it (or reclaim a failed/stale one), then:
        --trigger a burn-emulator-runner job execution with {varloc, model_version, treatment_area, treatment_area_crs, hash, out_path, report_path, claim_generation, fuels/topo paths (from data_version) [, ignition_density]} as env overrides-->
     <-- 202 pending, Location: /v1/jobs/<varloc>/<model_version>/<data_version>/<hash>
          a. read the bundle's config.yaml from the FUSE-mounted registry
          b. inject treatment_area + fuels/topo paths + a local temp out_path into the config
          c. run burn_emulator.run.run(**config), then upload the tif to <out_path>/<model_name>.tif
          d. write gs://<out>/_reports/... report {status: completed|failed, claim_generation, error}
  4. GCS OBJECT_FINALIZE on _reports/ -> Pub/Sub push -> burn-emulator-api POST /internal/pubsub/run-reports
     completed -> drop the claim; failed -> delete partial output, mark the claim failed
  5. caller polls GET <Location> until cached or failed
```

## Notes for the calling service

Validation of the treatment area and ignition density is done in the -model scripts and a little bit in the API. This is fine for most cases but in the case that the calling service does not want to instantiate the run and incur GPU costs of the seconds for that check, validation of those values should be done upstream (e.g egregious # of ignitions, coverage area, etc.). The MAX number of ignitions is 2**16.
