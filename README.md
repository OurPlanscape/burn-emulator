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
| `bundle-model VARLOC=<varloc>` | wraps `burn_emulator -m bundle` for one varloc (in `burn-emulator-model/`) |
| `publish-model VARLOC=<varloc> [BUNDLE_DIR=<path>]` | uploads a bundle and repoints `current` |
| `bundle-model-all` / `publish-model-all` | same, looped over every varloc in `configs/varlocs/varlocs.txt`, stopping at the first failure; `publish-model-all` doesn't accept `BUNDLE_DIR` |
| `publish-inputs DATA_VERSION=<version> FUELS_DIR=<dir> TOPO_DIR=<dir>` | uploads baseline/legalmax fuel tifs and topo tifs under one `data_version`, then repoints `current` |
| `train-all` | wraps `burn-emulator-model/scripts/train_varlocs.sh` |
| `ignitions VARLOC=<varloc> OUTPUTS_ROOT=<dir>` | wraps `burn-emulator-model/scripts/ignite_inference.sh` |
| `shell` | activates the model repo's venv and cds into it |

`build-*`/`push-*`/`publish-*` need `BURN_EMULATOR_ARTIFACT_STORE` / `BURN_EMULATOR_MODELS_URI` / `BURN_EMULATOR_INPUTS_URI` exported by the caller.

## Model registry (GCS)

```
gs://<bucket_name>/<models>/<varloc>/current       # text file: the active model version
gs://<bucket_name>/<models>/<varloc>/<model_version>/    # model.pt, stats.yaml, config.yaml
```

`<model_version>` = `<model.pt mtime>-<git sha>`. `bundle-model` builds a bundle, `publish-model` uploads it and repoints `current`.

## Request flow

```
caller --POST /v1/jobs {varloc, treatment_area, treatment_area_crs, job_name}--> burn-emulator-api
  1. validate varloc + job_name
  2. model_version = read gs://<models>/<varloc>/current  (60s cache)
     data_version  = read gs://<inputs>/current            (60s cache; fuels + topo)
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
