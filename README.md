# burn-emulator

A deep-learning emulator of a burn spread model to generate thousands of burns in seconds for conditional burn probability (CBP).

## Components

3 components in order to separate resource requirements and limit costs. The api requires no GPU resources and can serve identical requests straight from cache, only triggering the GPU runner job when necessary. The runner is a Cloud Run Job, so it incurs no idle cost between executions. See individual sub-repos for details.

| Directory | What it is |
| --- | --- |
| [`burn-emulator-model/`](burn-emulator-model/README.md) | Model library + CLI: train, evaluate, and `run()` inference. |
| [`burn-emulator-api/`](burn-emulator-api/README.md) | Go service: validate -> resolve model version -> GCS cache/dedupe -> trigger the runner job. |
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
| `publish-fuels FUELS_DIR=<dir>` | uploads baseline/legalmax fuel tifs |
| `publish-topo TOPO_DIR=<dir>` | uploads topo tifs |
| `train-all` | wraps `burn-emulator-model/scripts/train_varlocs.sh` |
| `ignitions VARLOC=<varloc> OUTPUTS_ROOT=<dir>` | wraps `burn-emulator-model/scripts/ignite_inference.sh` |
| `shell` | activates the model repo's venv and cds into it |

`build-*`/`push-*`/`publish-*` need `BURN_EMULATOR_ARTIFACT_STORE` / `BURN_EMULATOR_MODELS_URI` / `BURN_EMULATOR_FUELS_URI` exported by the caller.

## Model registry (GCS)

```
gs://<bucket_name>/<models>/<varloc>/current       # text file: the active version
gs://<bucket_name>/<models>/<varloc>/<version>/    # model.pt, stats.yaml, config.yaml
```

`<version>` = `<model.pt mtime>-<git sha>`. `bundle-model` builds a bundle, `publish-model` uploads it and repoints `current`.

## Request flow

```
caller --POST /v1/jobs {varloc, treatment_area, treatment_area_crs, job_name}--> burn-emulator-api
  1. validate varloc + job_name
  2. version       = read gs://<models>/<varloc>/current  (60s cache)
     fuels_version = read gs://<inputs>/fuels/current      (60s cache)
     topo_version  = read gs://<inputs>/topo/current       (60s cache)
     hash          = sha256(varloc + "|" + treatment_area + "|" + treatment_area_crs [+ "|" + ignition_density])
     out_path      = gs://<out>/<varloc>/<version>/<fuels_version>-<topo_version>/<hash>
  3. out_path exists?                                             -> 200 cached
     _runs/<version>/<fuels_version>-<topo_version>/<hash> claimed? -> 202 pending
     else claim it, then:
        --trigger a burn-emulator-runner job execution with {varloc, version, treatment_area, treatment_area_crs, hash, out_path, fuels/topo paths (from fuels_version/topo_version) [, ignition_density]} as env overrides, wait for it-->
          a. read the bundle from the FUSE-mounted registry, merge its *.yaml
          b. inject treatment_area + fuels/topo paths + out_path into the config
          c. run burn_emulator.run.run(**config) -> writes <out_path>/<model_name>_run.tif
          d. exit 0, or exit 1 on failure
     drop the claim
  4. <-- 200 {status: completed, hash, model_version, output_path}
```

## Notes for the calling service

Validation of the treatment area and ignition density is done in the -model scripts and a little bit in the API. This is fine for most cases but in the case that the calling service does not want to instantiate the run and incur GPU costs of the seconds for that check, validation of those values should be done upstream (e.g egregious # of ignitions, coverage area, etc.). The MAX number of ignitions is 2**16.
