# burn-emulator

A deep-learning emulator of a burn spread model to generate thousands of burns in seconds for conditional burn probability (CBP).

## Components

| Directory | What it is |
| --- | --- |
| [`burn-emulator-model/`](burn-emulator-model/README.md) | Model library + CLI: train, evaluate, and `run()` inference. |
| [`burn-emulator-api/`](burn-emulator-api/README.md) | Go service: validate -> resolve model version -> GCS cache/dedupe -> call the runner. |
| [`burn-emulator-runner/`](burn-emulator-runner/README.md) | GPU service: runs `burn_emulator.run.run()` with bounded concurrency. |

## Model registry (GCS)

```
gs://<bucket_name>/<models>/<varloc>/current       # text file: the active version
gs://<bucket_name>/<models>/<varloc>/<version>/    # model.pt, stats.yaml, config.yaml
```

`<version>` = `<model.pt mtime>-<git sha>`. `make publish-model VARLOC=…` uploads a bundle and repoints `current`.

## Request flow

```
caller --POST /v1/jobs {varloc, treatment_area, job_name}--> burn-emulator-api
  1. validate varloc + job_name
  2. version  = read gs://<models>/<varloc>/current        (60s cache)
     hash     = sha256(varloc + "|" + treatment_area)
     out_path = gs://<out>/<varloc>/<version>/<hash>
  3. out_path exists?                -> 200 cached
     _runs/<version>/<hash> claimed? -> 202 pending
     else claim it, then:
        --POST /infer {varloc, version, treatment_area, hash, out_path}--> burn-emulator-runner
          a. read the bundle from the FUSE-mounted registry, merge its *.yaml
          b. inject treatment_area + fuels/topo paths + out_path into the config
          c. run burn_emulator.run.run(**config) -> writes <out_path>/<model_name>_run.tif
          <-- 200 {status: completed, timing}
     drop the claim
  4. <-- 200 {status: completed, hash, model_version, output_path}
```

## Docker

```bash
make build-api
make build-runner
```
