# burn-emulator-api

Go service. Validates a request, resolves the model version, checks the GCS output cache and claim ledger, and on a miss triggers a [`burn-emulator-runner`](../burn-emulator-runner) Cloud Run Job execution and waits for it. No ML code; caller identity is verified upstream via GCP identities.

## Config

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `gs://` root of the model registry (reads `<varloc>/current`) |
| `BURN_EMULATOR_INPUTS_URI` | `gs://` root of the fuels/topo inputs (reads `fuels/current`, `topo/current`) |
| `BURN_EMULATOR_OUTPUT_URI` | `gs://` bucket for outputs + the claim ledger |
| `BURN_EMULATOR_RUNNER_JOB` | fully-qualified runner job name: `projects/*/locations/*/jobs/*` |
| `VARLOCS_FILE` | varloc allow-list (default `configs/varlocs.txt`) |

## `POST /v1/jobs`

```json
{
  "varloc": "WC711",
  "treatment_area": "<geojson>",
  "treatment_area_crs": "EPSG:4326",
  "job_name": "my-run-01",
  "ignition_density": 20
}
```

`varloc` must be in `burn-emulator-model/configs/varlocs/varlocs.txt`; `job_name` is 1-63 chars `[a-z0-9-]` but is never used and is only there for logging purposes; `ignition_density` is optional (ignitions per km², defaults to 20ignitions per km²;  `VarLoc` converts internally); omit it to use the value baked into the model bundle's `config.yaml`. Worth noting here that the max number of ignitions is 2**16. Verify this by using area/density upstream somewhere.

```json
{
  "job_name": "my-run-01",
  "hash": "1a2b3c4d…",
  "model_version": "20260829T1430Z-a1b2c3d", # this is from publish-model.sh in the model repo
  "fuels_version": "20260824",
  "topo_version": "20260824",
  "status": "completed",
  "varloc": "WC711",
  "cached": false,
  "output_path": "gs://<bucket>/WC711/<version>/<fuels_version>-<topo_version>/<hash>"
}
```

| `status` | HTTP | meaning |
| --- | --- | --- |
| `completed` | 200 | run finished; `<output_path>/<model_name>.tif` written |
| `cached` | 200 | output already existed |
| `pending` | 202 | identical run in flight, retry |

## `GET /healthz` -> `200 ok`

## Flow

```
1. validate varloc + job_name
2. version       = gs://<models>/<varloc>/current       (60s cache)
   fuels_version = gs://<inputs>/fuels/current           (60s cache)
   topo_version  = gs://<inputs>/topo/current            (60s cache)
   hash          = sha256(varloc + "|" + treatment_area + "|" + treatment_area_crs [+ "|" + ignition_density])
   out_path      = gs://<out>/<varloc>/<version>/<fuels_version>-<topo_version>/<hash>
3. out_path exists?                                          -> 200 cached
   _runs/<version>/<fuels_version>-<topo_version>/<hash> claimed? -> 202 pending
   else: claim it, trigger a runner job execution, wait, drop the claim -> 200 completed
```

The claim ledger is a zero-byte GCS object written with a generation precondition; it stops two identical concurrent requests both hitting the GPU. A stale claim is reclaimed after ~35 min.

## Timeouts

Each layer must budget at least as long as the one it wraps, outermost last. Preliminary tests show smaller varlocs (1000 hectares) run in a few seconds and fits well within window. Larger treatments may be an issue. Hopefully they don't take 20 minutes to run. It may be useful to split a large project area and parallelize that way.

| Layer | Value | Description |
| --- | --- | --- |
| `dispatch.runBudget` | 25 min | trigger the runner job execution + wait for it |
| `handlers.requestTimeout` | 30 min | run budget + GCS calls |
| `main.go` `http.Server.WriteTimeout` | 31 min |
| `burn_emulator_api_timeout` (Cloud Run) | 35 min | bounds on api |
| `burn_emulator_runner_timeout` (Cloud Run) | 20 min | bounds each runner job execution |

The two cloud run timeouts are set within infrastructure.

## Build

```bash
go build -o burn-emulator-api ./cmd/server   # Go 1.26+
docker build -t burn-emulator-api .
```
