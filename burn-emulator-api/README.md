# burn-emulator-api

Go service. Validates a request, resolves the model version, checks the GCS output cache and claim ledger, and on a miss runs the emulation synchronously on [`burn-emulator-runner`](../burn-emulator-runner). No ML code; caller identity is verified upstream via GCP identities.

## Config

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `gs://` root of the model registry (reads `<varloc>/current`) |
| `BURN_EMULATOR_OUTPUT_BUCKET` | `gs://` bucket for outputs + the claim ledger |
| `BURN_EMULATOR_RUNNER_URL` | runner base URL (and ID-token audience) |
| `VARLOCS_FILE` | varloc allow-list (default `configs/varlocs.yaml`) |

## `POST /v1/jobs`

```json
{ "varloc": "wc711", "treatment_area": "<geojson>", "job_name": "my-run-01" }
```

`varloc` must be in `varlocs.yaml`; `job_name` is 1–63 chars `[a-z0-9-]`.

```json
{
  "job_name": "my-run-01",
  "hash": "1a2b3c4d…",
  "model_version": "20260829T1430Z-a1b2c3d",
  "status": "completed",
  "varloc": "wc711",
  "cached": false,
  "output_path": "gs://<bucket>/wc711/<version>/<hash>"
}
```

| `status` | HTTP | meaning |
| --- | --- | --- |
| `completed` | 200 | run finished; `<output_path>/<model_name>_run.tif` written |
| `cached` | 200 | output already existed |
| `pending` | 202 | identical run in flight — retry |

## `GET /healthz` — `200 ok`

## Flow

```
1. validate varloc + job_name
2. version  = gs://<models>/<varloc>/current            (60s cache)
   hash     = sha256(varloc + "|" + treatment_area)
   out_path = gs://<out>/<varloc>/<version>/<hash>
3. out_path exists?                -> 200 cached
   _runs/<version>/<hash> claimed? -> 202 pending
   else: claim it, POST /infer to the runner, wait, drop the claim -> 200 completed
```

The claim ledger is a zero-byte GCS object written with a generation precondition — it stops two identical concurrent requests both hitting the GPU. A stale claim is reclaimed after ~3 min.

## Build

```bash
go build -o burn-emulator-api ./cmd/server   # Go 1.26+
docker build -t burn-emulator-api .
```
