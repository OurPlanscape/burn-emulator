# burn-emulator-api

Go service. Validates a request, resolves the model and data versions, checks the GCS output cache and claim, and on a miss triggers a [`burn-emulator-runner`](../burn-emulator-runner) Cloud Run Job execution and returns `202 pending` with a `Location` to poll. The runner writes a `_reports/` report, which a GCS Pub/Sub notification pushes to `POST /internal/pubsub/run-reports`. Callers are authenticated by Cloud Run IAM (`roles/run.invoker`).

## Config

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `gs://` root of the model registry (reads `<varloc>/current`) |
| `BURN_EMULATOR_INPUTS_URI` | `gs://` root of the inputs (reads `current` for the `data_version`) |
| `BURN_EMULATOR_OUTPUT_URI` | `gs://` bucket for outputs, `_claims/` and `_reports/` |
| `BURN_EMULATOR_RUNNER_GPU_JOB` | GPU runner job, used for `DL`: `projects/*/locations/*/jobs/*` |
| `BURN_EMULATOR_RUNNER_CPU_JOB` | CPU-only runner job, used for `PT` |

All are required; the server exits on startup if one is missing.

## `POST /v1/jobs`

```json
{
  "varloc": "WC711",
  "treatment_area": "<geojson>",
  "job_name": "my-run-01",
  "ignition_density": 20,
  "backend": "DL"
}
```

| field | |
| --- | --- |
| `varloc` | required; must be in `gs://<inputs>/<data_version>/varlocs/varlocs.txt` for the current `data_version`, else 400 |
| `treatment_area` | required; geojson, reprojected to EPSG:5070 by the runner. Its CRS comes from the geojson `crs` member, e.g. `"crs": {"type": "name", "properties": {"name": "EPSG:5070"}}`; without one it is read as EPSG:4326 |
| `job_name` | required; 1-63 chars `[a-z0-9-]`, starting and ending alphanumeric; logged and stored on the claim, not part of the hash |
| `ignition_density` | optional; ignitions per km², > 0; omit to use the bundle's `config.yaml` value (20). A run is capped at 2**16 ignitions, checked by the runner |
| `backend` | optional; `DL` (emulator, GPU job, default) or `PT` (pyretechnics, CPU-only job) |

`hash` = sha256 hex of `varloc|treatment_area[|ignition_density]`, plus a trailing `0` (DL) or `1` (PT).

```json
{
  "job_id": "WC711/20260829T143000Z-a1b2c3d/20260824/1a2b3c4d…",
  "job_name": "my-run-01",
  "hash": "1a2b3c4d…",
  "model_version": "20260829T143000Z-a1b2c3d",
  "backend": "DL",
  "data_version": "20260824",
  "status": "pending",
  "varloc": "WC711",
  "cached": false,
  "attempts": 1,
  "output_path": "gs://<bucket>/WC711/<model_version>/<data_version>/<hash>"
}
```

`model_version` is set by `model-release`, `data_version` (the fuels date) by `inputs-release`. `attempts` is omitted when 0; `error` is set when `status` is `failed`.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | `<output_path>/<model_name>.tif` (DL) or `model_<VARLOC>_pt_<data_version>.tif` (PT) exists |
| `pending` | 202 | run triggered by this request, or an identical run already in flight; `Location: /v1/jobs/<job_id>` |

POST is idempotent: an identical body dedupes onto the same run. A previously `failed` run is retried.

Errors are plain text: `400` invalid JSON, field or varloc, `413` body over 1 MiB, `500` if version resolution, a GCS call or the runner trigger fails.

## `GET /v1/jobs/{varloc}/{model_version}/{data_version}/{hash}`

Read-only status of the run named by `job_id` (the `Location` from the POST). The id pins the model and data versions. Same body as above.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | `<output_path>/<model_name>.tif` (DL) or `model_<VARLOC>_pt_<data_version>.tif` (PT) exists (same status as POST) |
| `pending` | 200 | run in flight |
| `failed` | 200 | the run failed (`error` is set), or its claim went stale without it reporting back; re-POST to retry |
| - | 404 | unknown id, or no output and no claim |
| - | 500 | a GCS lookup failed |

## `POST /internal/pubsub/run-reports`

Pub/Sub push endpoint. Receives GCS `OBJECT_FINALIZE` notifications for `gs://<out>/_reports/...`. The report's `status` and `claim_generation` metadata are re-read from GCS: `completed` releases the claim, `failed` deletes any partial output and marks the claim `failed`. 2xx acks, 5xx makes Pub/Sub redeliver.

## `GET /healthz` -> `200 ok`

## Flow

```
1. validate the request
2. data_version  = gs://<inputs>/current                                   (60s cache)
   varloc in gs://<inputs>/<data_version>/varlocs/varlocs.txt, else 400     (60s cache)
   model_version = gs://<models>/<varloc>/current                          (60s cache)
   hash          = sha256(varloc|treatment_area[|ignition_density]) + 0 (DL) | 1 (PT)
   out_path      = gs://<out>/<varloc>/<model_version>/<data_version>/<hash>
3. out_path exists?                                               -> 200 cached
   _claims/<varloc>/<model_version>/<data_version>/<hash> running?   -> 202 pending
   else: claim it (or reclaim a failed/stale one), trigger the GPU (DL) or CPU (PT) runner job -> 202 pending
4. runner writes _reports/<varloc>/<model_version>/<data_version>/<hash>
   -> GCS notification -> Pub/Sub push -> POST /internal/pubsub/run-reports -> release or fail the claim
5. caller polls GET /v1/jobs/<job_id> until cached or failed
```

The claim is a zero-byte GCS object written with a generation precondition, so identical concurrent requests share one run. Its generation is passed to the runner and echoed on the report; a report for an older generation does not touch a newer claim. A claim with no report is reclaimed after 25 min (`dispatch.runStaleAfter`). If the trigger fails with a 4xx the claim is released; on a timeout, network error or 5xx it is kept (POST returns 500, GET shows `pending`) and goes stale after 25 min.

## Output bucket layout

All three object kinds share the suffix `<varloc>/<model_version>/<data_version>/<hash>` (`dispatch.JobID.Path`, also the `job_id` in GET URLs).

```
gs://<out>/
├── <varloc>/<model_version>/<data_version>/<hash>/<model_name>.tif   # output (PT: model_<VARLOC>_pt_<data_version>.tif)
├── _claims/<varloc>/<model_version>/<data_version>/<hash>               # claim
└── _reports/<varloc>/<model_version>/<data_version>/<hash>               # runner report
```

| Prefix | Written by | Metadata | Lifetime |
| --- | --- | --- | --- |
| `<varloc>/...` | runner (upload after `run()`) | none | permanent; its existence is the cache hit |
| `_claims/` | api (`claimRun`, `failRun`) | `status` (`running` \| `failed`), `job_name`, `attempts`, `updated_at`, `error` | deleted on success (by the notification, or by the next POST/GET cache hit if the completed report was lost); a `failed` claim stays as a record until the next POST reclaims it |
| `_reports/` | runner (`_write_report`) | `status` (`completed` \| `failed`), `claim_generation`, `error` | seconds; deleted by the api once the notification is handled |

Only `_reports/` triggers the Pub/Sub notification (`object_name_prefix` in infrastructure).

## Timeouts

Each layer is the sum of what it wraps plus a margin. The Go values are derived in code; the Cloud Run and Pub/Sub values are set in infrastructure and must be updated by hand.

| Layer | Value | Built from |
| --- | --- | --- |
| `dispatch.releaseTimeout` | 10 s | one detached GCS cleanup call (claim release/check, output delete) |
| `dispatch.claimTriggerTimeout` | 90 s | claim + runner job trigger, detached from the caller |
| `dispatch.DetachedBudget` | 1m40s | `claimTriggerTimeout` + `releaseTimeout` (release after a failed trigger) |
| `handlers.requestTimeout` | 2 min | caller-attached steps: version resolution + cache check (POST), output + claim lookup (GET) |
| `handlers.MaxHandlerDuration` | 3m40s | `requestTimeout` + `DetachedBudget` |
| `main.go` `http.Server.WriteTimeout` | 4 min | `MaxHandlerDuration` + 20 s |
| `burn_emulator_api_timeout` (Cloud Run) | 300 s | `WriteTimeout` + 60 s |
| `handlers.reportTimeout` | 90 s | `POST /internal/pubsub/run-reports`; worst case ~110 s with 2 x `releaseTimeout` |
| `ack_deadline_seconds` (Pub/Sub) | 120 s | above the report worst case |
| `burn_emulator_runner_timeout` (Cloud Run) | 20 min | bounds each runner job execution; keep below `dispatch.runStaleAfter` (25 min) |

## Build

```bash
go build -o burn-emulator-api ./cmd/server   # Go 1.26+
docker build -t burn-emulator-api .
```
