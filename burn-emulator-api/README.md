# burn-emulator-api

Go service. Validates a request, resolves the model and data versions, checks the GCS output cache and claim, and on a miss triggers a [`burn-emulator-runner`](../burn-emulator-runner) Cloud Run Job execution and returns `202 pending` right away with a `Location` to poll. The runner reports back by writing a `_reports/` report, which a GCS Pub/Sub notification pushes to `POST /internal/pubsub/run-reports`. No ML code; caller identity is verified upstream via GCP identities.

## Config

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `gs://` root of the model registry (reads `<varloc>/current`) |
| `BURN_EMULATOR_INPUTS_URI` | `gs://` root of the fuels/topo inputs (reads `current` for the `data_version`, written by `publish_inputs.sh`) |
| `BURN_EMULATOR_OUTPUT_URI` | `gs://` bucket for outputs, `_claims/` and `_reports/` |
| `BURN_EMULATOR_RUNNER_GPU_JOB` | fully-qualified GPU runner job name, used for `DL`: `projects/*/locations/*/jobs/*` |
| `BURN_EMULATOR_RUNNER_CPU_JOB` | fully-qualified CPU-only runner job name, used for `PT` |

## `POST /v1/jobs`

```json
{
  "varloc": "WC711",
  "treatment_area": "<geojson>",
  "treatment_area_crs": "EPSG:5070", # everything gets reprojected to this anyway
  "job_name": "my-run-01",
  "ignition_density": 20,
  "backend": "DL" # optional: "DL" (emulator, GPU job, default) | "PT" (pyretechnics, CPU-only job)
}
```

`varloc` must be in `gs://<inputs>/<data_version>/varlocs/varlocs.txt` for the current `data_version` (published by `publish_inputs.sh`, 60s cache), otherwise 400; `job_name` is 1-63 chars `[a-z0-9-]` and is only used for logging and recorded on the claim (it does not affect the hash); `backend` picks the runner job and is part of the hash: the hash is the sha256 hex plus a trailing `0` (DL) or `1` (PT); `ignition_density` is optional (ignitions per km², defaults to 20 ignitions per km²;  `VarLoc` converts internally); omit it to use the value baked into the model bundle's `config.yaml`. Worth noting here that the max number of ignitions is 2**16. Verify this by using area/density upstream somewhere.

```json
{
  "job_id": "WC711/20260829T143000Z-a1b2c3d/20260824/1a2b3c4d…",
  "job_name": "my-run-01",
  "hash": "1a2b3c4d…",
  "model_version": "20260829T143000Z-a1b2c3d", # this is from publish-model.sh in the model repo
  "backend": "DL",
  "data_version": "20260824", # west fuels date (fuels + topo + varlocs), from publish-inputs in the model repo
  "status": "pending",
  "varloc": "WC711",
  "cached": false,
  "attempts": 1, # omitted when 0 (cached); error is also set when status is failed
  "output_path": "gs://<bucket>/WC711/<model_version>/<data_version>/<hash>"
}
```

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | `<output_path>/<model_name>.tif` (DL) or `model_<VARLOC>_pt_<data_version>.tif` (PT) exists |
| `pending` | 202 | run triggered by this request, or an identical run already in flight; `Location: /v1/jobs/<job_id>` |

POST is idempotent: an identical body dedupes onto the same run. A previously `failed` run is retried.

Errors are plain text: `400` invalid JSON or field, `413` body over 1 MiB, `500` if version resolution, a GCS call or the runner trigger fails.

## `GET /v1/jobs/{varloc}/{model_version}/{data_version}/{hash}`

Read-only status of the run named by `job_id` (the `Location` from the POST). The id pins the model + data versions, so a version bump mid-run doesn't change what it resolves to. Same body as above.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | `<output_path>/<model_name>.tif` (DL) or `model_<VARLOC>_pt_<data_version>.tif` (PT) exists (same status as POST) |
| `pending` | 200 | run in flight |
| `failed` | 200 | the run failed (`error` is set), or its claim went stale without it reporting back; re-POST to retry |
| - | 404 | unknown id, or no output and no claim |
| - | 500 | a GCS lookup failed |

## `POST /internal/pubsub/run-reports`

Pub/Sub push endpoint only. Receives GCS `OBJECT_FINALIZE` notifications for `gs://<out>/_reports/...` reports written by the runner. The report's `status` and `claim_generation` metadata are re-read from GCS; a `completed` report (the runner's own status, not an API status) releases the claim, `failed` deletes any partial output and marks the claim `failed` so `GET` reports it. Any 2xx acks, 5xx asks Pub/Sub to redeliver.

## `GET /healthz` -> `200 ok`

## Flow

```
1. validate job_name
2. data_version  = gs://<inputs>/current                 (60s cache; fuels + topo + varlocs)
   varloc in gs://<inputs>/<data_version>/varlocs/varlocs.txt, else 400   (60s cache)
   model_version = gs://<models>/<varloc>/current       (60s cache)
   hash          = sha256(varloc + "|" + treatment_area + "|" + treatment_area_crs [+ "|" + ignition_density])
   out_path      = gs://<out>/<varloc>/<model_version>/<data_version>/<hash>
3. out_path exists?                                               -> 200 cached
   _claims/<varloc>/<model_version>/<data_version>/<hash> running?   -> 202 pending
   else: claim it (or reclaim a failed/stale one), trigger a runner job execution -> 202 pending
4. runner writes _reports/<varloc>/<model_version>/<data_version>/<hash>
   -> GCS notification -> Pub/Sub push -> POST /internal/pubsub/run-reports -> release or fail the claim
5. caller polls GET /v1/jobs/<job_id> until cached or failed
```

The claim is a zero-byte GCS object written with a generation precondition; it stops two identical concurrent requests both hitting the GPU. The claim's generation is passed to the runner and echoed back on its `_reports/` report, so a late notification can't touch a newer run's claim. A claim with no notification (runner killed before writing its report) is reclaimed after ~25 min. If triggering the runner fails, the claim is released only when Cloud Run clearly rejected the call (4xx); on a timeout, network error or 5xx an execution may have started anyway, so the claim is kept (POST returns 500, GET shows `pending`) rather than risk a duplicate GPU run, and goes stale after ~25 min if nothing was running.

## Output bucket layout

All three object kinds share the suffix `<varloc>/<model_version>/<data_version>/<hash>` (`dispatch.JobID.Path`, which is also the `job_id` in GET URLs), so each path derives from the others by adding or dropping the prefix. Each version is its own path segment, so no two version combinations can map to the same path.

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

Only `_reports/` triggers the Pub/Sub notification (`object_name_prefix` in infrastructure). Varlocs never start with `_`, so the bookkeeping prefixes can't collide with outputs.

## Timeouts

Built bottom-up: each layer is the sum of what it wraps plus a margin. The Go layers are computed in code, so changing an inner value moves the outer ones; only the Cloud Run and Pub/Sub values in infrastructure need updating by hand.

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

The Cloud Run timeouts and the Pub/Sub subscription are set within infrastructure.

## Build

```bash
go build -o burn-emulator-api ./cmd/server   # Go 1.26+
docker build -t burn-emulator-api .
```
