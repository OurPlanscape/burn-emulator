# burn-emulator-api

Go service. Validates or picks the varloc, then picks the backend and model version for a request, checks the GCS output cache and claim, and on a miss triggers a [`burn-emulator-runner`](../burn-emulator-runner) job and returns `202 pending` with a `Location` to poll. Runner reports arrive through a GCS Pub/Sub notification on `POST /internal/pubsub/run-reports`. Callers are authenticated by Cloud Run IAM (`roles/run.invoker`). Request flow: [root README](../README.md#request-flow).

## Config

All required except `BURN_EMULATOR_RUNNER_GPU_JOB`; the server exits on startup if one is missing.

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `gs://` model registry root; reads `<varloc>/current` |
| `BURN_EMULATOR_INPUTS_URI` | `gs://` inputs root; reads `current` and `<inputs_version>/varlocs/*.gpkg` |
| `BURN_EMULATOR_OUTPUT_URI` | `gs://` bucket for outputs, `_claims/` and `_reports/` |
| `BURN_EMULATOR_RUNNER_GPU_JOB` | optional; `DL` runner job, `projects/*/locations/*/jobs/*`. Unset: `DL` requests run on PT |
| `BURN_EMULATOR_RUNNER_CPU_JOB` | `PT` runner job |

## `POST /v1/jobs`

```json
{
  "treatment_area": "<geojson>",
  "varloc": "WC711",
  "job_name": "my-run-01",
  "ignition_density": 20,
  "backend": "DL"
}
```

| field | |
| --- | --- |
| `treatment_area` | required; `Polygon` / `MultiPolygon` / `GeometryCollection` GeoJSON, bare or in a `Feature` / `FeatureCollection`; other geometry types are rejected, null geometries skipped, and all parts unioned. CRS from the top-level `crs` member (e.g. `"crs": {"type": "name", "properties": {"name": "EPSG:5070"}}`), rejected without one |
| `varloc` | optional; 1-32 alphanumeric chars, uppercased. Must be in `all_varlocs.gpkg` and intersect `treatment_area`; omitted: the largest overlap in `all_varlocs.gpkg` |
| `job_name` | optional; 1-63 chars `[a-z0-9-]`, starting and ending alphanumeric; stored on the claim, not hashed |
| `ignition_density` | optional; ignitions per km², > 0; default: the bundle's `config.yaml` value (DL) or 20 (PT). Capped at 2**16 ignitions per run, checked by the runner |
| `backend` | optional; `DL` (default) or `PT`. `DL` becomes `PT` when the varloc has no model (see Varloc selection); the response `backend` is what runs |

`hash` = sha256 hex of `treatment_area[|ignition_density]` + `0` (DL) or `1` (PT). `treatment_area` is re-encoded first (sorted keys, no whitespace); the runner gets the re-encoded copy.

```json
{
  "job_id": "20260824/WC711/20260829T143000Z-a1b2c3d/1a2b3c4d…",
  "job_name": "my-run-01",
  "hash": "1a2b3c4d…",
  "backend": "DL",
  "inputs_version": "20260824",
  "status": "cached",
  "cached": true,
  "varloc": "WC711",
  "model_version": "20260829T143000Z-a1b2c3d",
  "output_path": "gs://<bucket>/<inputs_version>/<varloc>/<model_version>/<hash>"
}
```

`model_version` is omitted for PT (`pt` in the job id). `attempts` is omitted when 0; `error` is set when `status` is `failed`.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | `<output_path>/<model_name>.tif` (DL) or `<VARLOC>_pt_<inputs_version>.tif` (PT) exists |
| `pending` | 202 | run triggered, or an identical run in flight; `Location: /v1/jobs/<job_id>` |

An identical body dedupes onto the same run; a `failed` run is retried.

Errors (plain text): `400` invalid JSON or field, unsupported `crs`, unknown `varloc`, or `treatment_area` outside the requested varloc / every varloc; `413` body over 1 MiB; `500` version resolution, GCS or runner trigger failure.

## `GET /v1/jobs/{inputs_version}/{varloc}/{model_version}/{hash}`

Status of the run named by `job_id`. Same body as POST.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | output exists |
| `pending` | 200 | run in flight |
| `failed` | 200 | the run failed (`error` is set) or its claim went stale; re-POST to retry |
| - | 404 | unknown id, or no output and no claim |
| - | 500 | GCS lookup failed |

## `POST /internal/pubsub/run-reports`

Pub/Sub push endpoint for GCS `OBJECT_FINALIZE` on `gs://<out>/_reports/`. Re-reads the report's `status` and `claim_generation` from GCS: `completed` releases the claim, `failed` deletes partial output and marks the claim `failed`. 2xx acks, 5xx redelivers.

## `GET /healthz` -> `200 ok`

## Varloc selection

`dispatch.selectVarLoc`, before the cache check:

Varloc, from `all_varlocs.gpkg`:

| request | varloc |
| --- | --- |
| `varloc` given, intersects `treatment_area` | that varloc |
| `varloc` given, not in the gpkg / no intersection | 400 |
| no `varloc` | largest overlap; none -> 400 |

Backend:

| `backend` | varloc in `valid_varlocs.gpkg` | `<models>/<varloc>/current` | runs |
| --- | --- | --- | --- |
| `DL`, no `BURN_EMULATOR_RUNNER_GPU_JOB` | - | - | PT (warning) |
| `DL` | yes | yes | DL with that `model_version` |
| `DL` | yes | no | PT (warning) |
| `DL` | no | - | PT (warning) |
| `PT` | - | - | PT |

Both gpkgs are cached per `inputs_version` and re-downloaded when their GCS generation changes (checked every 60s). They are read with `modernc.org/sqlite` and parsed with GEOS (`twpayne/go-geos`); the treatment area is reprojected into EPSG:5070 with PROJ (`twpayne/go-proj`). Overlap is the exact intersection area in EPSG:5070; ties go to the first varloc name. Accepted `crs`: CRS84 or any EPSG code known to PROJ. Invalid polygons are repaired with GEOS `MakeValid` (structure method); rings must be closed.

## Claims

A claim is a zero-byte GCS object written with a generation precondition; identical concurrent requests share one run. Its generation goes to the runner and back on the report; a report for an older generation leaves a newer claim alone. A claim without a report goes stale after 25 min (`dispatch.runStaleAfter`). A 4xx trigger failure releases the claim; a timeout, network error or 5xx keeps it (POST returns 500, GET shows `pending` until it goes stale).

## Output bucket layout

All objects share the suffix `<inputs_version>/<varloc>/<model_version | pt>/<hash>` (`dispatch.JobID.Path`, the `job_id`).

```
gs://<out>/
├── <job_id>/<model_name>.tif         # output (PT: <VARLOC>_pt_<inputs_version>.tif)
├── <job_id>/treatment_area.geojson   # runner input
├── _claims/<job_id>                  # claim
└── _reports/<job_id>                 # runner report
```

| Prefix | Written by | Metadata | Lifetime |
| --- | --- | --- | --- |
| `<job_id>/*.tif` | runner | none | permanent; its existence is the cache hit |
| `<job_id>/treatment_area.geojson` | api, before the trigger | none | deleted with the output on failure, rewritten on retry |
| `_claims/` | api | `status` (`running` \| `failed`), `job_name`, `attempts`, `updated_at`, `error` | deleted on success; a `failed` claim stays until the next POST reclaims it |
| `_reports/` | runner | `status` (`completed` \| `failed`), `claim_generation`, `error` | deleted once handled |

Only `_reports/` triggers the notification (`object_name_prefix` in infrastructure).

## Timeouts

| Rule | Values | Set in |
| --- | --- | --- |
| runner timeout < claim stale age | 20 min < 25 min | `burn_emulator_runner_timeout` (infra), `dispatch.runStaleAfter` |
| runner retries = 0 | 0 | `burn_emulator_runner_max_retries` (infra) |
| report worst case < Pub/Sub ack deadline | 90 s + 2 x 10 s < 120 s | `handlers.reportTimeout`, `dispatch.releaseTimeout`, `ack_deadline_seconds` (infra) |
| handler worst case < Cloud Run request timeout | 4 min < 300 s | `http.Server.WriteTimeout`, `burn_emulator_api_timeout` (infra) |

## Build

```bash
go build -o burn-emulator-api ./cmd/server   # Go 1.26+, cgo, libgeos-dev + libproj-dev
docker build -f burn-emulator-api/Dockerfile -t burn-emulator-api .   # from the repo root
```
