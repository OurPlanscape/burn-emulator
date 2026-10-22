# burn-emulator-api

Go service. Validates or picks the varloc, then picks the backend and model version for a request, checks the GCS output cache and claim, and on a miss triggers a [`burn-emulator-runner`](../burn-emulator-runner) job and returns `202 pending` with a `Location` to poll. Runner reports arrive through a GCS Pub/Sub notification on `POST /internal/pubsub/run-reports`. Callers are authenticated by Cloud Run IAM (`roles/run.invoker`). Request flow: [root README](../README.md#request-flow).

## Config

`BURN_EMULATOR_RUNNER_GPU_JOB` and `BURN_EMULATOR_METAL` are optional and the rest are required; the server exits on startup when a required one is missing. With `BURN_EMULATOR_METAL` set, the server reads the three bucket variables.

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_MODELS_URI` | `gs://` model registry bucket; reads `<varloc>/current`, sends the runner `<uri>/<varloc>/<model_version>` as `bundle_uri`. A bare bucket: the runner mounts it at `/models` |
| `BURN_EMULATOR_INPUTS_URI` | `gs://` inputs bucket; reads `current` and `<inputs_version>/varlocs/*.gpkg`. A bare bucket: the runner mounts it at `/inputs` |
| `BURN_EMULATOR_OUTPUT_URI` | `gs://` bucket for outputs, `_claims/` and `_reports/` |
| `BURN_EMULATOR_RUNNER_GPU_JOB` | optional; `DL` runner job, `projects/*/locations/*/jobs/*`. Unset: `DL` requests run on PT |
| `BURN_EMULATOR_RUNNER_CPU_JOB` | `PT` runner job |
| `BURN_EMULATOR_METAL` | boolean; the [Dockerfile](Dockerfile) sets it to `1`. Claims are written `queued` with the runner env as their body, and the [runner's claim loop](#metal-dispatch) runs them on the cluster. `DL` runs for every varloc with a bundle |

## `POST /v1/jobs`

```json
{
  "treatment_area": {
    "type": "Polygon",
    "crs": {"type": "name", "properties": {"name": "EPSG:5070"}},
    "coordinates": [[[-2100000, 2050000], [-2099000, 2050000], [-2099000, 2051000], [-2100000, 2051000], [-2100000, 2050000]]]
  },
  "varloc": "WC711",
  "job_name": "my-run-01",
  "ignition_density": 20,
  "backend": "DL"
}
```

| field | |
| --- | --- |
| `treatment_area` | required; a GeoJSON object (not a string): `Polygon` / `MultiPolygon` / `GeometryCollection` GeoJSON, bare or in a `Feature` / `FeatureCollection`; other geometry types are rejected, null geometries skipped, and all parts unioned. CRS from the top-level `crs` member (e.g. `"crs": {"type": "name", "properties": {"name": "EPSG:5070"}}`), EPSG:4326 without one; reprojected to EPSG:5070 |
| `varloc` | optional; 1-32 alphanumeric chars, uppercased. Must be in `all_varlocs.gpkg` and intersect `treatment_area`; omitted: the largest overlap in `all_varlocs.gpkg` |
| `job_name` | optional; 1-63 chars `[a-z0-9-]`, starting and ending alphanumeric; stored on the claim, not hashed |
| `ignition_density` | optional; ignitions per km², > 0; default: the bundle's `config.yaml` value, or `run_smoke.yaml`'s (20) for PT without a bundle. Capped at 2**16 ignitions per run, checked by the runner |
| `backend` | optional; `DL` (default) or `PT`. `DL` becomes `PT` when the varloc has no bundle (see Varloc selection); the response `backend` is what runs |

`hash` = sha256 hex of `treatment_area[|ignition_density]` + `0` (DL) or `1` (PT). `treatment_area` is re-encoded first (sorted keys, no whitespace). The runner gets it reprojected to EPSG:5070 and unioned, as a one-Feature FeatureCollection with a `crs` member.

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
  "output_dir": "gs://<bucket>/<inputs_version>/<varloc>/<model_version>/<hash>",
  "output_path": "gs://<bucket>/<inputs_version>/<varloc>/<model_version>/<hash>/output.tif",
  "output_meta": "gs://<bucket>/<inputs_version>/<varloc>/<model_version>/<hash>/meta.geojson"
}
```

`model_version` is the bundle used, omitted for PT without one (`pt` in the job id; `pt-<model_version>` with one). `attempts` is omitted when 0; `error` is set when `status` is `failed`.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | `output_path` exists; `output_meta` is the treatment area with the run and its bundle as properties |
| `pending` | 202 | run triggered, or an identical run in flight; `Location: /v1/jobs/<job_id>` |

An identical body dedupes onto the same run; a `failed` run is retried.

Errors (plain text): `400` invalid JSON or field, unsupported `crs`, unknown `varloc`, or `treatment_area` outside the requested varloc / every varloc; `413` body over 1 MiB; `500` version resolution, GCS or runner trigger failure.

## `GET /v1/jobs/{inputs_version}/{varloc}/{model_version}/{hash}`

Status of the run named by `job_id`. Same body as POST.

| `status` | HTTP | meaning |
| --- | --- | --- |
| `cached` | 200 | output exists |
| `pending` | 200 | run in flight |
| `failed` | 200 | the run failed (`error` is set) or its claim went stale; re-POST to retry. Read from `_failed/` once the claim is archived |
| - | 404 | unknown id: output, claim and archived claim are all absent |
| - | 500 | GCS lookup failed |

## `POST /internal/pubsub/run-reports`

Pub/Sub push endpoint for GCS `OBJECT_FINALIZE` on `gs://<out>/_reports/`. Re-reads the report's `status` and `claim_generation` from GCS: `completed` releases the claim, `failed` deletes partial output and marks the claim `failed`. 2xx acks, 5xx redelivers.

Afterwards, at most once a day across all instances (`_prune/last` rewritten with a generation precondition), claims last updated over a day ago (`failed`, or active and still waiting for a report) are copied to `_failed/<job_id>` and deleted (generation-matched). `dispatch.maybePrune`, before the response; bounded by 30 s and 500 claims per run.

## `GET /healthz` -> `200 ok`

## Varloc selection

`dispatch.selectVarLoc`, before the cache check:

Varloc, from `all_varlocs.gpkg`:

| request | varloc |
| --- | --- |
| `varloc` given, intersects `treatment_area` | that varloc |
| `varloc` given, not in the gpkg / no intersection | 400 |
| no `varloc` | largest overlap; none -> 400 |

Backend. The bundle is `<models>/<varloc>/current` when the varloc is in `valid_varlocs.gpkg`; PT uses it too, for its dataset config:

| `backend` | bundle | `BURN_EMULATOR_RUNNER_GPU_JOB` | runs | job id `model_version` |
| --- | --- | --- | --- | --- |
| `DL` | yes | set | DL | `<model_version>` |
| `DL` | yes | unset | PT (warning) | `pt-<model_version>` |
| `DL` | no | - | PT (warning) | `pt` |
| `PT` | yes | - | PT | `pt-<model_version>` |
| `PT` | no | - | PT, runner's `run_smoke.yaml` defaults | `pt` |

Both gpkgs are cached per `inputs_version` and re-downloaded when their GCS generation changes (checked every 60s). They are read with `modernc.org/sqlite` and parsed with GEOS (`twpayne/go-geos`); the treatment area is reprojected into EPSG:5070 with PROJ (`twpayne/go-proj`). Overlap is the exact intersection area in EPSG:5070; ties go to the first varloc name. Accepted `crs`: CRS84, or any EPSG code known to PROJ as `EPSG:<code>`, `urn:ogc:def:crs:EPSG:[<version>]:<code>` or `http://www.opengis.net/def/crs/EPSG/<version>/<code>`; none reads as EPSG:4326 (longitude, latitude). Invalid polygons are repaired with GEOS `MakeValid` (structure method); rings must be closed.

## Claims

A claim is a GCS object written with a generation precondition; identical concurrent requests share one run. Its generation goes to the runner and back on the report; a report for an older generation leaves a newer claim alone. An active claim (`staging`, `queued`, `running`) is stale from its `expires_at` on. The api sets `expires_at` on each write: `running` 30 min after the trigger fires (`dispatch.runStaleAfter`), `queued` 60 min after it was queued (`dispatch.queueStaleAfter`), `staging` 100 s after it was taken (`dispatch.stagingStaleAfter`). A runner loop sets it on take to `BURN_EMULATOR_RUN_TIMEOUT` + 10 min, on a requeue to the claim's queue window again, and marks expired claims `failed` with `GetJob`'s error. A claim without `expires_at` expires at `updated_at` + the api's age for its status. A 4xx trigger failure releases the claim; a timeout, network error or 5xx keeps it (POST returns 500, GET shows `pending` until it goes stale).

| `status` | written by | |
| --- | --- | --- |
| `staging` | api on create and reclaim (metal mode), until `meta.geojson` is written | active; no body |
| `running` | api on create and reclaim (Cloud Run mode), runner loop on take | active; loop: `runner` |
| `queued` | api (metal mode), after meta.geojson; runner loop when stopped mid-run | active; body = runner env JSON |
| `failed` | api on a failed report; runner loop on an expired claim | reclaimed by the next identical POST |

The api writes whole claims (create, reclaim, `queued`, `failed`). The runner loop only patches metadata, with generation and metageneration preconditions.

## Output bucket layout

All objects share the suffix `<inputs_version>/<varloc>/<model_version | pt[-<model_version>]>/<hash>` (`dispatch.JobID.Path`, the `job_id`).

```
gs://<out>/
├── <job_id>/output.tif               # output
├── <job_id>/meta.geojson             # runner input, then the run meta
├── _claims/<job_id>                  # claim
├── _reports/<job_id>                 # runner report
├── _failed/<job_id>                  # archived failed / stale claim
└── _prune/last                       # last daily prune
```

| Prefix | Written by | Metadata | Lifetime |
| --- | --- | --- | --- |
| `<job_id>/output.tif` | runner | none | permanent; its existence is the cache hit |
| `<job_id>/meta.geojson` | api before the trigger (treatment area, EPSG:5070); runner overwrites it before `output.tif` (EPSG:5070 Feature, run + bundle as properties) | none | permanent; deleted with the output on failure, rewritten on retry |
| `_claims/` | api; metadata also by the runner loop | `status` (see [Claims](#claims)), `job_name`, `attempts`, `updated_at`, `expires_at` (active), `error`, metal: `runner` | deleted on success; failed or stale claims move to `_failed/` after a day |
| `_reports/` | runner | `status` (`completed` \| `failed`), `claim_generation`, `error` | deleted once handled |
| `_failed/` | api daily prune | the claim's, `status=failed`, `archived_at`, stale ones `last_status` | permanent; latest failure per job id |
| `_prune/last` | api | none | rewritten once a day |

Only `_reports/` triggers the notification (the notification's object prefix filter).

## Timeouts

| Rule | Values | Set in |
| --- | --- | --- |
| runner timeout < claim stale age | 20 min < 30 min; metal: 30 min < 40 min | runner job task timeout, `dispatch.runStaleAfter`; metal: `BURN_EMULATOR_RUN_TIMEOUT`, + 10 min (`loop.REPORT_GRACE_SECONDS`) |
| runner retries = 0 | 0 | runner job max retries |
| report worst case < Pub/Sub ack deadline | 90 s + 2 x 10 s < 120 s | `handlers.reportTimeout`, `dispatch.releaseTimeout`, push subscription ack deadline |
| handler worst case < Cloud Run request timeout | 4 min < 300 s | `http.Server.WriteTimeout`, api service request timeout |

## Metal dispatch

With `BURN_EMULATOR_METAL` set, the api triggers nothing: each claim is taken `staging`, then rewritten `queued`, with the runner env as its body, once `meta.geojson` is written. A runner on the cluster, started with `BURN_EMULATOR_LOOP=1` (see [burn-emulator-runner](../burn-emulator-runner/README.md#slurm-loop)), polls `_claims/` and runs the claims itself. It talks to GCS as the runner SA. Reports return through `_reports/` and Pub/Sub, as in Cloud Run mode.

```
api: claim staging (no body), meta.geojson, then claim queued (generation G0, body = runner env)
runner loop: expired claims -> failed; queued -> running (runner = <host>/<slurm job>, expires_at = now + run timeout + 10 min), adds the inputs and bundle paths (left out of a metal claim) from its on-prem roots,
    meta.geojson, output.tif, _reports/<job_id> (claim_generation = G0)  ->  HandleReport as in Cloud Run mode
    stopped mid-run: running -> queued
```

Each loop transition is a metadata patch with `ifGenerationMatch=G0` and `ifMetagenerationMatch=<read>`; a loop whose patch fails moves on to the next claim. A claim the api has reclaimed in the meantime (new generation) is left alone.

## Build

```bash
go build -o burn-emulator-api ./cmd/server   # Go 1.26+, cgo, libgeos-dev + libproj-dev
docker build -f burn-emulator-api/Dockerfile -t burn-emulator-api .   # from the repo root
```
