## Overview

A repo for approximating a burn spread model using deep learning. GPU acceleration enables 1000s of burns in seconds, enabling faster CBP calculations at the cost of some additional uncertainty from modelling the derived spread model.

---

### Modelling

The default model is an archetypal CNN with one major adjustment: [circular kernels](https://arxiv.org/pdf/2107.02451). The reasoning for this adaptation is that very few objects in nature are 'square-shaped' thus, there is either unused learning capacity in the kernels used in convolutions or, the model must learn the mappings from that arbitrary shape choice to the expected shape of the burn. In addition, burns spread in a quasi-circular pattern (this obviously varies depending on various factors) which does not align with the grid anisotropy. The ignition point is set to a consistent point within the CNN context window (the center pixel) such that the model needs only to learn mappings from local connectivities between input pixels (fuels, etc.) to a spread shape from that point. In other words, it doesn't need to know where the burn will occur, just a fuzzy approximation of what it will look like. This can then be used to stamp 'burns' accross the landscape for the purpose of calculating conditional burn probability.

### Training Installation

```bash
# this should be done on one of the DGX sparks we have
git clone https://github.com/sig-gis/burn_emulator.git
cd burn-emulator/burn_emulator-model

# assuming you already have uv installed
uv sync
uv pip install -e .
```

### Training

```bash
# these should have the dataset file paths in there
burn_emulator -m train -c path/to/model/kwargs -c path/to/training/kwargs
burn_emulator -m test -c path/to/model/kwargs -c path/to/training/kwargs

# or

sbatch slurm/train_circle.slurm
sbatch slurm/test_circle.slurm

```

### Output file path structure

```
burn_emulator-model/
├── data/
│   ├── configs/
│   │   └── ...
│   ├── outputs/
│   │   ├── model_name/
│   │   │   ├── checkpoints/
│   │   │   ├── ... # logs etc.
├── src/
├── ...
```

### Input file path structure per model

```
path_to_inputs/
├── ignitions_path/
│   └── ...
├── topo_path/
│   └── ... # tifs containing aspect/slope
├── fuels_path/
│   └── ... # tifs containing all inputs
├── burn_paths/
│   ├── {ignition_number}
│   │   ├── {burn_time}
│   │   │   ├── fire_type.tif
├── ...
```

---

### API

This API is intended to be an internal microservice. It's fronted by a GKE regional **internal** load balancer (`gke-l7-rilb`) and expects callers to reach it over the same VPC. Its auth model — a GCP ID token checked against an allow-list of caller service accounts — is meant to identify trusted internal callers, not to withstand traffic from the public internet. Note: manifests for GKE infrastructure are not included here.

### API Installation

```bash
git clone https://github.com/sig-gis/burn_emulator.git
cd burn-emulator/burn-emulator-api

# requires Go 1.26.5+
go build -o burn-emulator-api .
```

### API Configuration

The server is configured entirely through environment variables, plus a `variations.yaml` allow-list:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_JOB_NAMESPACE` | k8s namespace jobs are created in |
| `BURN_EMULATOR_JOB_SERVICE_ACCOUNT` | service account jobs run as |
| `BURN_EMULATOR_ARTIFACT_STORE` | image registry for the model job container |
| `BURN_EMULATOR_OUTPUT_BUCKET` | bucket burn outputs are written to |
| `BURN_EMULATOR_TOKEN_AUDIENCE` | expected audience on caller ID tokens |
| `BURN_EMULATOR_ALLOWED_CALLERS` | comma-separated allow-list of caller service accounts |

### Request flow

```
[1] caller service
      | POST /v1/jobs, Authorization: Bearer <GCP ID token> (aud = BURN_EMULATOR_TOKEN_AUDIENCE)
      v
[2] burn-emulator-api
      | verify ID token (aud + email in BURN_EMULATOR_ALLOWED_CALLERS)
      | rate-limit (client IP)
      | validate variation/caching/job_name against variations.yaml
      | create batchv1.Job (RBAC: create/get/list on jobs only)
      v
[3] GKE: burn-emulator-variations namespace
      | Job controller schedules a Pod as BURN_EMULATOR_JOB_SERVICE_ACCOUNT
      | (Workload Identity -> GCP service account)
      v
[4] runner container (ideally this is a spot instance node)
      | model + checkpoint baked into image at build time (Dockerfile.model)
      | runs: burn-emulator -m run ... -o $BURN_EMULATOR_OUT_PATH
      | note: this reads into memory the entire FVS variation, stamps all burns, then
      |     outputs a raster of comparable size and transform
      v
[5] gs://BURN_EMULATOR_OUTPUT_BUCKET/{variation}/{job_name}/...
      (write-only - runner does not read inputs from the bucket at runtime)

[2] burn-emulator-api
      | responds immediately, does not wait for the job to finish
      v
[1] caller service
      received: 202 { "job_name": ..., "status": "scheduled" }
```

1. Caller sends `Authorization: Bearer <ID token>` to `POST /v1/jobs`; the api verifies the token's audience and caller email, rate-limits by client IP, and validates the request body against `variations.yaml`.
2. On success, the api creates a `batchv1.Job` via its in-cluster k8s client — its RBAC role only grants `create/get/list` on `jobs`, nothing broader.
3. GKE's Job controller schedules a Pod using `BURN_EMULATOR_JOB_SERVICE_ACCOUNT`, which is bound via Workload Identity to a GCP service account with write access to the output bucket.
4. The runner container writes burn outputs to the output bucket. It does **not** read inputs from the bucket at runtime — fuels/topo/checkpoint data are baked into the model image at build time (`Dockerfile.model`).
5. The api responds immediately with `202` and the generated job name; it does not wait for the job to finish.
