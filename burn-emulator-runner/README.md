# burn-emulator-runner

Cloud Run Job. Runs `burn_emulator.run.run()` once per execution for a request picked by [`burn-emulator-api`](../burn-emulator-api), then exits. One image backs `burn-emulator-runner-gpu-<env>` (`DL`) and `burn-emulator-runner-cpu-<env>` (`PT`, `RUN_DEVICE=cpu`).

With `BURN_EMULATOR_LOOP=1` (the [Dockerfile](Dockerfile) sets `0`), the same image runs as a long-lived Slurm job instead: see [Slurm loop](#slurm-loop). `main()` picks the mode.

## Config

Job template:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_DEBUG` | log per-run timings |
| `BURN_EMULATOR_LOOP` | `0` (Dockerfile): one execution; `1`: [Slurm loop](#slurm-loop) |
| `RUN_DEVICE` | `cuda` (default) or `cpu` |

Per execution, set by the api:

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_VARLOC` | varloc (requested, or picked by the api) |
| `BURN_EMULATOR_BACKEND` | `DL` or `PT` |
| `BURN_EMULATOR_MODEL_VERSION` | the varloc's bundle; always for DL, for PT when the varloc has one |
| `BURN_EMULATOR_BUNDLE_DIR` | with it: `/models/<varloc>/<model_version>` (GCS-FUSE) |
| `BURN_EMULATOR_BUNDLE_URI` | with it: `gs://` of that bundle, recorded in `meta.geojson` |
| `BURN_EMULATOR_INPUTS_VERSION` | recorded in `meta.geojson`; fills `run_smoke.yaml` for PT without a bundle |
| `BURN_EMULATOR_OUTPUT_META` | `<output_dir>/meta.geojson`: the treatment area in, the run meta out |
| `BURN_EMULATOR_HASH` | cache key |
| `BURN_EMULATOR_OUTPUT_PATH` | `<output_dir>/output.tif`, the cache hit |
| `BURN_EMULATOR_REPORT_PATH` | `gs://` `_reports/` object written when the run finishes |
| `BURN_EMULATOR_CLAIM_GENERATION` | echoed on the report |
| `BURN_EMULATOR_BASELINE_FUELS` | `/inputs/<inputs_version>/baseline` (GCS-FUSE) |
| `BURN_EMULATOR_LEGALMAX_FUELS` | `/inputs/<inputs_version>/legalmax` |
| `BURN_EMULATOR_TOPO_PATH` | `/inputs/<inputs_version>/topo` |
| `BURN_EMULATOR_FBFM_MAP_PATH` | `/inputs/<inputs_version>/fbfm/fbfm_behavior_adjectives.csv` |
| `BURN_EMULATOR_IGNITION_DENSITY` | optional; default: the bundle's `config.yaml` value, or `run_smoke.yaml`'s (20) without one |

## Model bundle

```
/models/<varloc>/current                 # active model_version (read by the api)
/models/<varloc>/<model_version>/
├── model.pt                             # checkpoint
├── stats.yaml                           # normalization stats
├── config.yaml                          # model_name, ckpt_path, model, activation, dataset, dataloader
└── bundle_meta.json                     # model_repo_sha, model_repo_dirty, model_class_path, model_code_sha256
```

DL only: a `model_code_sha256` that differs from this image's copy of `model_class_path`, or a missing `bundle_meta.json`, is logged as a warning; the run continues.

## Flow

```
1. download the treatment area from <output_meta>
2. with a bundle (DL, or PT on a trained varloc): load <bundle_dir>/config.yaml
   (the bundle dir is the experiment_dir; PT uses only its dataset config)
   PT without one: configs/varlocs/templates/run_smoke.yaml from the image, wind_range from configs/wind_directions.csv
3. inject treatment_area, fuels_paths, topo_path, fbfm_map_path, ignition_density, out_path
4. run(**config) -> a local tif under /tmp
5. overwrite <output_meta>, then upload the tif to <output_path> (the cache hit)
6. write the empty <report_path> object with metadata {status: completed|failed, claim_generation, error}
7. exit 0, or 1 on any error
```

SIGTERM before `run()` returns is a failure and still writes a report; after that, and while reporting a failure, it is ignored. `/tmp` is in-memory on Cloud Run and counts against the job memory. The runner job's max retries must stay 0.

## Output

```
<output_dir>/
├── meta.geojson             # api: the unioned treatment area in EPSG:5070; runner overwrites with one EPSG:5070 Feature,
│                            # geometry = unioned treatment area, properties = hash, backend, varloc,
│                            # inputs_version, ignition_density, output_path
│                            # + with a bundle: model_version, bundle_uri, every bundle_meta.json field
└── output.tif               # bands: no change, to crown, from crown
```

## Slurm loop

`BURN_EMULATOR_LOOP=1` is for an api with `BURN_EMULATOR_METAL` set, where claims are written `queued` with the runner env as their body (`dispatch.runnerEnv`). One process holds its GPU node for the life of the job and runs claims one at a time, DL or PT as each claim's `BURN_EMULATOR_BACKEND` says, until it gets SIGTERM. Inputs and bundles are read on prem, never downloaded. The api leaves the fuels, topo, fbfm map and bundle dir paths out of a metal claim, and the loop sets them under `BURN_EMULATOR_INPUTS_ROOT` (`<inputs_version>/...`) and `BURN_EMULATOR_BUNDLES_ROOT` (`<varloc>/<model_version>`). `make env-vars` points both at the checkout's `burn-emulator-model/data/`, where `make inputs-release` and `make model-release` keep their copies.

```
startup: warm the GPU; turn on burn_emulator's caches and read data/inputs/current's rasters whole into memory
loop:    list _claims/; staging, queued or running claims past expires_at -> failed (error as GetJob reports it)
         oldest queued claim first (DL or PT, as its BURN_EMULATOR_BACKEND says)
         queued -> running (metadata patch, generation + metageneration matched; runner = <host>/<slurm job>,
                            expires_at = now + BURN_EMULATOR_RUN_TIMEOUT + 10 min)
         read the env body; add the inputs and bundle paths from the two roots
         steps 1-6 of the flow above, claim_generation = the claim's generation
         nothing queued: sleep BURN_EMULATOR_POLL_SECONDS
```

Runs take their windows from the in-memory rasters (`burn_emulator.rasters`), and up to `BURN_EMULATOR_MAX_MODELS` bundles' models stay loaded on the GPU, least recently used evicted first (`burn_emulator.run.enable_caching`). Other inputs versions are read whole on first use. A run past `BURN_EMULATOR_RUN_TIMEOUT` is cancelled between batches and reported failed. On SIGTERM or SIGINT, the current run is cancelled, its claim is put back to `queued` with its original queue window for another loop, and the process exits. Several loops can share a bucket: the conditional patch lets only one of them take a given claim.

| Variable | Purpose |
| --- | --- |
| `BURN_EMULATOR_OUTPUT_URI` | `gs://` output bucket; claims under `_claims/` |
| `BURN_EMULATOR_INPUTS_ROOT` | `<inputs_version>/{baseline,legalmax,topo,fbfm}`; default `burn-emulator-model/data/inputs` |
| `BURN_EMULATOR_BUNDLES_ROOT` | `<varloc>/<model_version>/`; default `burn-emulator-model/data/bundles` |
| `BURN_EMULATOR_MAX_MODELS` | models kept loaded on the GPU; default 8 (`make env-vars` writes it) |
| `BURN_EMULATOR_POLL_SECONDS` | default 10 |
| `BURN_EMULATOR_RUN_TIMEOUT` | seconds, default 1800; a taken claim expires this + 10 min after it is taken |
| `GOOGLE_APPLICATION_CREDENTIALS` | runner SA key |

Started on the cluster by [`slurm/loop.slurm`](slurm/loop.slurm), from the repo root of a deploy checkout:

```bash
make env-vars                         # once; ~/.config/burn-emulator/env
make runner-venv                      # once per checkout / upgrade
make runner-loop                      # --gres=gpu:1 --exclusive --mem=0: DL on the GPU, PT on the node's cpus
```

The script runs the venv with the Dockerfile's runtime env plus `BURN_EMULATOR_LOOP=1`. `BURN_EMULATOR_OUTPUT_URI` and `GOOGLE_APPLICATION_CREDENTIALS` come from `~/.config/burn-emulator/env` (`make env-vars`; `BURN_EMULATOR_ENV_FILE` points elsewhere), through `burn-emulator-model/scripts/load_env.sh`. So do `BURN_EMULATOR_INPUTS_ROOT` and `BURN_EMULATOR_BUNDLES_ROOT`. Logs go to `burn-emulator-runner/data/logs/loop/`. The job runs until `scancel` or its time limit (`SBATCH_ARGS=--time=...`); either one requeues the claim in progress. Cloud Run never loops: the image keeps `BURN_EMULATOR_LOOP=0`.

## Build

```bash
docker build -f burn-emulator-runner/Dockerfile -t burn-emulator-runner .   # from the repo root
```
