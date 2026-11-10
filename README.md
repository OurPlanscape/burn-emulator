# burn-emulator

A deep-learning emulator of a burn spread model that generates thousands of burns in seconds for conditional burn probability (CBP).

## Components

[View architecture diagram](https://app.diagrams.net/#HOurPlanscape%2Fburn-emulator%2Fmain%2Fdiagram.drawio) ([source](diagram.drawio))

| Directory | What it is |
| --- | --- |
| [`burn-emulator-api/`](burn-emulator-api/README.md) | Go service, CPU-only: validates or picks the varloc, picks the backend, serves the GCS cache, triggers a runner job on a miss (or, in metal mode, queues a claim for the runner's claim loop on prem). |
| [`burn-emulator-runner/`](burn-emulator-runner/README.md) | Cloud Run Job (GPU for `DL`, CPU for `PT`): runs `burn_emulator.run.run()` once per execution, or as a long-lived claim loop on the cluster. |
| [`burn-emulator-model/`](burn-emulator-model/README.md) | Model library + CLI: training data, train, evaluate, bundle and `run()` inference. |

## Request flow

```
caller --POST /v1/jobs {treatment_area [, varloc, job_name, ignition_density, backend]}--> burn-emulator-api
  1. validate the request
  2. inputs_version = gs://<inputs>/current                                   (60s cache)
     varloc         = requested varloc if it intersects treatment_area in
                      <inputs_version>/varlocs/all_varlocs.gpkg (else 400),
                      or the largest overlap there (none -> 400)
     bundle         = gs://<models>/<varloc>/current if varloc is in valid_varlocs.gpkg (60s cache)
     DL if requested, there is a bundle and a GPU job is set (or metal mode), else PT; PT also uses the bundle when there is one
     hash           = sha256(treatment_area[|ignition_density]) + 0 (DL) | 1 (PT)
     output_dir     = gs://<out>/<inputs_version>/<varloc>/<model_version | pt[-<model_version>]>/<hash>
  3. output exists?                                                      -> 200 cached
     claim running?                                                      -> 202 pending + Location
     else claim it, write <output_dir>/meta.geojson, trigger the GPU (DL) or CPU (PT) job
                                                                         -> 202 pending + Location
     metal mode: once meta.geojson is written, the claim is rewritten queued with the runner env as its body;
     a long-lived runner on the cluster (BURN_EMULATOR_LOOP=1) takes it and runs it
  4. runner: loads <varloc>/<model_version> from the models mount when there is one, else PT uses run_smoke.yaml;
     run(), overwrite <output_dir>/meta.geojson with the run meta, upload <output_dir>/output.tif,
     write gs://<out>/_reports/... {status, claim_generation, error}
  5. _reports/ OBJECT_FINALIZE -> Pub/Sub push -> POST /internal/pubsub/run-reports
     completed -> release the claim; failed -> delete partial output, mark the claim failed
  6. caller polls GET <Location> until cached or failed
```

The api rejects malformed GeoJSON, an unsupported `crs`, an unknown `varloc`, and an area outside the requested varloc or every varloc. The ignition cap (2**16 ignitions: `ignition_density` per km² over the buffered treatment area) is checked by the runner, after a job has started.

## Outputs

Each run writes `<output_dir>/output.tif` (crown change fractions) and `<output_dir>/meta.geojson` (treatment area, run provenance and indirect benefit). Both are specified in [burn-emulator-runner Output](burn-emulator-runner/README.md#output).

## Model registry (GCS)

```
gs://<models>/<varloc>/current                  # text file: the active model_version
gs://<models>/<varloc>/<model_version>/         # model.pt, stats.yaml, config.yaml, bundle_meta.json
```

`<model_version>` = `<model.pt mtime, UTC YYYYMMDDTHHMMSSZ>-<7-char model_repo_sha>[-dirty]`, set by `publish_model.sh` from the bundle; DL and PT runs on a varloc with a bundle are cached under it.

## Varloc files

Under `burn-emulator-model/`:

| file | contents | written by |
| --- | --- | --- |
| `data/inputs/<inputs_version>/varlocs/all_varlocs.gpkg` | every varloc polygon, EPSG:5070; also what training-data generation reads (`config.varlocs_gpkg`) | by hand, with the rasters |
| `data/outputs/varlocs.txt` | trained varlocs | `mark_trained.sh`, `update-mark-trained` |
| `data/inputs/<inputs_version>/varlocs/valid_varlocs.gpkg` | `all_varlocs.gpkg` filtered to `varlocs.txt` | `valid-varlocs` |

`valid-varlocs` works on `data/inputs/current`'s version, and `inputs-release` / `varlocs-release` upload it unchanged.

## Makefile targets

Run from the repo root. `make help` lists every target; `[direct]` runs in this shell, `[slurm]` submits to the cluster, `[direct|slurm]` submits with `SLURM=1`.

Images (need `BURN_EMULATOR_ARTIFACT_STORE`):

| target | what it does |
| --- | --- |
| `api-image` / `api-release` | build / build + push `burn-emulator-api:<git-sha>[-dirty]`; a dirty tree is refused when `BURN_EMULATOR_ENV=production` |
| `runner-image` / `runner-release` | same, for `burn-emulator-runner` |
| `images-release` | `api-release` + `runner-release` under one `VERSION`; deploy both together |

Training data and training:

| target | what it does |
| --- | --- |
| `training-data VARLOC= [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1]` | `burn_emulator -m ignite` into `data/training_data/<varloc>/<inputs_version>_<ignitions_version>/` (versions default to `current.yaml`, `NUM_IGNITIONS` to 5000); `OVERWRITE=1` replaces an existing directory |
| `training-data-all [EXCLUDE=] [...] [SLURM=1]` | `training-data` for every varloc in `data/inputs/<inputs_version>/varlocs/all_varlocs.gpkg`, skipping complete ones (legalmax `outputs_table.csv`) unless `OVERWRITE=1`; exits nonzero listing failures. `SLURM=1` queues them for `slurm/ignitions.slurm` (`EXCLUDE` is slurm only) |
| `train-all [SLURM=1 [EXCLUDE=] [NODES=]]` | trains every varloc with complete training data; adds each to `data/outputs/varlocs.txt` |
| `train-release VARLOC= [NODES=]` | slurm: train -> `model-bundle` -> `model-release` -> `mark_trained.sh` + `varlocs-release` |
| `train-release-all [EXCLUDE=] [NODES=]` | the same for every varloc with complete training data |
| `update-mark-trained` | rewrites `data/outputs/varlocs.txt` to the varlocs with an `epoch-<num_epochs - 1>` checkpoint for the current architecture + data version; errors instead of writing an empty list |

Releases (need `BURN_EMULATOR_MODELS_URI` / `BURN_EMULATOR_INPUTS_URI`):

| target | what it does |
| --- | --- |
| `valid-varlocs` | filters `data/inputs/current`'s `varlocs/all_varlocs.gpkg` to `data/outputs/varlocs.txt` -> `varlocs/valid_varlocs.gpkg` |
| `model-bundle VARLOC=` | `burn_emulator -m bundle` -> `data/bundles/<VARLOC>/<model_version>/`, then points `data/bundles/<VARLOC>/current` at it: the models bucket's layout |
| `model-bundle-all` | `update-mark-trained`, `valid-varlocs`, then `model-bundle` for every varloc in `varlocs.txt` |
| `model-release VARLOC= [FORCE=1] [POINTER_ONLY=1]` | `scripts/publish_model.sh`: uploads `data/bundles/<VARLOC>/current`'s bundle, then repoints the bucket's `<VARLOC>/current` to the same version; `POINTER_ONLY=1` skips the upload (metal DL reads bundles on prem) |
| `model-release-all` | the same for every varloc in `varlocs.txt`; repoints only after every upload |
| `inputs-current [INPUTS_VERSION=]` | `scripts/set_inputs_current.sh`: checks the hand-made `data/inputs/<inputs_version>/` (`baseline/`, `legalmax/`, `topo/`, `varlocs/all_varlocs.gpkg`, `fbfm/fbfm_behavior_adjectives.csv`), then points `data/inputs/current` at it. `INPUTS_VERSION` defaults to `current.yaml` |
| `inputs-release [FORCE=1]` | `scripts/publish_inputs.sh`: uploads `data/inputs/current`'s version as is, then repoints the bucket's `current` |
| `varlocs-release` | `valid-varlocs`, then `scripts/publish_varlocs.sh` uploads `data/inputs/current`'s `varlocs/{valid,all}_varlocs.gpkg` over that published version (no repoint) |

Bundle and inputs dirs mirror their buckets; a release copies what is local `current`. Inputs are made by hand; only `current` is written locally. The runner loop reads the same dirs.

Local runs:

| target | what it does |
| --- | --- |
| `smoke VARLOC= [WIND_RANGE="<lo> <hi>"] [TREATMENT_AREA=] [OUT_PATH=] [PT=1]` | `burn_emulator -m run -d` with `configs/varlocs/templates/run_smoke.yaml`; prints timings and the [indirect benefit](burn-emulator-runner/README.md#metageojson). `WIND_RANGE` defaults to `configs/wind_directions.csv`, `TREATMENT_AREA` to `data/training_data/<varloc>/sample_pa.geojson` |
| `inference VARLOC= OUTPUTS_ROOT=` | `scripts/ignite_inference.sh`: evaluates every scenario under `OUTPUTS_ROOT` and saves each ignition's burn prediction |
| `inference-all OUTPUTS_ROOT=` | `inference` for every varloc in `varlocs.txt`, root `OUTPUTS_ROOT/<varloc>` |
| `shell` | syncs `burn-emulator-model/.venv-<arch>` (with the `data` extra) and opens a shell in it |

Runner loop (deploy checkout, metal mode api; see [burn-emulator-runner](burn-emulator-runner/README.md#slurm-loop)):

| target | what it does |
| --- | --- |
| `env-vars` | writes `~/.config/burn-emulator/env` (mode 600), no arguments. `BURN_EMULATOR_INPUTS_ROOT` and `BURN_EMULATOR_BUNDLES_ROOT` are always this checkout's `burn-emulator-model/data/{inputs,bundles}`. `GOOGLE_APPLICATION_CREDENTIALS` defaults to `~/.config/burn-emulator/runner-key.json`. `BURN_EMULATOR_MAX_MODELS` defaults to 8. `BURN_EMULATOR_{ENV,ARTIFACT_STORE,INPUTS_URI,MODELS_URI,OUTPUT_URI}` keep their current values (file or shell), and missing ones are asked for. Every make target, script and slurm job reads it through `burn-emulator-model/scripts/load_env.sh`. The file wins over the shell; a make command-line value wins over the file |
| `runner-venv` | syncs `burn-emulator-runner/.venv-<arch>` from its lockfile; the venv `slurm/loop.slurm` runs |
| `runner-loop [NODES=] [SBATCH_ARGS=]` | sbatches `burn-emulator-runner/slurm/loop.slurm` on a GPU node (`--gres=gpu:1 --exclusive --mem=0`): takes queued DL and PT claims until cancelled |
