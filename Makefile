SHELL      := /bin/bash
.ONESHELL:
.SILENT:
MAKEFLAGS  += --no-print-directory

API_DIR    ?= burn-emulator-api
MODEL_DIR  ?= burn-emulator-model
VENV_NAME  ?= $(or $(UV_PROJECT_ENVIRONMENT),.venv-$(shell uname -m))
VENV       ?= $(MODEL_DIR)/$(VENV_NAME)
RUNNER_DIR ?= burn-emulator-runner
VERSION    ?= $(shell git rev-parse --short HEAD)$(shell [ -z "$$(git status --porcelain)" ] || echo -dirty)

# ENV_FILE (make env-vars) is the source of truth for ENV_VARS: every target gets it here, and scripts
# and slurm jobs through burn-emulator-model/scripts/load_env.sh, which BURN_EMULATOR_ENV_LOADED skips
# under make. A make command-line value wins over the file, the file over the shell environment.
# VERSION gets -dirty on an uncommitted tree; image builds refuse it when BURN_EMULATOR_ENV=production.
ENV_FILE ?= $(or $(BURN_EMULATOR_ENV_FILE),$(HOME)/.config/burn-emulator/env)
ENV_VARS := BURN_EMULATOR_ENV BURN_EMULATOR_ARTIFACT_STORE BURN_EMULATOR_INPUTS_URI BURN_EMULATOR_MODELS_URI \
            BURN_EMULATOR_OUTPUT_URI GOOGLE_APPLICATION_CREDENTIALS BURN_EMULATOR_INPUTS_ROOT BURN_EMULATOR_BUNDLES_ROOT \
            BURN_EMULATOR_MAX_MODELS
-include $(ENV_FILE)
export $(foreach v,$(ENV_VARS),$(if $(filter undefined,$(origin $(v))),,$(v)))
export BURN_EMULATOR_ENV_LOADED := 1

API_IMAGE    := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-api:$(VERSION)
RUNNER_IMAGE := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-runner:$(VERSION)

# local mirrors of the buckets' layouts (make env-vars writes the same paths)
INPUTS_ROOT  := $(or $(BURN_EMULATOR_INPUTS_ROOT),$(CURDIR)/$(MODEL_DIR)/data/inputs)
BUNDLES_ROOT := $(or $(BURN_EMULATOR_BUNDLES_ROOT),$(CURDIR)/$(MODEL_DIR)/data/bundles)

# VARLOCS_TXT: trained varlocs. Under data/inputs/current's version: ALL_VARLOCS_GPKG (by hand) and
# VALID_VARLOCS_GPKG (ALL_VARLOCS_GPKG filtered to VARLOCS_TXT)
VARLOCS_TXT        ?= $(MODEL_DIR)/data/outputs/varlocs.txt
INPUTS_CURRENT_DIR  = $(INPUTS_ROOT)/$(shell cat "$(INPUTS_ROOT)/current" 2>/dev/null)
ALL_VARLOCS_GPKG   ?= $(INPUTS_CURRENT_DIR)/varlocs/all_varlocs.gpkg
VALID_VARLOCS_GPKG ?= $(INPUTS_CURRENT_DIR)/varlocs/valid_varlocs.gpkg
require_inputs_current = if [ ! -f "$(INPUTS_ROOT)/current" ]; then echo "error: $(INPUTS_ROOT)/current not found - run make inputs-current" >&2; exit 1; fi

.DEFAULT_GOAL := help

.PHONY: help api-image api-release runner-image runner-release images-release valid-varlocs model-bundle model-bundle-all model-release model-release-all inputs-current inputs-release varlocs-release train-all train-release train-release-all update-mark-trained inference inference-all smoke training-data training-data-all shell runner-venv runner-loop env-vars

help: ## show this help; [direct] runs here, [slurm] submits to the cluster, [direct|slurm] picks via SLURM=1
	awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(firstword $(MAKEFILE_LIST))

# ENV_FILE is rewritten whole. The roots always come from this checkout; the rest keep the file's (or
# shell's) values, and missing ones are asked for. One still unset is written commented out.
PROMPT_VARS := BURN_EMULATOR_ENV BURN_EMULATOR_ARTIFACT_STORE BURN_EMULATOR_INPUTS_URI BURN_EMULATOR_MODELS_URI BURN_EMULATOR_OUTPUT_URI
GCP_PROJECT ?= planscape-23d66
env-vars: ## [direct] write ENV_FILE (~/.config/burn-emulator/env), asking for missing values; offers to create a missing runner key
	set -e
	tty=1
	{ : </dev/tty; } 2>/dev/null || tty=
	$(foreach v,$(PROMPT_VARS),$(v)="$($(v))";)
	for v in $(PROMPT_VARS); do
	    [ -z "$${!v}" ] && [ -n "$$tty" ] && read -r -p "$$v: " "$$v" </dev/tty
	done
	GOOGLE_APPLICATION_CREDENTIALS="$(or $(GOOGLE_APPLICATION_CREDENTIALS),$(HOME)/.config/burn-emulator/runner-key.json)"
	BURN_EMULATOR_INPUTS_ROOT="$(CURDIR)/$(MODEL_DIR)/data/inputs"
	BURN_EMULATOR_BUNDLES_ROOT="$(CURDIR)/$(MODEL_DIR)/data/bundles"
	BURN_EMULATOR_MAX_MODELS="$(or $(BURN_EMULATOR_MAX_MODELS),8)"

	mkdir -p -m 700 "$(dir $(ENV_FILE))"
	tmp=$$(mktemp "$(ENV_FILE).XXXXXX")
	{
	    echo "# make env-vars; KEY=value, no quotes; read by the Makefile and $(MODEL_DIR)/scripts/load_env.sh"
	    for v in $(ENV_VARS); do
	        if [ -n "$${!v}" ]; then echo "$$v=$${!v}"; else echo "# $$v= (unset)"; fi
	    done
	} > "$$tmp"
	chmod 600 "$$tmp"
	mv "$$tmp" "$(ENV_FILE)"
	echo "wrote $(ENV_FILE):"
	grep -v '^# make' "$(ENV_FILE)"

	key=$$GOOGLE_APPLICATION_CREDENTIALS
	[ -f "$$key" ] && exit 0
	if [ -z "$$BURN_EMULATOR_ENV" ]; then
	    echo "warning: $$key not found; set BURN_EMULATOR_ENV to create it" >&2
	    exit 0
	fi
	sa="burn-runner-sa-$$BURN_EMULATOR_ENV@$(GCP_PROJECT).iam.gserviceaccount.com"
	if [ -z "$$tty" ]; then
	    echo "warning: $$key not found; rerun make env-vars in a terminal to create it" >&2
	    exit 0
	fi
	read -r -p "$$key not found; create a key for $$sa? [y/N] " answer </dev/tty
	case "$$answer" in
	    [yY]*) ;;
	    *) echo "skipped; the runner loop needs $$key" >&2; exit 0 ;;
	esac
	mkdir -p -m 700 "$$(dirname "$$key")"
	(umask 077 && gcloud iam service-accounts keys create "$$key" --iam-account="$$sa")
	echo "created $$key"

api-image: ## [direct] build the API image
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - run make env-vars" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(API_DIR)/Dockerfile -t $(API_IMAGE) .

api-release: api-image ## [direct] build and push the API image
	docker push $(API_IMAGE)

runner-image: ## [direct] build the runner image
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - run make env-vars" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(RUNNER_DIR)/Dockerfile -t $(RUNNER_IMAGE) .

runner-release: runner-image ## [direct] build and push the runner image
	docker push $(RUNNER_IMAGE)

runner-venv: ## [direct] sync burn-emulator-runner/.venv-<arch>, the venv slurm/loop.slurm runs
	UV_PROJECT_ENVIRONMENT="$(CURDIR)/$(RUNNER_DIR)/.venv-$$(uname -m)" uv sync --project $(RUNNER_DIR) --locked --no-dev

runner-loop: ## [slurm] start a runner claim loop (DL and PT claims) on a GPU node, whole node for PT's workers ([NODES=n1] [SBATCH_ARGS=])
	mkdir -p $(RUNNER_DIR)/data/logs/loop
	sbatch --gres=gpu:1 --exclusive --mem=0 $(if $(NODES),--nodelist=$(NODES)) $(SBATCH_ARGS) --export=ALL $(RUNNER_DIR)/slurm/loop.slurm

images-release: api-release runner-release ## [direct] build and push the API and runner images under one VERSION
	echo "pushed $(API_IMAGE)"
	echo "pushed $(RUNNER_IMAGE)"
	echo "bump both burn_emulator_*_image defaults in infrastructure/terraform/app/variables.tf to :$(VERSION)"

valid-varlocs: ## [direct] filter data/inputs/current's varlocs/all_varlocs.gpkg to varlocs.txt -> varlocs/valid_varlocs.gpkg
	$(require_inputs_current)
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	python scripts/filter_varlocs_gpkg.py -i "$(abspath $(ALL_VARLOCS_GPKG))" -v "$(abspath $(VARLOCS_TXT))" -o "$(abspath $(VALID_VARLOCS_GPKG))"

model-bundle: ## [direct] bundle one model into data/bundles/<VARLOC>/<model_version>/, point data/bundles/<VARLOC>/current at it (VARLOC=)
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	burn_emulator -m bundle -c configs/varlocs/current.yaml -vl $(VARLOC)

# valid-varlocs runs after the txt sync
model-bundle-all: update-mark-trained ## [direct] sync varlocs.txt to finished checkpoints, then bundle every trained varloc
	set -e
	$(MAKE) valid-varlocs
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(VARLOCS_TXT)")
	if [ $${#varlocs[@]} -eq 0 ]; then echo "error: $(VARLOCS_TXT) lists no varlocs" >&2; exit 2; fi
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) model-bundle VARLOC="$$varloc"
	done

model-release: ## [direct] upload data/bundles/<VARLOC>/current's bundle, then repoint the bucket's <VARLOC>/current (VARLOC= [POINTER_ONLY=1])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -z "$(BURN_EMULATOR_MODELS_URI)" ]; then echo "error: BURN_EMULATOR_MODELS_URI is not set - run make env-vars" >&2; exit 2; fi
	BURN_EMULATOR_BUNDLES_ROOT="$(BUNDLES_ROOT)" $(MODEL_DIR)/scripts/publish_model.sh "$(VARLOC)"

model-release-all: ## [direct] model-release for every trained varloc (VARLOCS_TXT); repoints after every upload ([POINTER_ONLY=1])
	set -e
	if [ -z "$(BURN_EMULATOR_MODELS_URI)" ]; then echo "error: BURN_EMULATOR_MODELS_URI is not set - run make env-vars" >&2; exit 2; fi
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(VARLOCS_TXT)")
	if [ $${#varlocs[@]} -eq 0 ]; then echo "error: $(VARLOCS_TXT) lists no varlocs" >&2; exit 2; fi
	BURN_EMULATOR_BUNDLES_ROOT="$(BUNDLES_ROOT)" $(MODEL_DIR)/scripts/publish_model.sh "$${varlocs[@]}"

# data/inputs/<inputs_version>/ is made by hand, in the inputs bucket's layout
inputs-current: ## [direct] check a hand-made data/inputs/<inputs_version>/ and point data/inputs/current at it ([INPUTS_VERSION=], default: current.yaml)
	set -e
	version="$(INPUTS_VERSION)"
	[ -n "$$version" ] || version=$$($(MODEL_DIR)/scripts/data_version.sh inputs_version)
	$(MODEL_DIR)/scripts/set_inputs_current.sh "$(INPUTS_ROOT)" "$$version"

inputs-release: ## [direct] upload data/inputs/current's version as is, then repoint the bucket's current
	if [ -z "$(BURN_EMULATOR_INPUTS_URI)" ]; then echo "error: BURN_EMULATOR_INPUTS_URI is not set - run make env-vars" >&2; exit 2; fi
	$(MODEL_DIR)/scripts/publish_inputs.sh "$(INPUTS_ROOT)" $(BURN_EMULATOR_INPUTS_URI)

varlocs-release: valid-varlocs ## [direct] rebuild data/inputs/current's varlocs/valid_varlocs.gpkg, then upload its varlocs/ over that published version
	if [ -z "$(BURN_EMULATOR_INPUTS_URI)" ]; then echo "error: BURN_EMULATOR_INPUTS_URI is not set - run make env-vars" >&2; exit 2; fi
	$(MODEL_DIR)/scripts/publish_varlocs.sh "$(INPUTS_ROOT)" $(VALID_VARLOCS_GPKG) $(ALL_VARLOCS_GPKG) $(BURN_EMULATOR_INPUTS_URI)

train-all: ## [direct|slurm] train every varloc with complete training data; adds each to data/outputs/varlocs.txt once trained ([SLURM=1 [EXCLUDE=v1,v2] [NODES=n1,n2]])
	if [ -n "$(SLURM)" ]; then
	    $(MODEL_DIR)/slurm/submit_train_all.sh $(if $(EXCLUDE),-x "$(EXCLUDE)") $(NODES)
	elif [ -n "$(EXCLUDE)" ]; then
	    echo "error: EXCLUDE needs SLURM=1" >&2; exit 2
	else
	    $(MODEL_DIR)/scripts/train_all.sh
	fi

train-release: ## [slurm] train + bundle + publish for one varloc (VARLOC= [NODES=n1,n2])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	$(MODEL_DIR)/slurm/submit_train_all.sh -p -v $(VARLOC) $(NODES)

train-release-all: ## [slurm] train + bundle + publish for every trainable varloc ([EXCLUDE=v1,v2] [NODES=n1,n2])
	$(MODEL_DIR)/slurm/submit_train_all.sh -p $(if $(EXCLUDE),-x "$(EXCLUDE)") $(NODES)

update-mark-trained: ## [direct] set data/outputs/varlocs.txt to exactly the varlocs with a final-epoch checkpoint (num_epochs in configs/<arch>/train.yaml) for the current architecture + data_version
	cd "$(MODEL_DIR)"
	scripts/update_trained.sh

inference: ## [direct] evaluate every scenario under OUTPUTS_ROOT and save each ignition's burn prediction (VARLOC= OUTPUTS_ROOT=)
	if [ -z "$(VARLOC)" ] || [ -z "$(OUTPUTS_ROOT)" ]; then
	    echo "error: pass VARLOC=<varloc> OUTPUTS_ROOT=<dir>" >&2; exit 2
	fi
	$(MODEL_DIR)/scripts/ignite_inference.sh $(VARLOC) $(OUTPUTS_ROOT)

# expects one scenario root per varloc at OUTPUTS_ROOT/<varloc>
inference-all: ## [direct] run inference for every varloc (OUTPUTS_ROOT=)
	set -e
	if [ -z "$(OUTPUTS_ROOT)" ]; then echo "error: pass OUTPUTS_ROOT=<dir>" >&2; exit 2; fi
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(VARLOCS_TXT)")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) inference VARLOC="$$varloc" OUTPUTS_ROOT="$(OUTPUTS_ROOT)/$$varloc"
	done

smoke: ## [direct] run the smoke test for one varloc in debug mode (VARLOC= [WIND_RANGE="lo hi"] [TREATMENT_AREA=] [OUT_PATH=] [IGNITION_DENSITY=] [PT=1])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	arch=$$(grep -oP '^architecture:[[:space:]]*\K\S+' configs/varlocs/current.yaml)
	wind_range="$(WIND_RANGE)"
	if [ -z "$$wind_range" ]; then
	    wind_range=$$(awk -F, -v k="$(VARLOC)" '$$1 == k {print $$2, $$3}' configs/wind_directions.csv)
	fi
	if [ -z "$$wind_range" ]; then echo "error: no wind range for $(VARLOC) in wind_directions.csv - pass WIND_RANGE=\"<lo> <hi>\"" >&2; exit 2; fi
	burn_emulator -m run -d -vl $(VARLOC) -wr $$wind_range \
	    -c configs/varlocs/current.yaml \
	    -c "configs/$$arch/model.yaml" \
	    -c configs/varlocs/templates/run_smoke.yaml \
	    $(if $(TREATMENT_AREA),-ta $(TREATMENT_AREA)) \
	    $(if $(OUT_PATH),-o $(OUT_PATH)) \
	    $(if $(IGNITION_DENSITY),-id $(IGNITION_DENSITY)) \
	    $(if $(PT),-pt)

# $(1): varloc, empty for -all
ignitions_sbatch = $(MODEL_DIR)/slurm/submit_ignitions_all.sh $(if $(1),-v $(1)) $(if $(EXCLUDE),-x "$(EXCLUDE)") $(if $(NUM_IGNITIONS),-n $(NUM_IGNITIONS)) $(if $(OVERWRITE),-o) $(if $(INPUTS_VERSION),-i $(INPUTS_VERSION)) $(if $(IGNITIONS_VERSION),-g $(IGNITIONS_VERSION))

training-data: ## [direct|slurm] generate ignitions for one varloc (VARLOC= [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1])
	set -e
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -n "$(SLURM)" ]; then $(call ignitions_sbatch,$(VARLOC)); exit 0; fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	dv="$(or $(INPUTS_VERSION),$$(scripts/data_version.sh inputs_version))_$(or $(IGNITIONS_VERSION),$$(scripts/data_version.sh ignitions_version))"
	burn_emulator -m ignite -vl $(VARLOC) -dv "$$dv" $(if $(NUM_IGNITIONS),-ni $(NUM_IGNITIONS)) $(if $(OVERWRITE),-ow)

# skips complete varlocs unless OVERWRITE=1; continues past failures and exits nonzero
training-data-all: ## [direct|slurm] generate ignitions for every varloc in data/inputs/<inputs_version>/varlocs/all_varlocs.gpkg ([EXCLUDE=v1,v2 (slurm)] [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1])
	set -e
	if [ -n "$(SLURM)" ]; then
	    $(call ignitions_sbatch,)
	    exit 0
	fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	dv="$(or $(INPUTS_VERSION),$$(scripts/data_version.sh inputs_version))_$(or $(IGNITIONS_VERSION),$$(scripts/data_version.sh ignitions_version))"
	mapfile -t varlocs < <(scripts/gpkg_varlocs.sh "$${dv%%_*}")
	echo "$${#varlocs[@]} varlocs in gpkg"
	failed=()
	for varloc in "$${varlocs[@]}"; do
	    if [ -z "$(OVERWRITE)" ] && [ -f "data/training_data/$$varloc/$$dv/legalmax/outputs_table.csv" ]; then
	        echo "skipping $$varloc: $$dv already complete"
	        continue
	    fi
	    if ! $(MAKE) -C "$(CURDIR)" training-data VARLOC="$$varloc" OVERWRITE=1 $(if $(NUM_IGNITIONS),NUM_IGNITIONS=$(NUM_IGNITIONS)) $(if $(INPUTS_VERSION),INPUTS_VERSION=$(INPUTS_VERSION)) $(if $(IGNITIONS_VERSION),IGNITIONS_VERSION=$(IGNITIONS_VERSION)); then
	        echo "failed $$varloc" >&2
	        failed+=("$$varloc")
	    fi
	done
	if [ $${#failed[@]} -gt 0 ]; then echo "$${#failed[@]} varlocs failed: $${failed[*]}" >&2; exit 1; fi

shell: ## [direct] sync the model venv and open a shell with it activated
	set -e
	UV_PROJECT_ENVIRONMENT="$(VENV_NAME)" uv sync --project "$(MODEL_DIR)" --locked --inexact --extra data
	exec bash --rcfile <(echo 'source ~/.bashrc; source "$(CURDIR)/$(VENV)/bin/activate"')
