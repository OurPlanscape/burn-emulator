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

# BURN_EMULATOR_ARTIFACT_STORE / BURN_EMULATOR_MODELS_URI / BURN_EMULATOR_INPUTS_URI
# must be exported by the caller (see burn-emulator-api/README.md / burn-emulator-model/README.md
# for what each points at). VERSION gets a -dirty suffix on an uncommitted tree; api-image/
# runner-image refuse to run with that suffix when BURN_EMULATOR_ENV is production
# (unset BURN_EMULATOR_ENV does not trigger this check - dev/staging pushes get a -dirty
# tag instead of colliding with the last clean push).

API_IMAGE    := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-api:$(VERSION)
RUNNER_IMAGE := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-runner:$(VERSION)

.DEFAULT_GOAL := help

.PHONY: help api-image api-release runner-image runner-release valid-varlocs model-bundle model-bundle-all model-release model-release-all inputs-release varlocs-release train-all train-release train-release-all inference inference-all smoke training-data training-data-all shell

help: ## show this help; [metal] runs here, [slurm] submits to the cluster, [metal|slurm] picks via SLURM=1
	awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

api-image: ## [metal] build the API image
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - export it (see README.md)" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(API_DIR)/Dockerfile -t $(API_IMAGE) .

api-release: api-image ## [metal] build and push the API image
	docker push $(API_IMAGE)

runner-image: ## [metal] build the runner image
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - export it (see README.md)" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(RUNNER_DIR)/Dockerfile -t $(RUNNER_IMAGE) .

runner-release: runner-image ## [metal] build and push the runner image
	docker push $(RUNNER_IMAGE)

valid-varlocs: ## [metal] filter varlocs gpkg to the valid set
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	python scripts/filter_varlocs_gpkg.py -v "$(abspath $(VARLOCS_TXT))" -o "$(abspath $(VARLOCS_GPKG))"

model-bundle: ## [metal] bundle one model (VARLOC=)
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	burn_emulator -m bundle -c configs/varlocs/current.yaml -vl $(VARLOC)

model-bundle-all: valid-varlocs ## [metal] bundle every varloc in varlocs.txt
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	if [ $${#varlocs[@]} -eq 0 ]; then echo "error: $(MODEL_DIR)/configs/varlocs/varlocs.txt lists no varlocs" >&2; exit 2; fi
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) model-bundle VARLOC="$$varloc"
	done

model-release: ## [metal] publish one model bundle (VARLOC= [BUNDLE_DIR=])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -z "$(BURN_EMULATOR_MODELS_URI)" ]; then echo "error: BURN_EMULATOR_MODELS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	if [ -n "$(BUNDLE_DIR)" ]; then
	    bundle_dir="$(BUNDLE_DIR)"
	else
	    arch=$$(grep -oP '^architecture:[[:space:]]*\K\S+' "$(MODEL_DIR)/configs/varlocs/current.yaml")
	    dv=$$($(MODEL_DIR)/scripts/data_version.sh)
	    varloc="$(VARLOC)"
	    bundle_dir="$(MODEL_DIR)/data/bundles/$${varloc^^}_$${arch}_$${dv}"
	fi
	$(MODEL_DIR)/scripts/publish_model.sh $(VARLOC) "$$bundle_dir" $(BURN_EMULATOR_MODELS_URI)

model-release-all: ## [metal] publish every varloc in varlocs.txt
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) model-release VARLOC="$$varloc"
	done

# FUELS_DIR holds {baseline,legalmax}/<layer>.tif, TOPO_DIR the topo tifs,
# VARLOCS_GPKG the valid-varlocs output and VARLOCS_TXT the api allow-list; all land under one DATA_VERSION
# (YYYYMMDD); defaults follow current.yaml's inputs_version and the dirs -m ignite clips from
VARLOCS_GPKG ?= $(MODEL_DIR)/data/outputs/valid_varlocs.gpkg
VARLOCS_TXT  ?= $(MODEL_DIR)/configs/varlocs/varlocs.txt
FBFM_MAP     ?= $(MODEL_DIR)/configs/fbfm_behavior_adjectives.csv

inputs-release: ## [metal] publish input rasters + varlocs ([DATA_VERSION=] [FUELS_DIR=] [TOPO_DIR=] [VARLOCS_GPKG= VARLOCS_TXT=] [FBFM_MAP=], default: current.yaml inputs_version)
	set -e
	if [ -z "$(BURN_EMULATOR_INPUTS_URI)" ]; then echo "error: BURN_EMULATOR_INPUTS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	data_version="$(DATA_VERSION)"
	[ -n "$$data_version" ] || data_version=$$($(MODEL_DIR)/scripts/data_version.sh inputs_version)
	fuels_dir="$(or $(FUELS_DIR),$(MODEL_DIR)/data/training_data/fuels_$$data_version)"
	topo_dir="$(or $(TOPO_DIR),$(MODEL_DIR)/data/training_data/topo_$$data_version)"
	$(MODEL_DIR)/scripts/publish_inputs.sh "$$data_version" "$$fuels_dir" "$$topo_dir" $(VARLOCS_GPKG) $(VARLOCS_TXT) $(FBFM_MAP) $(BURN_EMULATOR_INPUTS_URI)

varlocs-release: valid-varlocs ## [metal] rebuild the valid-varlocs gpkg, then replace only the varlocs txt + gpkg of a published data_version ([VARLOCS_TXT=] [VARLOCS_GPKG=] [DATA_VERSION=], default: current)
	if [ -z "$(BURN_EMULATOR_INPUTS_URI)" ]; then echo "error: BURN_EMULATOR_INPUTS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	$(MODEL_DIR)/scripts/publish_varlocs.sh "$(VARLOCS_TXT)" "$(VARLOCS_GPKG)" "$(DATA_VERSION)" "$(BURN_EMULATOR_INPUTS_URI)"

train-all: ## [metal|slurm] train every varloc with complete training data; adds each to varlocs.txt once trained ([SLURM=1 [EXCLUDE=v1,v2] [NODES=n1,n2]])
	if [ -n "$(SLURM)" ]; then
	    $(MODEL_DIR)/slurm/submit_train_all.sh $(if $(EXCLUDE),-x "$(EXCLUDE)") $(NODES)
	elif [ -n "$(EXCLUDE)" ]; then
	    echo "error: EXCLUDE needs SLURM=1" >&2; exit 2
	else
	    $(MODEL_DIR)/scripts/train_all.sh
	fi

# slurm: trains across the GPU nodes; each varloc is bundled + published once its training succeeds
train-release: ## [slurm] train + bundle + publish for one varloc (VARLOC= [NODES=n1,n2])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	$(MODEL_DIR)/slurm/submit_train_all.sh -p -v $(VARLOC) $(NODES)

train-release-all: ## [slurm] train + bundle + publish for every trainable varloc ([EXCLUDE=v1,v2] [NODES=n1,n2])
	$(MODEL_DIR)/slurm/submit_train_all.sh -p $(if $(EXCLUDE),-x "$(EXCLUDE)") $(NODES)

inference: ## [metal] run inference for one varloc (VARLOC= OUTPUTS_ROOT=)
	if [ -z "$(VARLOC)" ] || [ -z "$(OUTPUTS_ROOT)" ]; then
	    echo "error: pass VARLOC=<varloc> OUTPUTS_ROOT=<dir>" >&2; exit 2
	fi
	$(MODEL_DIR)/scripts/ignite_inference.sh $(VARLOC) $(OUTPUTS_ROOT)

# expects one scenario root per varloc at OUTPUTS_ROOT/<varloc>
inference-all: ## [metal] run inference for every varloc (OUTPUTS_ROOT=)
	set -e
	if [ -z "$(OUTPUTS_ROOT)" ]; then echo "error: pass OUTPUTS_ROOT=<dir>" >&2; exit 2; fi
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) inference VARLOC="$$varloc" OUTPUTS_ROOT="$(OUTPUTS_ROOT)/$$varloc"
	done

smoke: ## [metal] run the smoke test for one varloc in debug mode (VARLOC= [WIND_RANGE="lo hi"] [TREATMENT_AREA=] [OUT_PATH=] [PT=1])
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
	    $(if $(PT),-pt)

# INPUTS_VERSION / IGNITIONS_VERSION (YYYYMMDD) default to current.yaml; SLURM=1 runs submit_ignitions_all.sh: a
# slurm/ignitions.slurm worker working through a queue file; $(1) is the varloc, empty for -all
ignitions_sbatch = $(MODEL_DIR)/slurm/submit_ignitions_all.sh $(if $(1),-v $(1)) $(if $(EXCLUDE),-x "$(EXCLUDE)") $(if $(NUM_IGNITIONS),-n $(NUM_IGNITIONS)) $(if $(OVERWRITE),-o) $(if $(INPUTS_VERSION),-i $(INPUTS_VERSION)) $(if $(IGNITIONS_VERSION),-g $(IGNITIONS_VERSION))

training-data: ## [metal|slurm] generate ignitions for one varloc (VARLOC= [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1])
	set -e
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -n "$(SLURM)" ]; then $(call ignitions_sbatch,$(VARLOC)); exit 0; fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	dv="$(or $(INPUTS_VERSION),$$(scripts/data_version.sh inputs_version))_$(or $(IGNITIONS_VERSION),$$(scripts/data_version.sh ignitions_version))"
	burn_emulator -m ignite -vl $(VARLOC) -dv "$$dv" $(if $(NUM_IGNITIONS),-ni $(NUM_IGNITIONS)) $(if $(OVERWRITE),-ow)

# skips varlocs whose training data for the current data_version is complete (legalmax outputs_table.csv) unless OVERWRITE=1;
# keeps going past varlocs that fail and exits nonzero listing them
training-data-all: ## [metal|slurm] generate ignitions for every varloc in the varlocs gpkg ([EXCLUDE=v1,v2 (slurm)] [INPUTS_VERSION=] [IGNITIONS_VERSION=] [NUM_IGNITIONS=] [OVERWRITE=1] [SLURM=1])
	set -e
	if [ -n "$(SLURM)" ]; then $(call ignitions_sbatch,); exit 0; fi
	source "$(VENV)/bin/activate"
	cd "$(MODEL_DIR)"
	dv="$(or $(INPUTS_VERSION),$$(scripts/data_version.sh inputs_version))_$(or $(IGNITIONS_VERSION),$$(scripts/data_version.sh ignitions_version))"
	mapfile -t varlocs < <(scripts/gpkg_varlocs.sh)
	echo "$${#varlocs[@]} varlocs in gpkg"
	failed=()
	for varloc in "$${varlocs[@]}"; do
	    if [ -z "$(OVERWRITE)" ] && [ -f "data/training_data/$$varloc/$$dv/legalmax/outputs_table.csv" ]; then
	        echo "skipping $$varloc: $$dv already complete"
	        continue
	    fi
	    # a failing varloc (e.g. no fuels coverage) doesn't stop the rest; reported at the end
	    if ! $(MAKE) -C "$(CURDIR)" training-data VARLOC="$$varloc" OVERWRITE=1 $(if $(NUM_IGNITIONS),NUM_IGNITIONS=$(NUM_IGNITIONS)) $(if $(INPUTS_VERSION),INPUTS_VERSION=$(INPUTS_VERSION)) $(if $(IGNITIONS_VERSION),IGNITIONS_VERSION=$(IGNITIONS_VERSION)); then
	        echo "failed $$varloc" >&2
	        failed+=("$$varloc")
	    fi
	done
	if [ $${#failed[@]} -gt 0 ]; then echo "$${#failed[@]} varlocs failed: $${failed[*]}" >&2; exit 1; fi

shell: ## [metal] sync the model venv and open a shell with it activated
	set -e
	UV_PROJECT_ENVIRONMENT="$(VENV_NAME)" uv sync --project "$(MODEL_DIR)" --locked --inexact --extra data
	exec bash --rcfile <(echo 'source ~/.bashrc; source "$(CURDIR)/$(VENV)/bin/activate"')
