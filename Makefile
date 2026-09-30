SHELL      := /bin/bash
.ONESHELL:
.SILENT:
MAKEFLAGS  += --no-print-directory

API_DIR    ?= burn-emulator-api
MODEL_DIR  ?= burn-emulator-model
RUNNER_DIR ?= burn-emulator-runner
VERSION    ?= $(shell git rev-parse --short HEAD)$(shell [ -z "$$(git status --porcelain)" ] || echo -dirty)

# BURN_EMULATOR_ARTIFACT_STORE / BURN_EMULATOR_MODELS_URI / BURN_EMULATOR_INPUTS_URI
# must be exported by the caller (see burn-emulator-api/README.md / burn-emulator-model/README.md
# for what each points at). VERSION gets a -dirty suffix on an uncommitted tree; build-api/
# build-runner refuse to run with that suffix when BURN_EMULATOR_ENV is prod/production
# (unset BURN_EMULATOR_ENV does not trigger this check - dev/staging pushes get a -dirty
# tag instead of colliding with the last clean push).

API_IMAGE    := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-api:$(VERSION)
RUNNER_IMAGE := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-runner:$(VERSION)

.DEFAULT_GOAL := help

.PHONY: help build-api push-api build-runner push-runner valid-varlocs bundle-model bundle-model-all publish-model publish-model-all publish-inputs train-all inference inference-all smoke ignitions ignitions-all shell

help: ## show this help
	awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build-api: ## build the API image
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - export it (see README.md)" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in prod|production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(API_DIR)/Dockerfile -t $(API_IMAGE) .

push-api: build-api ## build and push the API image
	docker push $(API_IMAGE)

build-runner: ## build the runner image
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - export it (see README.md)" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in prod|production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(RUNNER_DIR)/Dockerfile -t $(RUNNER_IMAGE) .

push-runner: build-runner ## build and push the runner image
	docker push $(RUNNER_IMAGE)

valid-varlocs: ## filter varlocs gpkg to the valid set
	source "$(MODEL_DIR)/.venv/bin/activate"
	cd "$(MODEL_DIR)"
	python scripts/filter_varlocs_gpkg.py

bundle-model: ## bundle one model (VARLOC=)
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(MODEL_DIR)/.venv/bin/activate"
	cd "$(MODEL_DIR)"
	burn_emulator -m bundle -c configs/varlocs/current.yaml -vl $(VARLOC)

bundle-model-all: valid-varlocs ## bundle every varloc in varlocs.txt
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) bundle-model VARLOC="$$varloc"
	done

publish-model: ## publish one model bundle (VARLOC= [BUNDLE_DIR=])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -z "$(BURN_EMULATOR_MODELS_URI)" ]; then echo "error: BURN_EMULATOR_MODELS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	if [ -n "$(BUNDLE_DIR)" ]; then
	    bundle_dir="$(BUNDLE_DIR)"
	else
	    current_yaml="$(MODEL_DIR)/configs/varlocs/current.yaml"
	    arch=$$(grep -oP '^architecture:[[:space:]]*\K\S+' "$$current_yaml")
	    dv=$$(grep -oP '^data_version:[[:space:]]*\K\S+' "$$current_yaml")
	    dv_iso=$$([[ "$$dv" =~ ^[0-9]{8}$$ ]] && echo "$$dv" || date -u -d "$$dv" +%Y%m%d)
	    varloc="$(VARLOC)"
	    bundle_dir="$(MODEL_DIR)/data/bundles/$${varloc^^}_$${arch}_$${dv_iso}"
	fi
	$(MODEL_DIR)/scripts/publish_model.sh $(VARLOC) "$$bundle_dir" $(BURN_EMULATOR_MODELS_URI)

publish-model-all: ## publish every varloc in varlocs.txt
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) publish-model VARLOC="$$varloc"
	done

# FUELS_DIR holds both baseline_*.tif and legalmax_*.tif, TOPO_DIR the topo tifs;
# both land under one DATA_VERSION (DDMonYYYY or YYYYMMDD)
publish-inputs: ## publish input rasters (DATA_VERSION= FUELS_DIR= TOPO_DIR=)
	if [ -z "$(BURN_EMULATOR_INPUTS_URI)" ]; then echo "error: BURN_EMULATOR_INPUTS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	if [ -z "$(DATA_VERSION)" ] || [ -z "$(FUELS_DIR)" ] || [ -z "$(TOPO_DIR)" ]; then
	    echo "error: pass DATA_VERSION=<version> FUELS_DIR=<dir> TOPO_DIR=<dir>" >&2; exit 2
	fi
	$(MODEL_DIR)/scripts/publish_inputs.sh $(DATA_VERSION) $(FUELS_DIR) $(TOPO_DIR) $(BURN_EMULATOR_INPUTS_URI)

train-all: ## train every varloc
	$(MODEL_DIR)/scripts/train_varlocs.sh

inference: ## run inference for one varloc (VARLOC= OUTPUTS_ROOT=)
	if [ -z "$(VARLOC)" ] || [ -z "$(OUTPUTS_ROOT)" ]; then
	    echo "error: pass VARLOC=<varloc> OUTPUTS_ROOT=<dir>" >&2; exit 2
	fi
	$(MODEL_DIR)/scripts/ignite_inference.sh $(VARLOC) $(OUTPUTS_ROOT)

# expects one scenario root per varloc at OUTPUTS_ROOT/<varloc>
inference-all: ## run inference for every varloc (OUTPUTS_ROOT=)
	set -e
	if [ -z "$(OUTPUTS_ROOT)" ]; then echo "error: pass OUTPUTS_ROOT=<dir>" >&2; exit 2; fi
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) inference VARLOC="$$varloc" OUTPUTS_ROOT="$(OUTPUTS_ROOT)/$$varloc"
	done

smoke: ## run the smoke test for one varloc in debug mode (VARLOC= [WIND_RANGE="lo hi"] [OUT_PATH=])
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(MODEL_DIR)/.venv/bin/activate"
	cd "$(MODEL_DIR)"
	arch=$$(grep -oP '^architecture:[[:space:]]*\K\S+' configs/varlocs/current.yaml)
	wind_range="$(WIND_RANGE)"
	if [ -z "$$wind_range" ]; then
	    wind_key=$$(sed -E 's/^([A-Za-z]+)([0-9]+)$$/\U\1_\2/' <<< "$(VARLOC)")
	    wind_range=$$(awk -F, -v k="$$wind_key" '$$1 == k {print $$2, $$3}' data/training_data/wind_directions.csv)
	fi
	if [ -z "$$wind_range" ]; then echo "error: no wind range for $(VARLOC) in wind_directions.csv - pass WIND_RANGE=\"<lo> <hi>\"" >&2; exit 2; fi
	burn_emulator -m run -d -vl $(VARLOC) -wr $$wind_range \
	    -c configs/varlocs/current.yaml \
	    -c "configs/$$arch/model.yaml" \
	    -c configs/varlocs/templates/run_smoke.yaml \
	    $(if $(OUT_PATH),-o $(OUT_PATH))

ignitions: ## generate ignitions for one varloc (VARLOC= [NUM_IGNITIONS=] [OVERWRITE=1])
	set -e
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(MODEL_DIR)/.venv/bin/activate"
	cd "$(MODEL_DIR)"
	dv=$$(grep -oP '^data_version:[[:space:]]*\K\S+' configs/varlocs/current.yaml)
	burn_emulator -m ignite -vl $(VARLOC) -dv "$$dv" $(if $(NUM_IGNITIONS),-ni $(NUM_IGNITIONS)) $(if $(OVERWRITE),-ow)
	varlocs_txt=configs/varlocs/varlocs.txt
	if ! grep -qxF "$(VARLOC)" "$$varlocs_txt"; then
	    { grep -vE '^[[:space:]]*$$' "$$varlocs_txt"; echo "$(VARLOC)"; } | LC_ALL=C sort -u > "$$varlocs_txt.tmp"
	    mv "$$varlocs_txt.tmp" "$$varlocs_txt"
	    echo "added $(VARLOC) to $$varlocs_txt"
	fi

ignitions-all: ## generate ignitions for every varloc in varlocs.txt
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) ignitions VARLOC="$$varloc"
	done

shell: ## sync the model venv and open a shell with it activated
	set -e
	uv sync --project "$(MODEL_DIR)" --locked --inexact --extra data
	exec bash --rcfile <(echo 'source ~/.bashrc; source "$(CURDIR)/$(MODEL_DIR)/.venv/bin/activate"')
