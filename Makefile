SHELL      := /bin/bash
.ONESHELL:

API_DIR    ?= burn-emulator-api
MODEL_DIR  ?= burn-emulator-model
RUNNER_DIR ?= burn-emulator-runner
VERSION    ?= $(shell git rev-parse --short HEAD)$(shell [ -z "$$(git status --porcelain)" ] || echo -dirty)

# BURN_EMULATOR_ARTIFACT_STORE / BURN_EMULATOR_MODELS_URI / BURN_EMULATOR_FUELS_URI
# must be exported by the caller (see burn-emulator-api/README.md / burn-emulator-model/README.md
# for what each points at). VERSION gets a -dirty suffix on an uncommitted tree; build-api/
# build-runner refuse to run with that suffix when BURN_EMULATOR_ENV is prod/production
# (unset BURN_EMULATOR_ENV does not trigger this check - dev/staging pushes get a -dirty
# tag instead of colliding with the last clean push).

API_IMAGE    := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-api:$(VERSION)
RUNNER_IMAGE := $(BURN_EMULATOR_ARTIFACT_STORE)/burn-emulator-runner:$(VERSION)

.PHONY: build-api push-api build-runner push-runner bundle-model bundle-model-all publish-model publish-model-all publish-fuels publish-topo train-all ignitions shell

build-api:
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - export it (see README.md)" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in prod|production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(API_DIR)/Dockerfile -t $(API_IMAGE) .

push-api: build-api
	docker push $(API_IMAGE)

build-runner:
	if [ -z "$(BURN_EMULATOR_ARTIFACT_STORE)" ]; then echo "error: BURN_EMULATOR_ARTIFACT_STORE is not set - export it (see README.md)" >&2; exit 2; fi
	case "$(BURN_EMULATOR_ENV)" in prod|production) case "$(VERSION)" in *-dirty) echo "error: refusing to build for $(BURN_EMULATOR_ENV) from a dirty git tree - commit first" >&2; exit 2 ;; esac ;; esac
	docker build -f $(RUNNER_DIR)/Dockerfile -t $(RUNNER_IMAGE) .

push-runner: build-runner
	docker push $(RUNNER_IMAGE)

bundle-model:
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	source "$(MODEL_DIR)/.venv/bin/activate"
	cd "$(MODEL_DIR)"
	burn_emulator -m bundle -c configs/varlocs/current.yaml -vl $(VARLOC)

bundle-model-all:
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) bundle-model VARLOC="$$varloc"
	done

publish-model:
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -z "$(BURN_EMULATOR_MODELS_URI)" ]; then echo "error: BURN_EMULATOR_MODELS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	if [ -n "$(BUNDLE_DIR)" ]; then
	    bundle_dir="$(BUNDLE_DIR)"
	else
	    current_yaml="$(MODEL_DIR)/configs/varlocs/current.yaml"
	    arch=$$(grep -oP '^architecture:[[:space:]]*\K\S+' "$$current_yaml")
	    dv=$$(grep -oP '^data_version:[[:space:]]*\K\S+' "$$current_yaml")
	    dv_iso=$$([[ "$$dv" =~ ^[0-9]{8}$$ ]] && echo "$$dv" || date -u -d "$$dv" +%Y%m%d)
	    bundle_dir="$(MODEL_DIR)/data/bundles/$(VARLOC)_$${arch}_$${dv_iso}"
	fi
	$(MODEL_DIR)/scripts/publish_model.sh $(VARLOC) "$$bundle_dir" $(BURN_EMULATOR_MODELS_URI)

publish-model-all:
	set -e
	mapfile -t varlocs < <(grep -vE '^[[:space:]]*$$' "$(MODEL_DIR)/configs/varlocs/varlocs.txt")
	for varloc in "$${varlocs[@]}"; do
	    $(MAKE) publish-model VARLOC="$$varloc"
	done

# FUELS_DIR should hold both baseline_*.tif and legalmax_*.tif;
publish-fuels:
	if [ -z "$(BURN_EMULATOR_FUELS_URI)" ]; then echo "error: BURN_EMULATOR_FUELS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	$(MODEL_DIR)/scripts/publish_fuels.sh $(FUELS_DIR) $(BURN_EMULATOR_FUELS_URI)

publish-topo:
	if [ -z "$(BURN_EMULATOR_FUELS_URI)" ]; then echo "error: BURN_EMULATOR_FUELS_URI is not set - export it (see README.md)" >&2; exit 2; fi
	$(MODEL_DIR)/scripts/publish_topo.sh $(TOPO_DIR) $(BURN_EMULATOR_FUELS_URI)

train-all:
	$(MODEL_DIR)/scripts/train_varlocs.sh

ignitions:
	if [ -z "$(VARLOC)" ] || [ -z "$(OUTPUTS_ROOT)" ]; then
	    echo "error: pass VARLOC=<varloc> OUTPUTS_ROOT=<dir>" >&2; exit 2
	fi
	$(MODEL_DIR)/scripts/ignite_inference.sh $(VARLOC) $(OUTPUTS_ROOT)

shell:
	@bash -c "source $(MODEL_DIR)/.venv/bin/activate && cd $(MODEL_DIR) && exec \$$SHELL"
