SHELL      := /bin/bash
.ONESHELL:

API_DIR    ?= burn-emulator-api
MODEL_DIR  ?= burn-emulator-model
RUNNER_DIR ?= burn-emulator-runner
REGISTRY   ?= $(BURN_EMULATOR_ARTIFACT_STORE)
VERSION    ?= $(shell git rev-parse --short HEAD)

API_IMAGE    := $(REGISTRY)/burn-emulator-api:$(VERSION)
RUNNER_IMAGE := $(REGISTRY)/burn-emulator-runner:$(VERSION)

.PHONY: build-api push-api build-runner push-runner publish-model publish-fuels publish-topo

build-api:
	docker build -f $(API_DIR)/Dockerfile -t $(API_IMAGE) .

push-api: build-api
	docker push $(API_IMAGE)

build-runner:
	docker build -f $(RUNNER_DIR)/Dockerfile -t $(RUNNER_IMAGE) .

push-runner: build-runner
	docker push $(RUNNER_IMAGE)

# BUNDLE_DIR defaults to the bundle for VARLOC built from the currently active
# architecture/data_version in configs/varlocs/current.yaml, i.e. where
# `burn_emulator -m bundle -c configs/varlocs/current.yaml -vl VARLOC` writes it.
publish-model:
	if [ -z "$(VARLOC)" ]; then echo "error: pass VARLOC=<varloc>" >&2; exit 2; fi
	if [ -n "$(BUNDLE_DIR)" ]; then
	    bundle_dir="$(BUNDLE_DIR)"
	else
	    current_yaml="$(MODEL_DIR)/configs/varlocs/current.yaml"
	    arch=$$(grep -oP '^architecture:[[:space:]]*\K\S+' "$$current_yaml")
	    dv=$$(grep -oP '^data_version:[[:space:]]*\K\S+' "$$current_yaml")
	    dv_iso=$$([[ "$$dv" =~ ^[0-9]{8}$$ ]] && echo "$$dv" || date -u -d "$$dv" +%Y%m%d)
	    bundle_dir="$(MODEL_DIR)/data/bundles/$(VARLOC)_$${arch}_$${dv_iso}"
	fi
	$(MODEL_DIR)/scripts/publish_model.sh $(VARLOC) "$$bundle_dir" $(MODELS_URI)

# FUELS_DIR holds both baseline_*.tif and legalmax_*.tif;
publish-fuels:
	$(MODEL_DIR)/scripts/publish_fuels.sh $(FUELS_DIR) $(FUELS_URI)

publish-topo:
	$(MODEL_DIR)/scripts/publish_topo.sh $(TOPO_DIR) $(FUELS_URI)
