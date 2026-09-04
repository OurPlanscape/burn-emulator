SHELL      := /bin/bash
.ONESHELL:

API_DIR    ?= burn-emulator-api
MODEL_DIR  ?= burn-emulator-model
RUNNER_DIR ?= burn-emulator-runner
REGISTRY   ?= $(BURN_EMULATOR_ARTIFACT_STORE)
VERSION    ?= $(shell git rev-parse --short HEAD)

API_IMAGE    := $(REGISTRY)/burn-emulator-api:$(VERSION)
RUNNER_IMAGE := $(REGISTRY)/burn-emulator-runner:$(VERSION)

.PHONY: build-api push-api build-runner push-runner publish-model publish-fuels

build-api:
	docker build -f $(API_DIR)/Dockerfile -t $(API_IMAGE) .

push-api: build-api
	docker push $(API_IMAGE)

build-runner:
	docker build -f $(RUNNER_DIR)/Dockerfile -t $(RUNNER_IMAGE) .

push-runner: build-runner
	docker push $(RUNNER_IMAGE)

BUNDLE_DIR ?= $(MODEL_DIR)/data/bundles/$(VARLOC)
publish-model:
	$(MODEL_DIR)/scripts/publish_model.sh $(VARLOC) $(BUNDLE_DIR) $(MODELS_URI)

publish-fuels:
	$(MODEL_DIR)/scripts/publish_fuels.sh $(FUELS_DIR) $(LAYER) $(FUELS_URI)
