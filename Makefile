SHELL      := /bin/bash
.ONESHELL:

API_DIR    ?= burn-emulator-api
MODEL_DIR  ?= burn-emulator-model
DOCKER_DIR ?= deploy/docker
REGISTRY   ?= $(BURN_EMULATOR_ARTIFACT_STORE)
VERSION    ?= $(shell git rev-parse --short HEAD)

API_IMAGE   := $(REGISTRY)/burn-emulator-api/:$(VERSION)
MODEL_IMAGE := $(REGISTRY)/burn-emulator-model/:$(VERSION)

XLA_VERSION ?=
CKPT_NAME   ?=
CKPT_PATH   ?=
STAGE_DIR   ?=

.PHONY: build-api push-api build-model push-model build-all push-all deploy-api

build-api:
	docker build -f $(DOCKER_DIR)/Dockerfile.api -t $(API_IMAGE) $(API_DIR)

push-api: build-api
	docker push $(API_IMAGE)

build-model:
	docker build -f $(DOCKER_DIR)/Dockerfile.model \
		--build-arg XLA_VERSION=$(XLA_VERSION) \
		--build-arg CKPT_NAME=$(CKPT_NAME) \
		--build-arg CKPT_PATH=$(CKPT_PATH) \
		--build-arg STAGE_DIR=$(STAGE_DIR) \
		-t $(MODEL_IMAGE) $(MODEL_DIR)

push-model: build-model
	docker push $(MODEL_IMAGE)
