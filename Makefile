include .env

.PHONY: help
.DEFAULT_GOAL := help

help: ## This help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: build-gpu ## Build Image From Dockerfile

build-gpu: gpu-image gpu-jupyter-image gpu-vscode-image

gpu-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG) --target tf-gpu

gpu-jupyter-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG)-jupyter --target tf-gpu-jupyter

gpu-vscode-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG)-vscode --target tf-gpu-vscode