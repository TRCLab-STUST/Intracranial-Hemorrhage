include .env

.PHONY: help
.DEFAULT_GOAL := help

help: ## This help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: build-gpu ## Build Image From Dockerfile

# gpu-jupyter-image gpu-vscode-image
build-gpu: gpu-image gpu-jupyter-image

gpu-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):latest --target tf-gpu

gpu-jupyter-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):latest-jupyter --target tf-gpu-jupyter

#gpu-vscode-image:
#	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):latest-vscode --target tf-gpu-vscode