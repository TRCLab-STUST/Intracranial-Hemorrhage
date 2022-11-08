include .env

.PHONY: help
.DEFAULT_GOAL := help

help: ## This help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

runtime: ## Run default runtime
	docker run -it --rm -v $(shell pwd):/ich $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG)

tensorboard: ## Up tensorboard serivce on port 8081
	docker run -it --rm -v $(shell pwd):/ich -p 8087:6007 $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG) tensorboard --logdir ./ICH-Segmentation/logs --port 6007 --host 0.0.0.0 

jupyter: ## Up jupyter serivce on port 8081
	docker run -it --rm -v $(shell pwd):/ich -p 8081:8888 -d $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG)-jupyter

mount: ## Mount NAS Dataset via NFS
	sudo mount -t nfs $(NFS_HOST):$(NFS_ICH420_DIR) ./ICH-Segmentation/datasets/ICH_420
	sudo mount -t nfs $(NFS_HOST):$(NFS_ICH127_DIR) ./ICH-Segmentation/datasets/ICH_127

umount: ## Unmount Dataset
	sudo umount ./ICH-Segmentation/datasets/ICH_420
	sudo umount ./ICH-Segmentation/datasets/ICH_127

build: build-gpu ## Build Image From Dockerfile

build-gpu: gpu-image gpu-jupyter-image gpu-vscode-image

gpu-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG) --target tf-gpu

gpu-jupyter-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG)-jupyter --target tf-gpu-jupyter

gpu-vscode-image:
	docker build . -t $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_VERSION_TAG)-vscode --target tf-gpu-vscode
