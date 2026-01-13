# === Hands-On LLM Training Makefile ===
# Quick commands for training, evaluation, and publishing

# Default target - show help when running `make` with no arguments
.DEFAULT_GOAL := help

# Colors for terminal output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
MAGENTA := \033[35m
BOLD := \033[1m
RESET := \033[0m

# Default Python interpreter
PYTHON := python3
MODULE := src.main

# Config name (Hydra base)
TRAIN_CONFIG := train
EVAL_CONFIG := eval
BASELINE_EVAL_CONFIG := eval_baseline

# Virtual environment (optional)
VENV := .venv

# === Remote training (configure for your setup) ===
# Uncomment and configure these for remote GPU training:
REMOTE_USER := ubuntu
REMOTE_HOST := 69.19.137.76
REMOTE_DIR := ~/Dev/new-movie-training

# Example configuration:
# REMOTE_USER := ubuntu
# REMOTE_HOST := 69.19.137.76 #192.168.1.100
# REMOTE_DIR := ~/Dev/new-llm-training

# === LakeFS Metadata Management (optional - configure if using) ===
# R2_ENDPOINT := https://your-r2-endpoint.com
# LAKEFS_BUCKET := your-lakefs-bucket
DOCKER_COMPOSE_PATH := infra/docker-compose

.PHONY: push ssh-train ssh-eval ssh-init push-metadata pull-metadata

# --- Sync local -> remote (incremental) ---
push:
	@if [ -z "$(REMOTE_USER)" ] || [ -z "$(REMOTE_HOST)" ]; then \
		echo "❌ Error: REMOTE_USER and REMOTE_HOST must be configured in Makefile"; \
		echo "📝 Edit the Makefile and set:"; \
		echo "   REMOTE_USER := your_username"; \
		echo "   REMOTE_HOST := your_gpu_server_ip"; \
		exit 1; \
	fi
	@echo "🚀 Syncing project to $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)..."
	rsync -avz --delete \
		--exclude '.venv' \
		--exclude '__pycache__' \
		--exclude 'data/tokenized' \
		--exclude 'outputs' \
		--exclude 'artifacts' \
		--exclude '.git' \
		. $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)
	@echo "✅ Sync complete."

# --- Initialize remote environment once ---
ssh-init:
	@if [ -z "$(REMOTE_USER)" ] || [ -z "$(REMOTE_HOST)" ]; then \
		echo "❌ Error: REMOTE_USER and REMOTE_HOST must be configured"; \
		exit 1; \
	fi
	@echo "🧰 Initializing remote environment on $(REMOTE_USER)@$(REMOTE_HOST)..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "cd $(REMOTE_DIR) && make init"
	@echo "✅ Remote environment ready."

# --- Run full training remotely (always safe) ---
ssh-train:
	@if [ -z "$(REMOTE_USER)" ] || [ -z "$(REMOTE_HOST)" ]; then \
		echo "❌ Error: REMOTE_USER and REMOTE_HOST must be configured"; \
		exit 1; \
	fi
	@echo "💻 Starting remote training on $(REMOTE_USER)@$(REMOTE_HOST)..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "\
		cd $(REMOTE_DIR) && \
		source .venv/bin/activate && \
		PYTHONPATH=. python3 -m src.main train --config-name $(TRAIN_CONFIG) \
	"

# --- Run training in background with tmux (survives disconnect) ---
ssh-train-bg:
	@if [ -z "$(REMOTE_USER)" ] || [ -z "$(REMOTE_HOST)" ]; then \
		echo "❌ Error: REMOTE_USER and REMOTE_HOST must be configured"; \
		exit 1; \
	fi
	@echo "💻 Starting remote training in tmux session 'train'..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "\
		tmux kill-session -t train 2>/dev/null || true && \
		tmux new-session -d -s train '\
			cd $(REMOTE_DIR) && \
			source .venv/bin/activate && \
			PYTHONPATH=. python3 -m src.main train --config-name $(TRAIN_CONFIG); \
			echo \"Training complete. Press enter to exit.\"; \
			read \
		' \
	"
	@echo "✅ Training started in background. Reconnect with: ssh $(REMOTE_USER)@$(REMOTE_HOST) -t 'tmux attach -t train'"

# Ensure venv/bin is in path if it exists
ifneq ("$(wildcard $(VENV)/bin/activate)","")
    ACTIVATE := . $(VENV)/bin/activate &&
else
    ACTIVATE :=
endif

# === Core commands ===

.PHONY: help init venv install clean train train-bg eval publish format lint convert tokenize merge gguf

help:
	@echo ""
	@echo "$(BOLD)🚀 LLM Training Pipeline$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Setup:$(RESET)"
	@echo "  $(GREEN)make init$(RESET)            Full setup: check system deps, create venv, install packages"
	@echo "  $(GREEN)make venv$(RESET)            Create Python virtual environment (.venv) only"
	@echo "  $(GREEN)make install$(RESET)         Install Python packages from requirements.txt only"
	@echo ""
	@echo "$(BOLD)$(CYAN)Data Preparation:$(RESET)"
	@echo "  $(GREEN)make convert$(RESET)         Convert ChatGPT JSON exports to training JSONL format"
	@echo "  $(GREEN)make tokenize$(RESET)        Pre-tokenize dataset for faster training"
	@echo "                       Usage: make tokenize TRAIN_CONFIG=train_gemma3_27b"
	@echo ""
	@echo "$(BOLD)$(CYAN)Training & Evaluation:$(RESET)"
	@echo "  $(GREEN)make train$(RESET)           Run LoRA fine-tuning on specified model"
	@echo "                       Usage: make train TRAIN_CONFIG=train_gemma3_27b"
	@echo "  $(GREEN)make train-bg$(RESET)        Run training in tmux (survives disconnect)"
	@echo "                       Usage: make train-bg TRAIN_CONFIG=train_gemma3_27b"
	@echo "  $(GREEN)make eval$(RESET)            Evaluate fine-tuned model against test set"
	@echo "                       Usage: make eval EVAL_CONFIG=eval_gemma3_27b"
	@echo "  $(GREEN)make eval-baseline$(RESET)   Evaluate base model (before fine-tuning)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Model Export:$(RESET)"
	@echo "  $(GREEN)make merge$(RESET)           Merge LoRA adapter weights into base model"
	@echo "  $(GREEN)make gguf$(RESET)            Convert merged model to GGUF format for llama.cpp"
	@echo "                       Usage: make gguf TRAIN_CONFIG=train_gemma3_27b QUANT=Q5_K_M"
	@echo "  $(GREEN)make publish$(RESET)         Copy model to local artifacts directory"
	@echo "  $(GREEN)make publish-hf$(RESET)      Upload model to HuggingFace Hub"
	@echo "                       Usage: HF_REPO_ID=user/repo make publish-hf TRAIN_CONFIG=..."
	@echo ""
	@echo "$(BOLD)$(CYAN)Remote Training:$(RESET)"
	@echo "  $(GREEN)make push$(RESET)            Sync project to remote GPU server via rsync"
	@echo "  $(GREEN)make ssh-init$(RESET)        Initialize venv and install deps on remote server"
	@echo "  $(GREEN)make ssh-train$(RESET)       Run training on remote server (foreground)"
	@echo "  $(GREEN)make ssh-train-bg$(RESET)    Run training on remote server in tmux (background)"
	@echo "  $(GREEN)make ssh-eval$(RESET)        Run evaluation on remote server"
	@echo ""
	@echo "$(BOLD)$(CYAN)Utilities:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)           Remove cache files and __pycache__ directories"
	@echo "  $(GREEN)make format$(RESET)          Format Python code with black"
	@echo "  $(GREEN)make lint$(RESET)            Run pylint checks on source code"
	@echo ""
	@echo "$(BOLD)$(CYAN)HuggingFace:$(RESET)"
	@echo "  $(GREEN)make hf-login$(RESET)        Login to HuggingFace Hub"
	@echo "  $(GREEN)make hf-cache$(RESET)        Set HF cache to ephemeral disk (saves space on root)"
	@echo "                       Usage: make hf-cache HF_CACHE_DIR=/ephemeral/.cache/huggingface"
	@echo ""

init:
	@echo "$(BOLD)$(CYAN)Updating package lists...$(RESET)"
	@sudo apt-get update
	@echo "$(BOLD)$(CYAN)Checking system dependencies...$(RESET)"
	@which python3 > /dev/null || (echo "$(YELLOW)Installing python3...$(RESET)" && sudo apt-get install -y python3)
	@PY_VERSION=$$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
		python3 -c "import venv" 2>/dev/null || (echo "$(YELLOW)Installing python$$PY_VERSION-venv...$(RESET)" && sudo apt-get install -y python$$PY_VERSION-venv)
	@which pip3 > /dev/null || (echo "$(YELLOW)Installing pip...$(RESET)" && sudo apt-get install -y python3-pip)
	@which git > /dev/null || (echo "$(YELLOW)Installing git...$(RESET)" && sudo apt-get install -y git)
	@dpkg -s build-essential > /dev/null 2>&1 || (echo "$(YELLOW)Installing build-essential...$(RESET)" && sudo apt-get install -y build-essential)
	@echo "$(GREEN)✅ System dependencies OK$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✅ Virtual environment created$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)Installing Python packages...$(RESET)"
	. $(VENV)/bin/activate && pip install --upgrade pip wheel setuptools && pip install -r requirements.txt
	@echo ""
	@echo "$(GREEN)✅ Setup complete!$(RESET)"
	@echo ""
	@echo "$(BOLD)Next steps:$(RESET)"
	@echo "  1. Activate venv:  $(CYAN)source .venv/bin/activate$(RESET)"
	@echo "  2. Tokenize data:  $(CYAN)make tokenize TRAIN_CONFIG=train_gemma3_27b$(RESET)"
	@echo "  3. Start training: $(CYAN)make train TRAIN_CONFIG=train_gemma3_27b$(RESET)"

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

install:
	$(ACTIVATE) pip install --upgrade pip && pip install -r requirements.txt

convert:
	@echo "🔄 Converting ChatGPT format files to JSONL..."
	$(ACTIVATE) $(PYTHON) scripts/convert_data.py
	@echo "✅ Conversion complete."

tokenize:
	@echo "🔤 Pre-tokenizing dataset (uses all CPU cores)..."
	$(ACTIVATE) $(PYTHON) scripts/tokenize_dataset.py -c configs/$(TRAIN_CONFIG).yaml
	@echo "✅ Tokenized dataset saved to data/tokenized"
	@echo "⚠️  Update your config: dataset_path: data/tokenized"

train:
	$(ACTIVATE) PYTHONPATH=./ $(PYTHON) -m $(MODULE) train --config-name $(TRAIN_CONFIG)

train-bg:
	@echo "$(BOLD)$(CYAN)Starting training in tmux session 'train'...$(RESET)"
	@tmux kill-session -t train 2>/dev/null || true
	@tmux new-session -d -s train "cd $(CURDIR) && source $(VENV)/bin/activate && PYTHONPATH=./ $(PYTHON) -m $(MODULE) train --config-name $(TRAIN_CONFIG); echo 'Training complete. Press enter to exit.'; read"
	@echo "$(GREEN)✅ Training started in background$(RESET)"
	@echo ""
	@echo "$(YELLOW)Reconnect with:$(RESET) tmux attach -t train"
	@echo "$(YELLOW)Detach with:$(RESET)    Ctrl+B then D"

eval:
	$(ACTIVATE) $(PYTHON) -m $(MODULE) eval --config-name $(EVAL_CONFIG)

eval-baseline:
	$(ACTIVATE) $(PYTHON) -m $(MODULE) eval --config-name $(BASELINE_EVAL_CONFIG)

# === Model Packaging ===
QUANT := Q5_K_M
LLAMA_CPP := $(HOME)/Dev/llama.cpp
CUDA_DEVICE := 0

# Derive model name from TRAIN_CONFIG (e.g., train_llama3_8b -> llama3-8b)
# Usage: make gguf TRAIN_CONFIG=train_llama3_8b
MODEL_NAME = $(shell echo "$(TRAIN_CONFIG)" | sed 's/^train_//;s/_/-/g')

merge:
	@echo "🔀 Merging LoRA adapter into base model..."
	$(ACTIVATE) $(PYTHON) scripts/merge_model.py outputs/runs/latest/model --cuda $(CUDA_DEVICE)
	@echo "✅ Merged model saved to outputs/runs/latest/merged"

gguf: #merge
	@if [ -z "$(TRAIN_CONFIG)" ] || [ "$(TRAIN_CONFIG)" = "train" ]; then \
		echo "❌ Error: TRAIN_CONFIG required. Usage: make gguf TRAIN_CONFIG=train_llama3_8b"; \
		exit 1; \
	fi
	@echo "📦 Converting to GGUF: $(MODEL_NAME) ($(QUANT))..."
	@if [ ! -f outputs/runs/latest/merged/$(MODEL_NAME)-f16.gguf ]; then \
		echo "Creating F16 GGUF..."; \
		$(ACTIVATE) $(PYTHON) $(LLAMA_CPP)/convert_hf_to_gguf.py \
			outputs/runs/latest/merged \
			--outfile outputs/runs/latest/merged/$(MODEL_NAME)-f16.gguf \
			--outtype f16; \
	else \
		echo "F16 GGUF already exists, skipping conversion..."; \
	fi
	$(LLAMA_CPP)/build/bin/llama-quantize \
		outputs/runs/latest/merged/$(MODEL_NAME)-f16.gguf \
		outputs/runs/latest/merged/$(MODEL_NAME)-$(QUANT).gguf \
		$(QUANT)
	@echo "✅ GGUF saved: outputs/runs/latest/merged/$(MODEL_NAME)-f16.gguf"
	@echo "✅ GGUF saved: outputs/runs/latest/merged/$(MODEL_NAME)-$(QUANT).gguf"

publish:
	$(ACTIVATE) $(PYTHON) -m $(MODULE) publish outputs/runs/latest/model

publish-hf:
	@if [ -z "$(TRAIN_CONFIG)" ] || [ "$(TRAIN_CONFIG)" = "train" ]; then \
		echo "❌ Error: TRAIN_CONFIG required. Usage: make publish-hf TRAIN_CONFIG=train_llama3_8b"; \
		exit 1; \
	fi
	@if [ -z "$(HF_REPO_ID)" ]; then \
		echo "❌ Error: HF_REPO_ID required. Export it first: export HF_REPO_ID=username/repo"; \
		exit 1; \
	fi
	@echo "📤 Publishing $(MODEL_NAME) to HuggingFace $(HF_REPO_ID)/$(MODEL_NAME)..."
	$(ACTIVATE) hf upload $(HF_REPO_ID) \
		outputs/runs/latest/merged \
		$(MODEL_NAME) \
		--commit-message "Upload $(MODEL_NAME) model files"
	@echo "✅ Published to https://huggingface.co/$(HF_REPO_ID)/tree/main/$(MODEL_NAME)"

clean:
	rm -rf outputs/__pycache__ */__pycache__ *.pyc
	@echo "Cleaned outputs and caches."

format:
	$(ACTIVATE) black src

lint:
	$(ACTIVATE) pylint src || echo "Lint issues (non-fatal)"

# === HuggingFace Setup ===
HF_CACHE_DIR := /ephemeral/.cache/huggingface

.PHONY: hf-login hf-cache

hf-login:
	@echo "$(BOLD)$(CYAN)Logging in to HuggingFace Hub...$(RESET)"
	$(ACTIVATE) huggingface-cli login
	@echo "$(GREEN)✅ HuggingFace login complete$(RESET)"

hf-cache:
	@echo "$(BOLD)$(CYAN)Setting HuggingFace cache to $(HF_CACHE_DIR)...$(RESET)"
	@REAL_USER=$${SUDO_USER:-$$USER}; \
		REAL_HOME=$$(eval echo ~$$REAL_USER); \
		sudo mkdir -p $(HF_CACHE_DIR) && \
		sudo chown -R $$REAL_USER:$$REAL_USER $(HF_CACHE_DIR) && \
		echo "" >> $$REAL_HOME/.bashrc && \
		echo "# HuggingFace cache location" >> $$REAL_HOME/.bashrc && \
		echo "export HF_HOME=$(HF_CACHE_DIR)" >> $$REAL_HOME/.bashrc && \
		echo "export HF_DATASETS_CACHE=$(HF_CACHE_DIR)/datasets" >> $$REAL_HOME/.bashrc && \
		echo "export TRANSFORMERS_CACHE=$(HF_CACHE_DIR)/transformers" >> $$REAL_HOME/.bashrc && \
		echo "$(GREEN)✅ HF cache configured for user $$REAL_USER$(RESET)" && \
		echo "" && \
		echo "$(YELLOW)Run: source ~/.bashrc$(RESET)" && \
		echo "" && \
		echo "export HF_HOME=$(HF_CACHE_DIR)" && \
		echo "export HF_DATASETS_CACHE=$(HF_CACHE_DIR)/datasets" && \
		echo "export TRANSFORMERS_CACHE=$(HF_CACHE_DIR)/transformers"
# --- Run evaluation remotely ---
ssh-eval:
	@if [ -z "$(REMOTE_USER)" ] || [ -z "$(REMOTE_HOST)" ]; then \
		echo "❌ Error: REMOTE_USER and REMOTE_HOST must be configured"; \
		exit 1; \
	fi
	@echo "🧪 Starting remote evaluation on $(REMOTE_USER)@$(REMOTE_HOST)..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "\
		cd $(REMOTE_DIR) && \
		source .venv/bin/activate && \
		PYTHONPATH=. python3 -m src.main eval --config-name $(EVAL_CONFIG) \
	"

.PHONY: ssh-eval-baseline
ssh-eval-baseline:
	@if [ -z "$(REMOTE_USER)" ] || [ -z "$(REMOTE_HOST)" ]; then \
		echo "❌ Error: REMOTE_USER and REMOTE_HOST must be configured"; \
		exit 1; \
	fi
	@echo "🧪 Starting remote baseline evaluation on $(REMOTE_USER)@$(REMOTE_HOST)..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "\
		cd $(REMOTE_DIR) && \
		source .venv/bin/activate && \
		PYTHONPATH=. python3 -m src.main eval --config-name $(BASELINE_EVAL_CONFIG) \
	"

# === LakeFS Metadata Management ===

push-metadata:
	@echo "🗄️  Backing up LakeFS metadata to R2..."
	docker compose -f $(DOCKER_COMPOSE_PATH)/docker-compose.yml --env-file $(DOCKER_COMPOSE_PATH)/base.env --env-file $(DOCKER_COMPOSE_PATH)/.env exec postgres pg_dump -U lakefs lakefs > postgres_backup.sql
	bash -c "set -a; source $(DOCKER_COMPOSE_PATH)/.env; aws s3 cp postgres_backup.sql s3://$(LAKEFS_BUCKET)/metadata/postgres_backup.sql --endpoint-url $(R2_ENDPOINT)"
	rm postgres_backup.sql
	@echo "✅ Metadata pushed to R2"

pull-metadata:
	@echo "⬇️  Restoring LakeFS metadata from R2..."
	bash -c "set -a; source $(DOCKER_COMPOSE_PATH)/.env; aws s3 cp s3://$(LAKEFS_BUCKET)/metadata/postgres_backup.sql postgres_backup.sql --endpoint-url $(R2_ENDPOINT)"
	docker compose -f $(DOCKER_COMPOSE_PATH)/docker-compose.yml --env-file $(DOCKER_COMPOSE_PATH)/base.env --env-file $(DOCKER_COMPOSE_PATH)/.env up -d postgres
	@echo "⏳ Waiting for Postgres to start..."
	@sleep 15
	docker compose -f $(DOCKER_COMPOSE_PATH)/docker-compose.yml --env-file $(DOCKER_COMPOSE_PATH)/base.env --env-file $(DOCKER_COMPOSE_PATH)/.env exec -T postgres psql -U lakefs lakefs < postgres_backup.sql
	rm postgres_backup.sql
	@echo "✅ Metadata restored from R2"
