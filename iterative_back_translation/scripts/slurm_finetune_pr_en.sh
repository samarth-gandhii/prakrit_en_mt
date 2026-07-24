#!/bin/bash
#SBATCH --job-name=ibt_finetune_pr_en
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:60
#SBATCH --mem=64G
#SBATCH -t 03-00:00:00
#SBATCH --output=iterative_back_translation/logs/%x_%j.out
#SBATCH --error=iterative_back_translation/logs/%x_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="${REPO_ROOT}/iterative_back_translation"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${REPO_ROOT}"

source hpc_env/bin/activate

if [[ -z "${HF_TOKEN:-}" ]]; then
	if [[ -n "${HF_TOKEN_FILE:-}" && -f "${HF_TOKEN_FILE}" ]]; then
		export HF_TOKEN="$(tr -d ' \t\r\n' < "${HF_TOKEN_FILE}")"
	elif [[ -f "$HOME/.config/huggingface/token" ]]; then
		export HF_TOKEN="$(tr -d ' \t\r\n' < "$HOME/.config/huggingface/token")"
	fi
fi

echo "Start Time: $(date)"

# Detect local base model (used for TOKENIZER only)
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
	HF_MODEL_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ai4bharat--indictrans2-indic-en-1B"
	CACHED_SNAPSHOT=""
	if [[ -d "$HF_MODEL_CACHE_ROOT/snapshots" ]]; then
		CACHED_SNAPSHOT="$(find "$HF_MODEL_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
	fi

	if [[ -n "$CACHED_SNAPSHOT" && -f "$CACHED_SNAPSHOT/config.json" ]]; then
		BASE_MODEL_PATH="$CACHED_SNAPSHOT"
	else
		BASE_MODEL_PATH="$PWD/indictrans2_base_model"
	fi
fi

# Model weights to continue fine-tuning from
PRETRAINED_MODEL="models/prakrit_to_eng_v2-final"

if [[ -d "$BASE_MODEL_PATH" ]]; then
	export HF_HUB_OFFLINE="1"
	export TRANSFORMERS_OFFLINE="1"
	echo "Using local base model (tokenizer): $BASE_MODEL_PATH"
	echo "Continuing from pretrained weights: $PRETRAINED_MODEL"
	python iterative_back_translation/scripts/run_finetune_pr_en.py \
		--epoch 5 \
		--model-dir iterative_back_translation/models/prakrit_to_eng_iterative \
		--train iterative_back_translation/data/iteration1_parallel.tsv \
		--base-model "$BASE_MODEL_PATH" \
		--pretrained-model "$PRETRAINED_MODEL" \
		--local-files-only "$@"
else
	echo "WARN: Local base model not found at $BASE_MODEL_PATH; attempting online download."
	python iterative_back_translation/scripts/run_finetune_pr_en.py \
		--epoch 5 \
		--model-dir iterative_back_translation/models/prakrit_to_eng_iterative \
		--train iterative_back_translation/data/iteration1_parallel.tsv \
		--pretrained-model "$PRETRAINED_MODEL" "$@"
fi

echo "End Time: $(date)"
