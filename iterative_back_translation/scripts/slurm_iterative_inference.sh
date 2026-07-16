#!/bin/bash
#SBATCH --job-name=ibt_inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:16
#SBATCH --mem=32G
#SBATCH -t 03-00:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

PROJECT_ROOT="${SLURM_SUBMIT_DIR}/.."
REPO_ROOT="${PROJECT_ROOT}/.."
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

if [[ -d "$BASE_MODEL_PATH" ]]; then
	export HF_HUB_OFFLINE="1"
	export TRANSFORMERS_OFFLINE="1"
	echo "Using local base model: $BASE_MODEL_PATH"
	python iterative_back_translation/scripts/run_inference_iterative.py \
		--input iterative_back_translation/data/prakrit_50k.txt \
		--output iterative_back_translation/data/iteration1_parallel.tsv \
		--model-dir models/prakrit_to_eng_v2-final \
		--base-model "$BASE_MODEL_PATH" \
		--local-files-only "$@"
else
	echo "WARN: Local base model not found at $BASE_MODEL_PATH; attempting online download."
	python iterative_back_translation/scripts/run_inference_iterative.py \
		--input iterative_back_translation/data/prakrit_50k.txt \
		--output iterative_back_translation/data/iteration1_parallel.tsv \
		--model-dir models/prakrit_to_eng_v2-final "$@"
fi

echo "End Time: $(date)"
