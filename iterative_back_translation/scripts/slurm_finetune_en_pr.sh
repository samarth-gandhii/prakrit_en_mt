#!/bin/bash
#SBATCH --job-name=ibt_finetune_en_pr
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

# Model weights to continue fine-tuning from (define FIRST, used in fallback below)
PRETRAINED_MODEL="/home/shrikant/2026/Summer Internship/Samarth/Prakrit_MT/models/eng_to_prakrit_2-final"

# Force offline mode — never attempt HuggingFace network calls
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
	# Tokenizer MUST come from the base model (indictrans2-en-indic-1B).
	# The pretrained weights dir (eng_to_prakrit_2-final) has NO tokenizer files.
	# Look for it in the HuggingFace local cache first.
	HF_MODEL_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ai4bharat--indictrans2-en-indic-1B"
	CACHED_SNAPSHOT=""
	if [[ -d "$HF_MODEL_CACHE_ROOT/snapshots" ]]; then
		CACHED_SNAPSHOT="$(find "$HF_MODEL_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
	fi

	if [[ -n "$CACHED_SNAPSHOT" && -f "$CACHED_SNAPSHOT/config.json" ]]; then
		BASE_MODEL_PATH="$CACHED_SNAPSHOT"
	else
		# Cache not found — cannot proceed without the tokenizer source.
		# User must cache the model or set BASE_MODEL_PATH explicitly.
		echo "ERROR: Could not find a local snapshot of ai4bharat/indictrans2-en-indic-1B."
		echo "       Cache it first (with internet access), or set:"
		echo "         BASE_MODEL_PATH=/path/to/indictrans2-en-indic-1B sbatch $0"
		exit 1
	fi
fi

if [[ ! -d "$BASE_MODEL_PATH" ]]; then
	echo "ERROR: Local model not found at: $BASE_MODEL_PATH"
	echo "Set BASE_MODEL_PATH to the local model directory and re-submit."
	exit 1
fi

echo "Using local base model (tokenizer): $BASE_MODEL_PATH"
echo "Continuing from pretrained weights:  $PRETRAINED_MODEL"

python iterative_back_translation/scripts/run_finetune_en_pr.py \
	--epoch 5 \
	--model-dir iterative_back_translation/models/eng_to_prakrit_iterative \
	--train iterative_back_translation/data/iteration1_parallel.tsv \
	--base-model "$BASE_MODEL_PATH" \
	--pretrained-model "$PRETRAINED_MODEL" \
	--local-files-only "$@"

echo "End Time: $(date)"
