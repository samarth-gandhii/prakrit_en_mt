#!/bin/bash
#SBATCH --job-name=ibt_finetune_en_pr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:40
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

# Tokenizer source: cached indictrans2-en-indic-1B base model
# Find the snapshot hash directory inside the known cache location
HF_SNAPSHOTS_ROOT="/home/shrikant/.cache/huggingface/hub/models--ai4bharat--indictrans2-en-indic-1B/snapshots"
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
	BASE_MODEL_PATH="$(find "$HF_SNAPSHOTS_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
	# If no hash subdir found, use the snapshots dir itself (some HF versions store files there)
	if [[ -z "$BASE_MODEL_PATH" && -d "$HF_SNAPSHOTS_ROOT" ]]; then
		BASE_MODEL_PATH="$HF_SNAPSHOTS_ROOT"
	fi
fi

if [[ ! -d "$BASE_MODEL_PATH" ]]; then
	echo "ERROR: Could not find cached indictrans2-en-indic-1B tokenizer."
	echo "       Expected at: $HF_SNAPSHOTS_ROOT"
	echo "       Cache it on the login node first, or set BASE_MODEL_PATH manually."
	exit 1
fi

echo "Using local base model (tokenizer): $BASE_MODEL_PATH"
echo "Continuing from pretrained weights:  $PRETRAINED_MODEL"

python iterative_back_translation/scripts/run_finetune_en_pr.py \
	--epoch 1 \
	--model-dir iterative_back_translation/models/eng_to_prakrit_iterative_1epoch \
	--train iterative_back_translation/data/iteration1_parallel.tsv \
	--base-model "$BASE_MODEL_PATH" \
	--pretrained-model "$PRETRAINED_MODEL" \
	--local-files-only "$@"

echo "End Time: $(date)"
