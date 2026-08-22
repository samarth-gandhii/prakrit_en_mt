#!/bin/bash
#SBATCH --job-name=ibt_infer_en2pr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:20
#SBATCH --mem=32G
#SBATCH -t 03-00:00:00
#SBATCH --output=iterative_back_translation/logs/%x_%j.out
#SBATCH --error=iterative_back_translation/logs/%x_%j.err

# ============================================================
# English → Prakrit inference
# Takes monolingual English, infers Prakrit via En→Pr model,
# produces parallel TSV used to fine-tune the Pr→En model.
# ============================================================

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="${REPO_ROOT}/iterative_back_translation"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${REPO_ROOT}"

# Force unbuffered Python output so logs appear in real-time
export PYTHONUNBUFFERED=1

source hpc_env/bin/activate

if [[ -z "${HF_TOKEN:-}" ]]; then
	if [[ -n "${HF_TOKEN_FILE:-}" && -f "${HF_TOKEN_FILE}" ]]; then
		export HF_TOKEN="$(tr -d ' \t\r\n' < "${HF_TOKEN_FILE}")"
	elif [[ -f "$HOME/.config/huggingface/token" ]]; then
		export HF_TOKEN="$(tr -d ' \t\r\n' < "$HOME/.config/huggingface/token")"
	fi
fi

echo "Start Time: $(date)"
echo "Working Dir: $(pwd)"

# ---- Configurable per iteration ----
INPUT_FILE="iterative_back_translation/data/eng_dataset_3.txt"
OUTPUT_TSV="iterative_back_translation/data/iter1-v5_eng(set3)_to_pra_parallel.tsv"
MODEL_DIR="iterative_back_translation/models/eng_to_prakrit_v5_iter_2-final"
# -------------------------------------

# Base model: indictrans2-en-indic-1B (En→Pr direction)
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
	HF_MODEL_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ai4bharat--indictrans2-en-indic-1B"
	CACHED_SNAPSHOT=""
	if [[ -d "$HF_MODEL_CACHE_ROOT/snapshots" ]]; then
		CACHED_SNAPSHOT="$(find "$HF_MODEL_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
	fi

	if [[ -n "$CACHED_SNAPSHOT" && -f "$CACHED_SNAPSHOT/config.json" ]]; then
		BASE_MODEL_PATH="$CACHED_SNAPSHOT"
	else
		BASE_MODEL_PATH="$PWD/indictrans2_base_model_en_indic"
	fi
fi

if [[ -d "$BASE_MODEL_PATH" ]]; then
	export HF_HUB_OFFLINE="1"
	export TRANSFORMERS_OFFLINE="1"
	echo "Using local base model: $BASE_MODEL_PATH"
	python iterative_back_translation/scripts/run_inference_iterative.py \
		--input "$INPUT_FILE" \
		--output "$OUTPUT_TSV" \
		--model-dir "$MODEL_DIR" \
		--base-model "$BASE_MODEL_PATH" \
		--direction eng2pra \
		--batch-size 64 \
		--local-files-only "$@"
else
	echo "WARN: Local base model not found at $BASE_MODEL_PATH; attempting online download."
	python iterative_back_translation/scripts/run_inference_iterative.py \
		--input "$INPUT_FILE" \
		--output "$OUTPUT_TSV" \
		--model-dir "$MODEL_DIR" \
		--direction eng2pra \
		--batch-size 64 "$@"
fi

echo "End Time: $(date)"
