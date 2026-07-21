#!/bin/bash
#SBATCH --job-name=ibt_evaluate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:16
#SBATCH --mem=32G
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

# ---------- Evaluate the iterative Prakrit→English model ----------
echo "=========================================="
echo "Evaluating Prakrit→English iterative model"
echo "=========================================="

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

MODEL_DIR="iterative_back_translation/models/prakrit_to_eng_iterative-final"
EVAL_OUTPUT="iterative_back_translation/data/eval_prakrit_to_eng_iterative.txt"

if [[ -d "$MODEL_DIR" ]]; then
	if [[ -d "$BASE_MODEL_PATH" ]]; then
		export HF_HUB_OFFLINE="1"
		export TRANSFORMERS_OFFLINE="1"
		echo "Using local base model: $BASE_MODEL_PATH"
		python run_inference.py \
			--sample-count 100 \
			--model-dir "$MODEL_DIR" \
			--output "$EVAL_OUTPUT" \
			--base-model "$BASE_MODEL_PATH" \
			--local-files-only "$@"
	else
		python run_inference.py \
			--sample-count 100 \
			--model-dir "$MODEL_DIR" \
			--output "$EVAL_OUTPUT" "$@"
	fi

	echo ""
	echo "BLEU Score (Prakrit→English iterative model):"
	python run_bleu.py --hyp "$EVAL_OUTPUT" --ref inferences/reference_english_100.txt
else
	echo "SKIP: Model not found at $MODEL_DIR. Train it first."
fi

echo ""
echo "End Time: $(date)"
