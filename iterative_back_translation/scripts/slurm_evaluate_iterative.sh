#!/bin/bash
#SBATCH --job-name=ibt_evaluate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:40
#SBATCH --mem=32G
#SBATCH -t 03-00:00:00
#SBATCH --output=iterative_back_translation/logs/%x_%j.out
#SBATCH --error=iterative_back_translation/logs/%x_%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${REPO_ROOT}/iterative_back_translation/logs"
mkdir -p "${REPO_ROOT}/iterative_back_translation/data"
cd "${REPO_ROOT}"

source hpc_env/bin/activate

# Force offline mode
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

echo "Start Time: $(date)"

# ---------- Model directories (newly finetuned iterative models) ----------
PR_EN_MODEL_DIR="${REPO_ROOT}/iterative_back_translation/models/prakrit_to_eng_iterative-final"
EN_PR_MODEL_DIR="${REPO_ROOT}/iterative_back_translation/models/eng_to_prakrit_iterative-final"

# ---------- Test data paths ----------
# Prakrit→English
PR_EN_INPUT="/home/shrikant/2026/Summer Internship/Samarth/prakrit_en_mt/inferences/input_prakrit_100.txt"
PR_EN_REF="/home/shrikant/2026/Summer Internship/Samarth/prakrit_en_mt/inferences/reference_english_100.txt"
PR_EN_HYPO="${REPO_ROOT}/iterative_back_translation/data/eval_prakrit_to_eng_iterative.txt"

# English→Prakrit
EN_PR_INPUT="/home/shrikant/2026/Summer Internship/Samarth/Prakrit_MT/inferences/input_english_100.txt"
EN_PR_REF="/home/shrikant/2026/Summer Internship/Samarth/Prakrit_MT/inferences/reference_prakrit_100.txt"
EN_PR_HYPO="${REPO_ROOT}/iterative_back_translation/data/eval_eng_to_prakrit_iterative.txt"

# ---------- Base model tokenizer paths (from HF cache) ----------
# pr→en tokenizer: indictrans2-indic-en-1B
HF_SNAPSHOTS_PREN="/home/shrikant/.cache/huggingface/hub/models--ai4bharat--indictrans2-indic-en-1B/snapshots"
BASE_MODEL_PREN="$(find "$HF_SNAPSHOTS_PREN" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
if [[ -z "$BASE_MODEL_PREN" && -d "$HF_SNAPSHOTS_PREN" ]]; then
	BASE_MODEL_PREN="$HF_SNAPSHOTS_PREN"
fi

# en→pr tokenizer: indictrans2-en-indic-1B
HF_SNAPSHOTS_ENPR="/home/shrikant/.cache/huggingface/hub/models--ai4bharat--indictrans2-en-indic-1B/snapshots"
BASE_MODEL_ENPR="$(find "$HF_SNAPSHOTS_ENPR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
if [[ -z "$BASE_MODEL_ENPR" && -d "$HF_SNAPSHOTS_ENPR" ]]; then
	BASE_MODEL_ENPR="$HF_SNAPSHOTS_ENPR"
fi

INFERENCE_SCRIPT="${REPO_ROOT}/finetuning_and_inferencing_using_indictrans2_models/inferencing_using_saved_model.py"

# ==========================================================
# 1. Evaluate Prakrit → English iterative model
# ==========================================================
echo ""
echo "=========================================="
echo "Evaluating Prakrit→English iterative model"
echo "=========================================="
echo "  Model:     $PR_EN_MODEL_DIR"
echo "  Input:     $PR_EN_INPUT"
echo "  Reference: $PR_EN_REF"
echo "  Output:    $PR_EN_HYPO"
echo "  Tokenizer: $BASE_MODEL_PREN"

if [[ ! -d "$PR_EN_MODEL_DIR" ]]; then
	echo "SKIP: Model not found at $PR_EN_MODEL_DIR — train it first."
elif [[ ! -f "$PR_EN_INPUT" ]]; then
	echo "SKIP: Input file not found at $PR_EN_INPUT."
elif [[ ! -d "$BASE_MODEL_PREN" ]]; then
	echo "ERROR: Tokenizer for pr→en not found. Cache indictrans2-indic-en-1B on the login node first."
	exit 1
else
	python "$INFERENCE_SCRIPT" \
		--input  "$PR_EN_INPUT" \
		--model  "$PR_EN_MODEL_DIR" \
		--output "$PR_EN_HYPO" \
		--base-model "$BASE_MODEL_PREN" \
		--src-lang hin_Deva \
		--tgt-lang eng_Latn \
		--batch-size 4 \
		--local-files-only

	echo ""
	echo "BLEU Score (Prakrit→English iterative model):"
	python run_bleu.py --hyp "$PR_EN_HYPO" --ref "$PR_EN_REF"
fi

# ==========================================================
# 2. Evaluate English → Prakrit iterative model
# ==========================================================
echo ""
echo "=========================================="
echo "Evaluating English→Prakrit iterative model"
echo "=========================================="
echo "  Model:     $EN_PR_MODEL_DIR"
echo "  Input:     $EN_PR_INPUT"
echo "  Reference: $EN_PR_REF"
echo "  Output:    $EN_PR_HYPO"
echo "  Tokenizer: $BASE_MODEL_ENPR"

if [[ ! -d "$EN_PR_MODEL_DIR" ]]; then
	echo "SKIP: Model not found at $EN_PR_MODEL_DIR — train it first."
elif [[ ! -f "$EN_PR_INPUT" ]]; then
	echo "SKIP: Input file not found at $EN_PR_INPUT."
elif [[ ! -d "$BASE_MODEL_ENPR" ]]; then
	echo "ERROR: Tokenizer for en→pr not found. Cache indictrans2-en-indic-1B on the login node first."
	exit 1
else
	python "$INFERENCE_SCRIPT" \
		--input  "$EN_PR_INPUT" \
		--model  "$EN_PR_MODEL_DIR" \
		--output "$EN_PR_HYPO" \
		--base-model "$BASE_MODEL_ENPR" \
		--src-lang eng_Latn \
		--tgt-lang hin_Deva \
		--batch-size 4 \
		--local-files-only

	echo ""
	echo "BLEU Score (English→Prakrit iterative model):"
	python run_bleu.py --hyp "$EN_PR_HYPO" --ref "$EN_PR_REF"
fi

echo ""
echo "End Time: $(date)"
