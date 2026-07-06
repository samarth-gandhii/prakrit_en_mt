#!/bin/bash
#SBATCH --job-name=prakrit_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:1
#SBATCH --mem=32G
#SBATCH -t 03-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# Load the exact SVNIT modules
module load anaconda3-2024.2
module load cuda-12.8

# Activate isolated environment
source hpc_env/bin/activate

# Handle Hugging Face paths and bypass interactive prompts
export HF_HUB_DISABLE_SYMLINKS_WARNING="true"
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$HF_HOME"
export WANDB_DISABLED="true"

# Inject Read Token to bypass the gated model check
export HF_TOKEN="your_huggingface_read_token_here"

echo "CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)"

# LoRA is only triggered if explicitly passed (e.g. sbatch scripts/slurm_finetune.sh --use-lora) via "$@"
python run_finetune.py --epoch 5 --model-dir models/prakrit_to_eng --train data/prakrit_eng.clean.tsv "$@"
