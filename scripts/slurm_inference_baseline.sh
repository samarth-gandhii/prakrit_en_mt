#!/bin/bash
#SBATCH --job-name=prakrit_infer_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

python run_inference_baseline.py --sample-count 100 --output inferences/prakrit_to_english.baseline.txt "$@"
