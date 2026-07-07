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

source hpc_env/bin/activate

# export HF_HOME="/home/shrikant/hf_cache"
# export TRANSFORMERS_CACHE="/home/shrikant/hf_cache"
# mkdir -p $HF_HOME

# cd "/home/shrikant/2026/Summer Internship/Samarth/prakrit_en_mt"

export HF_TOKEN=""

echo "Start Time: $(date)"

python run_finetune.py --epoch 5 --model-dir models/prakrit_to_eng --train data/prakrit_eng.clean.tsv "$@"

echo "End Time: $(date)"