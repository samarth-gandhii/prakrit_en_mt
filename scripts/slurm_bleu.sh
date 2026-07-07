#!/bin/bash
#SBATCH --job-name=prakrit_bleu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:1
#SBATCH --mem=32G
#SBATCH -t 03-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

source hpc_env/bin/activate

export HF_TOKEN=""

echo "Start Time: $(date)"

python run_bleu.py --hyp inferences/prakrit_to_english.txt --ref inferences/reference_english_100.txt

echo "End Time: $(date)"
