#!/bin/bash
#SBATCH --job-name=prakrit_finetune
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:amphere:1
#SBATCH --mem=25G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

set +u
source /home3/qvlw18/miniconda3/etc/profile.d/conda.sh
conda activate prakrit_mt
set -u

echo "CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)"

python run_finetune.py --epoch 5 --model-dir models/prakrit_to_eng --train data/prakrit_eng.clean.tsv
