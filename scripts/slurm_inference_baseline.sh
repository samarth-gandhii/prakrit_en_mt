#!/bin/bash
#SBATCH --job-name=prakrit_infer_base
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:amphere:1
#SBATCH --mem=25G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

source ~/.bashrc
conda activate prakrit_mt

python run_inference_baseline.py --sample-count 100 --output inferences/prakrit_to_english.baseline.txt
