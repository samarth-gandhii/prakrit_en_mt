#!/bin/bash
#SBATCH --job-name=Shri_AVerImaTeC
#SBATCH --nodes=1
#SBATCH --nodelist=node2
#SBATCH --gres=shard:1
#SBATCH --output=HPC_logs/node2_output_%j.log
#SBATCH --error=HPC_logs/node2_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 03-00:00:00
#SBATCH --mem=32G

echo "Start Time: $(date)"

cd /home/shrikant/2025/My_projects/HPC-Test

## python

echo "End Time: $(date)"
