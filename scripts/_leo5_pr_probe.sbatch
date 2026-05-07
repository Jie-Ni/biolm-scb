#!/bin/bash
#SBATCH --job-name=pr_probe
#SBATCH --partition=std
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/gpfs/gpfs1/scratch/c9881022/biolm_physics/logs/pr_probe_%A_%a.out
#SBATCH --error=/gpfs/gpfs1/scratch/c9881022/biolm_physics/logs/pr_probe_%A_%a.err

source /home/c988/c9881022/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/c9881022/vllm_env

cd /gpfs/gpfs1/scratch/c9881022/biolm_physics

# 8 encoder/text MLM models for cycle-74 redo extension
MODELS=(
  "bert-base-uncased text encoder"
  "roberta-base text encoder"
  "distilbert-base-uncased text encoder"
  "albert-base-v2 text encoder"
  "google/electra-base-discriminator text encoder"
  "microsoft/deberta-v3-base text encoder"
  "facebook/esm2_t30_150M_UR50D protein encoder"
  "facebook/esm2_t12_35M_UR50D protein encoder"
)

IFS=' ' read -r MODEL TASK ARCH <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Probing ${MODEL} (${TASK}/${ARCH}) seed=0"
python code/_leo5_pr_probe.py \
    --model "${MODEL}" \
    --seed 0 \
    --task-class "${TASK}" \
    --architecture "${ARCH}" \
    --out runs/pr_profile/
