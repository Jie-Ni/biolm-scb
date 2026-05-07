#!/bin/bash
#SBATCH --job-name=pr_probe3
#SBATCH --partition=std
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/gpfs/gpfs1/scratch/c9881022/biolm_physics/logs/pr_probe3_%A_%a.out
#SBATCH --error=/gpfs/gpfs1/scratch/c9881022/biolm_physics/logs/pr_probe3_%A_%a.err

source /home/c988/c9881022/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/c9881022/vllm_env

cd /gpfs/gpfs1/scratch/c9881022/biolm_physics

# Round 3: Pythia 6.9B/12B + NucT validation (3 seeds) + XLM-R large + ProGen2 series
MODELS_AND_SEEDS=(
  "EleutherAI/pythia-6.9b text decoder 0"
  "EleutherAI/pythia-12b text decoder 0"
  "InstaDeepAI/nucleotide-transformer-500m-1000g DNA encoder 1"
  "InstaDeepAI/nucleotide-transformer-500m-1000g DNA encoder 2"
  "InstaDeepAI/nucleotide-transformer-2.5b-multi-species DNA encoder 0"
  "FacebookAI/xlm-roberta-large text encoder 0"
  "hugohrban/progen2-small protein decoder 0"
  "hugohrban/progen2-medium protein decoder 0"
  "hugohrban/progen2-base protein decoder 0"
  "hugohrban/progen2-large protein decoder 0"
)

IFS=' ' read -r MODEL TASK ARCH SEED <<< "${MODELS_AND_SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Probing ${MODEL} (${TASK}/${ARCH}) seed=${SEED}"
python code/_leo5_pr_probe.py \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --task-class "${TASK}" \
    --architecture "${ARCH}" \
    --out runs/pr_profile/
