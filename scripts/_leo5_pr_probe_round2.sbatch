#!/bin/bash
#SBATCH --job-name=pr_probe2
#SBATCH --partition=std
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/gpfs/gpfs1/scratch/c9881022/biolm_physics/logs/pr_probe2_%A_%a.out
#SBATCH --error=/gpfs/gpfs1/scratch/c9881022/biolm_physics/logs/pr_probe2_%A_%a.err

source /home/c988/c9881022/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/c9881022/vllm_env

cd /gpfs/gpfs1/scratch/c9881022/biolm_physics

# Round 2: extend coverage
# Causal LMs not in fig123_alllayer + biology-specific encoders + nucleotide
MODELS=(
  "EleutherAI/pythia-70m text decoder"
  "EleutherAI/pythia-160m text decoder"
  "EleutherAI/pythia-410m text decoder"
  "EleutherAI/pythia-1.4b text decoder"
  "EleutherAI/pythia-2.8b text decoder"
  "facebook/esm2_t6_8M_UR50D protein encoder"
  "dmis-lab/biobert-base-cased-v1.2 text encoder"
  "allenai/scibert_scivocab_uncased text encoder"
  "InstaDeepAI/nucleotide-transformer-500m-1000g DNA encoder"
  "xlm-roberta-base text encoder"
)

IFS=' ' read -r MODEL TASK ARCH <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Probing ${MODEL} (${TASK}/${ARCH}) seed=0"
python code/_leo5_pr_probe.py \
    --model "${MODEL}" \
    --seed 0 \
    --task-class "${TASK}" \
    --architecture "${ARCH}" \
    --out runs/pr_profile/
