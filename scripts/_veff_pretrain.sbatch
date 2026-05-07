#!/bin/bash
#SBATCH --account=p201130
#SBATCH --job-name=veff_pre
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/fs201130/jn20658/biolm_veff/logs/veff_%A_%a.out
#SBATCH --error=/scratch/fs201130/jn20658/biolm_veff/logs/veff_%A_%a.err

source /home/jn20658/miniconda3/etc/profile.d/conda.sh
conda activate turbognn

# Local cache lives on the compute node (per PI directive: avoid /scratch /data)
mkdir -p /local/hf_cache /local/tmp
export HF_HOME=/local/hf_cache
export TMPDIR=/local/tmp
export HF_DATASETS_CACHE=/local/hf_cache/datasets
export TRANSFORMERS_CACHE=/local/hf_cache/transformers
export TOKENIZERS_PARALLELISM=false

OUT_BASE=/scratch/fs201130/jn20658/biolm_veff/runs
mkdir -p $OUT_BASE

# Six V_eff configurations
CONFIGS=(v_256 v_1k v_8k v_50k v_100k v_200k)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Starting V_eff pretrain config=$CONFIG on $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv

# datasets package not in turbognn, install if needed
python -c "import datasets" 2>/dev/null || pip install --user datasets

cd /scratch/fs201130/jn20658/biolm_veff

python code/_veff_pretrain.py \
    --config "$CONFIG" \
    --out "$OUT_BASE" \
    --tokens 5000000000 \
    --seq-len 1024 \
    --batch-size 32 \
    --grad-accum 2 \
    --lr 6e-4 \
    --probe-interval 2000 \
    --ckpt-interval 10000 \
    --seed 42 \
    --local-cache /local/hf_cache \
    --hf-dataset HuggingFaceFW/fineweb-edu \
    --hf-subset sample-10BT

echo "[$(date)] $CONFIG DONE"
