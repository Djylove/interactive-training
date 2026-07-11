#!/bin/bash

#SBATCH --job-name=data_mixing_sentiment
#SBATCH --account=aip-yuntian
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus=l40s:1
#SBATCH --output=/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2/logs/%x-%j.out
#SBATCH --error=/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2/logs/%x-%j.err

set -euo pipefail

PROJ_DIR="/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2"
SIF="/home/kstarxin/projects/aip-yuntian/kstarxin/venv/pal_vllm_019_verl_516657f_new_fla.sif"
LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJ_DIR"

SEED=42
echo "[$(date)] seed $SEED"

if [[ -f "$PROJ_DIR/env.sh" ]]; then
    set +u
    source <(grep -v '^apptainer ' "$PROJ_DIR/env.sh")
    set -u
fi

NOISE_SOURCES="product finance"
NOISE_FRAC="0.4 0.4"
CMD="cd $PROJ_DIR && set -e && source init.sh && python -m examples.data_mixing_sentiment \
    --mode babysat \
    --max-rounds 10 \
    --seed $SEED \
    --noise-source $NOISE_SOURCES \
    --noise-frac $NOISE_FRAC \
    --wandb-experiment data_mixing_sentiment_seed$SEED \
    --output-dir logs/data_mixing_sentiment/seed$SEED/sentiment \
    --memory-path logs/data_mixing_sentiment/seed$SEED/sentiment_memory.jsonl \
    2>&1"

srun apptainer exec --nv \
    --bind /scratch/kstarxin:/scratch/kstarxin \
    --bind /project/6101847/kstarxin:/project/6101847/kstarxin \
    --bind /home/kstarxin:/home/kstarxin \
    "$SIF" bash -c "$CMD"

echo "[$(date)] Done (seed $SEED)."
