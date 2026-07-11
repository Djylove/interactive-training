#!/bin/bash

# Babysat arm of the 3-class data-mixing experiment: 1 baseline round plus
# ROUNDS agent rounds (run_rounds counts --max-rounds on top of the baseline).
# Reference numbers from the seed-42 gate (no-agent, same budget):
#   uniform 0.5703 | class_balanced 0.5691 | best single source (movie) 0.6939

#SBATCH --job-name=data_mixing_sentiment_babysat
#SBATCH --account=aip-yuntian
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2/logs/%x-%j.out
#SBATCH --error=/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2/logs/%x-%j.err

set -euo pipefail

PROJ_DIR="/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2"
SIF="/home/kstarxin/projects/aip-yuntian/kstarxin/venv/pal_vllm_019_verl_516657f_new_fla.sif"
LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJ_DIR"

SEED=42
ROUNDS=10
echo "[$(date)] babysat, seed $SEED, $ROUNDS agent rounds"

if [[ -f "$PROJ_DIR/env.sh" ]]; then
    set +u
    source <(grep -v '^apptainer ' "$PROJ_DIR/env.sh")
    set -u
fi

CMD="cd $PROJ_DIR && set -e && source init.sh && python -m examples.data_mixing_sentiment \
    --mode babysat \
    --max-rounds $ROUNDS \
    --seed $SEED \
    --wandb-experiment data_mixing_sentiment_babysat_seed$SEED \
    --output-dir logs/data_mixing_sentiment/babysat_seed$SEED/sentiment \
    --memory-path logs/data_mixing_sentiment/babysat_seed$SEED/sentiment_memory.jsonl \
    2>&1"

srun apptainer exec --nv \
    --bind /scratch/kstarxin:/scratch/kstarxin \
    --bind /project/6101847/kstarxin:/project/6101847/kstarxin \
    --bind /home/kstarxin:/home/kstarxin \
    "$SIF" bash -c "$CMD"

echo "[$(date)] Done (babysat, seed $SEED)."
