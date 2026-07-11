#!/bin/bash

#SBATCH --job-name=muon_gpt
#SBATCH --account=aip-yuntian
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
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

# Source env.sh but drop its trailing interactive `apptainer shell` line.
# This exports the APPTAINERENV_* vars (API keys, certs, wandb, HF token) that
# Apptainer forwards into the container, and loads the apptainer module.
if [[ -f "$PROJ_DIR/env.sh" ]]; then
    set +u
    source <(grep -v '^apptainer ' "$PROJ_DIR/env.sh")
    set -u
fi

# init.sh is re-sourced inside the container to set PYTHONPATH and DC-conditional
# modules; the APPTAINERENV_* vars set above arrive as plain env vars inside.
CMD="cd $PROJ_DIR && set -e && source init.sh && python -m examples.muon_gpt \
    --max-rounds 10 \
    --seed $SEED \
    --wandb-experiment muon_gpt_seed$SEED \
    --output-dir logs/muon_gpt/seed$SEED/muon_gpt \
    --memory-path logs/muon_gpt/seed$SEED/muon_gpt_memory.jsonl \
    2>&1"

srun apptainer exec --nv \
    --bind /scratch/kstarxin:/scratch/kstarxin \
    --bind /project/6101847/kstarxin:/project/6101847/kstarxin \
    --bind /home/kstarxin:/home/kstarxin \
    "$SIF" bash -c "$CMD"

echo "[$(date)] Done (seed $SEED)."
