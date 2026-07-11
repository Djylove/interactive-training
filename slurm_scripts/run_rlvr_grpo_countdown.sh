#!/bin/bash
#SBATCH --job-name=rlvr_grpo_countdown
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
ROUNDS=7  # published memory trace: one baseline + seven LLM-operated rounds
echo "[$(date)] seed $SEED, $ROUNDS agent rounds"

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
CMD="cd $PROJ_DIR && set -e && source init.sh && python -m examples.rlvr_grpo_countdown \
    --max-rounds $ROUNDS \
    --seed $SEED \
    --wandb-experiment rlvr_grpo_countdown_seed$SEED \
    --output-dir logs/rlvr_grpo_countdown/seed$SEED/countdown \
    --memory-path logs/rlvr_grpo_countdown/seed$SEED/countdown_memory.jsonl \
    2>&1"

srun apptainer exec --nv \
    --bind /scratch/kstarxin:/scratch/kstarxin \
    --bind /project/6101847/kstarxin:/project/6101847/kstarxin \
    --bind /home/kstarxin:/home/kstarxin \
    "$SIF" bash -c "$CMD"

echo "[$(date)] Done (seed $SEED)."
