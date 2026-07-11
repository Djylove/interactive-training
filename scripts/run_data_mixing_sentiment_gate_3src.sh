#!/bin/bash

# Validation gate for the 3-source (tweet/finance/product) combination-optimum design
# (all no-agent, 1 seed, 2000-step rounds, sequential):
#   0. data smoke check (per-source class distributions)
#   1-3. single-source probes
#   4-6. pair mixes (uniform within the pair)
#   7. uniform over all three
#   8. class-balanced sampling (the standard non-LLM fix)
# Gate: best pair/triple >= best single + ~3 macro-F1 points, and class-balanced
# stays near uniform.
#
# Run INSIDE the apptainer container on a GPU node, e.g.:
#   apptainer exec --nv --bind /home/kstarxin:/home/kstarxin <SIF> \
#       bash scripts/run_data_mixing_sentiment_gate_3src.sh

set -euo pipefail

PROJ_DIR="/home/kstarxin/projects/aip-yuntian/kstarxin/interactive_training_v2"
cd "$PROJ_DIR"
source init.sh

SEED=42
GATE_DIR="logs/data_mixing_sentiment/gate_3src"
echo "[$(date)] 3-source gate runs, seed $SEED"

echo "===== [0/8] data smoke check: per-source class distributions ====="
python - <<'PY'
from collections import Counter
from examples.data_mixing_sentiment import _normalize, SOURCES
for s in SOURCES:
    ds = _normalize(s, 2000, 42)
    print(f"{s:8s} n={len(ds):5d} classes={dict(sorted(Counter(ds['labels']).items()))}")
PY

run_arm() {  # run_arm <name> <extra args...>
    local name=$1; shift
    echo "===== [$(date)] arm: $name ====="
    python -m examples.data_mixing_sentiment \
        --max-rounds 1 \
        --seed "$SEED" \
        --wandb-experiment "data_mixing_sentiment_gate3src_$name" \
        --output-dir "$GATE_DIR/$name/sentiment" \
        --memory-path "$GATE_DIR/$name/sentiment_memory.jsonl" \
        "$@" 2>&1
}

run_arm probe_tweet         --no-agent --sources tweet
run_arm probe_finance       --no-agent --sources finance
run_arm probe_product       --no-agent --sources product
run_arm pair_product_tweet  --no-agent --sources product tweet
run_arm pair_product_finance --no-agent --sources product finance
run_arm pair_tweet_finance  --no-agent --sources tweet finance
run_arm uniform             --no-agent
run_arm class_balanced      --mode class-balanced

echo "===== gate summary (best eval_target_macro_f1 per arm) ====="
python - <<'PY'
import json, glob, os
for path in sorted(glob.glob("logs/data_mixing_sentiment/gate_3src/*/sentiment_memory.jsonl")):
    arm = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        rounds = [json.loads(l) for l in f if l.strip()]
    if rounds:
        best = max(r["score"] for r in rounds)
        print(f"{arm:22s} macro_f1={best:.4f} (best_step={rounds[0].get('best_step')})")
PY

echo "[$(date)] Done (3-source gate, seed $SEED)."
