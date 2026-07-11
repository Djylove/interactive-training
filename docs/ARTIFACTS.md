# Reproducing the Paper Artifacts

Five round-level memory ledgers are committed under `logs/`:

- `logs/hf_bert_imdb_multiround/seed42/bert_imdb_memory.jsonl`
- `logs/data_mixing_sentiment/babysat_seed42/sentiment_memory.jsonl`
- `logs/layerwise_lr_gpt/seed42/layerwise_lr_gpt_memory.jsonl`
- `logs/muon_gpt/seed42/muon_gpt_memory.jsonl`
- `logs/rlvr_grpo_countdown/seed42/countdown_memory.jsonl`

Regenerate the five frontier panels and provenance manifest:

```bash
pip install -e ".[plots]"
python scripts/plot_all_frontiers.py --output-dir figures
```

Each manifest record includes:

- input and output SHA-256;
- source commit and Matplotlib version;
- direction, round count, baseline and best scores;
- strict post-baseline frontier improvements;
- summarized action count; and
- cumulative input/output tokens and configured cost.

The ledgers reproduce cross-round scores, initial configurations, plans, summarized
actions, reflections, and usage. They do not contain:

- complete per-step metric trajectories;
- Aim or W&B databases;
- checkpoints;
- Slurm stdout/stderr;
- wall-clock runtime or GPU-hour accounting; or
- multi-seed uncertainty.

The Muon paper trace is a distinct 3,000-step, 11-round run. The 2:29 public video is
a 1,000-step, five-round demonstration whose original event log was removed. The
public demo therefore presents the video and the paper-trace explorer as separately
labeled artifacts and never merges their scores or actions.
