# Interactive Training 2 Hybrid Public Demo

The public site deliberately separates three experiences:

1. **Video walkthrough** — the canonical five-round, 1,000-step Muon demonstration
   from the paper video. The original event log is unavailable; no events are
   synthesized.
2. **Muon paper-trace explorer** — an 11-round, 3,000-step round-level view generated
   from the committed memory ledger. It is explicitly distinct from the video.
3. **Live CPU micro-demo** — a real tiny-BERT sentiment run with one no-LLM baseline
   and one deterministic scripted round. Anonymous users can change safe knobs but
   cannot provide API keys or invoke destructive actions.

## Components

- Customized Aim branch `aim_interactive_training_v2`: cockpit, v2 proxy, demo cards.
- `gateway/`: FastAPI queue and one CPU trainer worker.
- `assets/`: video and paper-trace manifests.
- `deploy/`: nginx and systemd templates.

## Local gateway test

```bash
pip install -e ".[transport]"
pip install -e "demo/gateway[dev]"
python scripts/build_demo_assets.py
pytest demo/gateway/tests
uvicorn it2_demo.app:app --host 127.0.0.1 --port 39081
```

The live CPU worker additionally requires:

```bash
pip install -e ".[transport,aim,hf]"
pip install -e "demo/gateway[trainer]"
```

Pre-cache the public model and dataset before enabling anonymous jobs:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'); \
AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)"
python -c "from datasets import load_dataset; load_dataset('stanfordnlp/imdb')"
```

## Production invariants

- Trainer control ports bind to loopback only.
- One CPU trainer runs at a time.
- No provider credentials enter the gateway or trainer environment.
- Replay and paper-trace controls are read-only.
- Public live mode denies agent configuration, context changes, checkpoint loading,
  and module reset.
- Each live session has a signed cookie, finite action budget, and hard wall timeout.
- Deployment requires adequate disk/memory headroom; do not deploy to a 99%-full
  filesystem.

See `demo/deploy/` for service templates. Adapt paths in a staging environment before
cutting over nginx.
