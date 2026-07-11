# Interactive Training 2 Hybrid Public Demo

The public site exposes two primary reviewer paths:

1. **Recorded paper-trace inspector** — an 11-round Muon view with 3,000 steps per
   round,
   generated from the committed seed-42 journal. It is explicitly distinct
   from the video.
2. **Live CPU micro-demo** — a real tiny-BERT sentiment run with one no-LLM reference
   and one deterministic scripted round. Anonymous users can change safe settings but
   cannot provide API keys or invoke destructive actions.

The supplementary **video walkthrough** is a separate five-round Muon demonstration
with 1,000 steps per round. Its original event log is unavailable, and the site does
not synthesize one.

## Components

- Customized Aim branch `interactive-training-v2`: monitoring/control workspace,
  v2 proxy, and demo cards. The exact revision is in `aim.lock.json`.
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

To reproduce the paper's complete `/live` workspace, use the exact companion commit
from `aim.lock.json`:

```bash
git clone --branch interactive-training-v2 \
  https://github.com/yuntian-group/aim.git ../aim
git -C ../aim checkout 11dc2691fa9886ceb932262ef3d3c85b810ff6c5
export AIM_SRC=../aim
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
- Recorded paper-trace controls are read-only.
- Public live mode denies agent configuration, context changes, checkpoint loading,
  and module reset.
- Each live session has a signed cookie, finite action budget, and hard wall timeout.
- Deployment requires adequate disk/memory headroom; do not deploy to a 99%-full
  filesystem.

See `demo/deploy/` for service templates. Adapt paths in a staging environment before
cutting over nginx.
