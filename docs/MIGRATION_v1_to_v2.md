# Migrating from Interactive Training v1 to v2

Interactive Training 2 is a major architectural release. v1 remains available at
[`v1.0.0`](https://github.com/yuntian-group/interactive-training/tree/v1.0.0) and
[`legacy/v1`](https://github.com/yuntian-group/interactive-training/tree/legacy/v1).

## Architecture

v1 embedded a FastAPI server and fixed command families into a Hugging Face Trainer
mixin. v2 introduces a framework-independent `TrainingSession` core: applications
register their own knobs and actions, then call a control point from Hugging Face
callbacks, an optimizer patch, or a custom loop.

## Installation

```bash
git clone --branch v2.0.2 \
  https://github.com/yuntian-group/interactive-training.git
cd interactive-training
python -m pip install -e .
```

Python 3.10+ is required. Install optional integrations explicitly:

```bash
python -m pip install -e ".[transport,agents,hf,aim]"
```

## Hugging Face Trainer

### v1

```python
from interactive_training import make_interactive

InteractiveTrainer = make_interactive(Trainer)
trainer = InteractiveTrainer(...)
trainer.train()
```

### v2

```python
from interactive_training import TrainingSession, make_interactive

session = TrainingSession(goal="eval_loss", frontend=True)
InteractiveTrainer = make_interactive(Trainer)
trainer = InteractiveTrainer(..., session=session)
trainer.train()
```

The v1 import remains valid, but the explicit session is recommended because it owns
goals, transports, the event journal, agent configuration, and multi-round state.

## HTTP protocol

| v1 | v2 |
| --- | --- |
| `GET /api/get_info/` | `GET /state` |
| `GET /api/get_logs/` | `GET /events?since=<seq>` |
| `POST /api/command/` | `POST /actions` |
| `WS /ws/message/` | `WS /events?since=<seq>` |
| nested JSON command args | structured `{type, payload, source}` |

Example v2 action:

```json
{
  "type": "set_knob",
  "payload": {"name": "lr", "value": 0.00002},
  "source": "human:client"
}
```

## Data and model controls

v2 replaces fixed command classes with application-owned registration:

```python
session.register_knob(
    "mixture_weight",
    get=get_weight,
    set=set_weight,
    min=0.0,
    max=1.0,
)

@session.action("switch_dataset", "Switch the active dataset", ["name"])
def switch_dataset(payload, session):
    ...
```

The v1 `make_interactive_dataset`, arbitrary model-layer operations, and
`model_layer_parameter_update` are not silently emulated. Keep production users on
`v1.0.0` until those behaviors are represented as explicit, validated v2 knobs or
actions.

## Frontend

v1 bundled a same-origin React dashboard on port 7007. v2 uses:

- a thin localhost control API;
- an Aim transport for persistent run history; and
- the customized Aim Live workspace for human control and auditing.

The public demo at <https://interactivetraining.ai> includes a recorded Muon
walkthrough and a constrained CPU micro-training sandbox. It never accepts public LLM
API keys.

## Multi-round semantics

When an LLM agent is attached, `run_rounds` adds one no-LLM reference round followed
by the configured number of fresh LLM-guided rounds. Model weights reset; the
explicit session journal persists. Each JSONL record contains the plan, round score,
best step, summarized actions, reflection, and cumulative usage.

## Rollback

```bash
pip install "git+https://github.com/yuntian-group/interactive-training@v1.0.0"
```

Do not depend on `main` for v1 behavior.
