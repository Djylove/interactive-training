# Interactive Training 2

> Upgrading from v1? The original Hugging Face mixin and React dashboard are
> preserved at tag [`v1.0.0`](https://github.com/yuntian-group/interactive-training/tree/v1.0.0)
> and branch [`legacy/v1`](https://github.com/yuntian-group/interactive-training/tree/legacy/v1).
> See [`docs/MIGRATION_v1_to_v2.md`](docs/MIGRATION_v1_to_v2.md).

Interactive Training 2 provides a framework-independent control-plane core for
steering active machine-learning runs. Training code registers typed settings and
structured actions with a `TrainingSession`; humans, scripts, heuristics, and
automated agents then use the same action protocol at explicit control points. The
optional LLM agent plans an initial configuration, acts while a round is running, and
writes a reflection to an explicit journal that informs later fresh rounds.

This repository is a research prototype accompanying an EMNLP System Demonstrations
submission. It is not an optimizer and does not guarantee that LLM interventions are
safe or beneficial.

## What is included

- `src/interactive_training/core/`: sessions, knobs, actions, events, goals, checkpoints, memory, and
  deterministic-round helpers.
- `src/interactive_training/agents/`: an optional plan/act/reflect LLM agent.
- `src/interactive_training/transport/`: HTTP/WebSocket control, Aim logging, and a Python client.
- `src/interactive_training/integrations/`: Hugging Face `Trainer` wrapping and optimizer autopatching.
- `src/interactive_training/recipes/`: reusable control surfaces for optimizers, GANs, Gym, and RLVR.
- `examples/`: BERT finetuning, data mixing, layerwise GPT learning rates,
  Muon–AdamW training, GRPO Countdown, and auxiliary experiments.
- `tests/`: core, transport, agent, Aim-transport, and recipe tests.

The customized Aim `/live` interface shown in the paper is published at
[`yuntian-group/aim`](https://github.com/yuntian-group/aim/tree/interactive-training-v2).
Its exact branch and commit are recorded in [`demo/aim.lock.json`](demo/aim.lock.json).
Standard Aim metric logging and the HTTP control API work without the fork; the
paper's monitoring-and-control workspace requires the pinned version via `AIM_SRC`.

## Reviewer quick paths

1. **Zero install:** open <https://interactivetraining.ai/live> to inspect the
   recorded paper trace or submit bounded controls to the queued tiny-BERT CPU
   sandbox. The public service never accepts LLM API keys.
2. **Core smoke test:** clone tag `v2.0.2`, install the transport extras, and run
   `python tests/run_tests.py`. This path needs no GPU, provider key, or Aim fork.
3. **Full Aim workspace:** clone the exact companion revision recorded in
   `demo/aim.lock.json`, set `AIM_SRC`, and run the BERT frontend example.

The Muon video is supplementary and deliberately separate from the committed
11-round paper trace. See [`demo/README.md`](demo/README.md) for the provenance of
each public mode.

## Installation

Python 3.10 or newer is required.

Clone the immutable release rather than relying on the moving default branch:

```bash
git clone --branch v2.0.2 \
  https://github.com/yuntian-group/interactive-training.git
cd interactive-training
```

### Core library

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### HTTP, Aim, and LLM agent

```bash
python -m pip install -e ".[transport,aim,agents]"
```

### Hugging Face demonstrations

```bash
python -m pip install -e ".[transport,aim,agents,hf]"
```

### RLVR and vision demonstrations

The GRPO example additionally needs the `rlvr` extra and a CUDA environment supported
by vLLM. The STL-10 GAN uses the `vision` extra.

```bash
python -m pip install -e ".[rlvr]"
python -m pip install -e ".[vision]"
```

Provider credentials are read from `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
Never commit keys or cluster environment files. A key may also be supplied at runtime
through the `configure_agent` action; it is write-only and is redacted from state and
events.

## Quick CPU smoke test

```bash
python tests/run_tests.py
```

The test runner exercises the in-process control path, HTTP transport, scripted
agents, recipes, checkpoint bookkeeping, and memory without making an LLM request.
The manual Aim proxy smoke test is:

```bash
python tests/e2e_live_smoke.py
```

It requires Aim and, for the `/api/live` proxy path, the companion Aim fork.

## Minimal direct integration

```python
from interactive_training import LLMAgent, TrainingSession

session = TrainingSession(
    goal="val_loss",
    agent=LLMAgent(every=100),
    memory="memory.jsonl",
)

session.register_knob(
    "lr",
    get=get_learning_rate,
    set=set_learning_rate,
    min=0.0,
    max=1e-2,
    description="optimizer learning rate",
)

with session.run():
    for step, batch in enumerate(loader):
        loss = update_model(batch)
        control = session.step({"loss": float(loss)}, step=step)
        if control.stop:
            break
```

For Hugging Face Trainer:

```python
from interactive_training import make_interactive
from transformers import Trainer

InteractiveTrainer = make_interactive(Trainer)
trainer = InteractiveTrainer(..., session=session)
trainer.train()
```

For loops that cannot be edited directly, `session.autopatch(optimizer)` wraps
`optimizer.step()` as a control point.

## Action and event APIs

When the HTTP transport is active:

- `GET /state`: current status, goal, knobs, action schemas, agent configuration,
  round metadata, checkpoints, and model tree.
- `POST /actions`: submit an action object.
- `GET /events?since=<seq>`: recover retained events after a sequence number.
- `WS /events?since=<seq>`: subscribe to live events.

Example:

```bash
curl -X POST http://127.0.0.1:9876/actions \
  -H 'Content-Type: application/json' \
  -d '{"type":"set_knob","payload":{"name":"lr","value":0.00002},"source":"human:cli"}'
```

Built-in actions include `set_knob`, `evaluate`, `save_checkpoint`,
`load_checkpoint`, `pause`, `resume`, `stop`, `reset_module`, `note`, `set_agent`,
`configure_agent`, and `set_context`. Applications can register additional handlers.
The supplied LLM agent is denied destructive and self-reconfiguration actions.

## Multi-round semantics

`TrainingSession.run_rounds(...)` adds one no-LLM reference round whenever an agent is
attached. `--max-rounds N` therefore means one reference plus `N` agent-guided rounds.
The showcased examples initialize a fresh model each round and reapply the same seed;
the explicit session journal, not model weights, persists between rounds.

Each journal JSONL record contains the initial configuration, best score and step,
actions, reflection, and token accounting. `scripts/plot_memory_scores.py` plots one
journal file; `scripts/plot_all_frontiers.py` regenerates all five paper panels and
their SHA-256 provenance manifest.

## Demonstrations

### BERT/IMDB with Aim

```bash
python -m examples.hf_bert_imdb_frontend \
  --max-rounds 3 \
  --max-steps 300 \
  --agent-every 50 \
  --preflight
```

This is the recommended low-cost demo. `--preflight` pauses before the first round so
the user can inspect the session and configure the agent.

### Muon–AdamW GPT with Aim

```bash
python -m examples.muon_gpt_frontend --max-rounds 4
```

This is the paper screencast path and is intended for an H100-class GPU. It trains a
Qwen-style model from scratch on streamed FineWeb-Edu.

### Headless experiment examples

```bash
python -m examples.hf_bert_imdb_multiround --max-rounds 10
python -m examples.data_mixing_sentiment --max-rounds 10
python -m examples.layerwise_lr_gpt --max-rounds 10
python -m examples.muon_gpt --max-rounds 10
python -m examples.rlvr_grpo_countdown --max-rounds 7
```

These are expensive research runs, not quickstart tests. Review each script's CLI and
hardware requirements before launching it.

## Aim setup

`source init_aim.sh` creates a local Aim virtual environment. Clone the pinned
companion branch and set `AIM_SRC` to that editable checkout to obtain the custom
`/live` workspace:

```bash
git clone --branch interactive-training-v2 \
  https://github.com/yuntian-group/aim.git ../aim
export AIM_SRC=../aim
source init_aim.sh
```

Without `AIM_SRC`, the script installs stock Aim. Stock Aim stores and displays
metrics, but it does not include the paper's custom control panels.

## Reproducing reported figures

The five seed-42 session journals used by the paper are committed under `logs/`.
Regenerate the individual panels and provenance manifest with:

```bash
python -m pip install -e ".[plots]"
python scripts/plot_all_frontiers.py --output-dir figures
python scripts/export_memory_evidence.py --output-dir generated
```

The resulting `figures/frontier_manifest.json` records the source commit, input and
output hashes, plotting version, round counts, baseline/best scores, action counts, and
cumulative token/cost fields.

A complete re-execution artifact should additionally include:

- the exact git commit and command line;
- best-round JSON and per-step Aim/W&B exports;
- model and dataset revisions;
- seed, training/evaluation budgets, and intervention cadence;
- LLM provider/model/API date and prompt context;
- GPU type, runtime, and token usage.

The journal release is sufficient to reproduce every cross-round score, new-running-best
classification, summarized action, reflection, and cumulative token/cost value. It
does not contain per-step metric curves, checkpoints, Slurm output, wall-clock runtime,
or GPU-hour accounting.

## Safety and privacy

- Setting values are converted and clamped to registered bounds.
- Agent permissions exclude checkpoint loading, pausing, module reset, context
  changes, and self-configuration.
- API keys are not returned by `/state` and are redacted from recorded action payloads.
- Prompts may contain proprietary telemetry and are sent to the configured provider.
- The LLM call is synchronous at a control point; provider latency can pause progress.
- Custom action handlers remain responsible for semantic validation and rollback.

Use action limits, approval gates, resource budgets, and provider retention settings
appropriate to the workload.

## Citation

The v2 citation will be added after archival publication. For the original system:

```bibtex
@inproceedings{zhang-etal-2025-interactive,
  title = {Interactive Training: Feedback-Driven Neural Network Optimization},
  author = {Zhang, Wentao and Lu, Yang Young and Deng, Yuntian},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in
               Natural Language Processing: System Demonstrations},
  year = {2025},
  pages = {851--861},
  doi = {10.18653/v1/2025.emnlp-demos.65}
}
```

## License

MIT. See [LICENSE](LICENSE).
