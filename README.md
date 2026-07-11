# Interactive Training 2

> Upgrading from v1? The original Hugging Face mixin and React dashboard are
> preserved at tag [`v1.0.0`](https://github.com/yuntian-group/interactive-training/tree/v1.0.0)
> and branch [`legacy/v1`](https://github.com/yuntian-group/interactive-training/tree/legacy/v1).
> See [`docs/MIGRATION_v1_to_v2.md`](docs/MIGRATION_v1_to_v2.md).

Interactive Training 2 is a framework-agnostic control plane for steering active
machine-learning training runs. Training code registers live knobs and structured
actions with a `TrainingSession`; humans, scripts, heuristics, and automated operators
then use the same action protocol at explicit control points. The supplied LLM client
is one reference operator: it plans an initial configuration, acts while a round is
running, and writes a reflection that informs later fresh rounds.

This repository is a research prototype accompanying an EMNLP System Demonstrations
submission. It is not an optimizer and does not guarantee that LLM interventions are
safe or beneficial.

## What is included

- `src/interactive_training/core/`: sessions, knobs, actions, events, goals, checkpoints, memory, and
  deterministic-round helpers.
- `src/interactive_training/agents/`: a plan/act/reflect LLM reference operator.
- `src/interactive_training/transport/`: HTTP/WebSocket control, Aim logging, and a Python client.
- `src/interactive_training/integrations/`: Hugging Face `Trainer` wrapping and optimizer autopatching.
- `src/interactive_training/recipes/`: reusable control surfaces for optimizers, GANs, Gym, and RLVR.
- `examples/`: BERT finetuning, data mixing, layerwise GPT learning rates,
  Muon–AdamW training, GRPO Countdown, and auxiliary experiments.
- `tests/`: core, transport, agent, Aim-transport, and recipe tests.

The customized Aim `/live` interface shown in the paper currently lives in a
companion Aim fork. Standard Aim metric logging and the HTTP control API work from
this repository; reproducing the exact paper UI additionally requires that fork via
`AIM_SRC`. The fork must be published or merged before the paper artifact is
considered complete.

## Installation

Python 3.10 or newer is required.

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
- `GET /events?since=<seq>`: replay retained events.
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

`TrainingSession.run_rounds(...)` adds one no-agent baseline round whenever an agent is
attached. `--max-rounds N` therefore means one baseline plus `N` agent rounds.
The showcased examples initialize a fresh model each round and reapply the same seed;
session memory, not model weights, persists between rounds.

Each memory JSONL record contains the initial configuration, best score and step,
actions, reflection, and token accounting. `scripts/plot_memory_scores.py` plots one
memory file; `scripts/plot_all_frontiers.py` regenerates all five paper panels and
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
the operator can inspect the session and configure the agent.

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

`source init_aim.sh` creates a local Aim virtual environment. Set `AIM_SRC` to an
editable checkout of the companion fork to obtain the custom `/live` workspace:

```bash
export AIM_SRC=/path/to/aim-fork
source init_aim.sh
```

Without `AIM_SRC`, the script installs stock Aim. Stock Aim stores and displays
metrics, but it does not include the paper's custom control panels.

## Reproducing reported figures

The five seed-42 session-memory ledgers used by the paper are committed under `logs/`.
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
- operator provider/model/API date and prompt context;
- GPU type, runtime, and token usage.

The memory release is sufficient to reproduce every cross-round score, strict frontier
classification, summarized action, reflection, and cumulative token/cost value. It
does not contain per-step metric curves, checkpoints, Slurm output, wall-clock runtime,
or GPU-hour accounting.

## Safety and privacy

- Knob values are converted and clamped to registered bounds.
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
