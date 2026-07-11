# Changelog

All notable changes follow semantic versioning.

## [2.0.0] - Unreleased

### Added

- Framework-agnostic `TrainingSession` with registered knobs, actions, goals, and
  event streams.
- Explicit apply-before-advance control points for Hugging Face Trainer, patched
  optimizers, and custom loops.
- Plan/act/reflect LLM reference operator with bounded permissions and persistent
  round memory.
- HTTP/WebSocket transport, Aim transport, and customized Aim Live workspace.
- Multi-round baseline semantics, deterministic-round helpers, DDP control broadcast,
  checkpoint bookkeeping, and action-result auditing.
- BERT, data-mixing, layerwise GPT, Muon–AdamW, RLVR, and GAN examples.
- Five committed seed-42 memory ledgers and reproducible frontier plotting tools.
- MIT license, security policy, tests, packaging extras, and hybrid public demo.

### Breaking

- Python 3.10+ and Pydantic 2 are required.
- The implementation now lives under the `interactive_training` package namespace.
- The control API is `GET /state`, `POST /actions`, and `GET`/`WS /events`.
- The primary rich UI is the customized Aim Live workspace.
- Dataset reload and arbitrary model-layer mutation APIs from v1 are not part of the
  v2 core; use the `v1.0.0` tag or migrate these behaviors to registered controls.

### Preserved

- `from interactive_training import make_interactive` remains available.
- v1 is immutable at tag `v1.0.0` and maintained on branch `legacy/v1`.

## [1.0.0] - 2026-07-10

- Frozen release of the original Hugging Face mixin, FastAPI server, dataset mixin,
  and bundled React dashboard.
