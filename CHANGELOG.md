# Changelog

All notable changes follow semantic versioning.

## [2.0.2] - 2026-07-11

### Changed

- Compacted the generated cross-round journal table for publication-scale reading.
- Made PyPI publication opt-in until trusted publishing is configured.

### Fixed

- Added the compiler toolchain required to build the pinned Aim fork in the demo
  container.

## [2.0.1] - 2026-07-11

### Changed

- Documented zero-install, core-smoke, and full-Aim reviewer paths.
- Linked the canonical `interactive-training-v2` Aim branch and machine-readable
  commit pin.
- Replaced unavailable PyPI installation instructions with an immutable Git tag.
- Reworded generated evidence tables around event journals and new running bests.

### Fixed

- Removed stale claims that the Aim fork was unpublished.
- Corrected the companion Aim branch name in demo documentation.

## [2.0.0] - 2026-07-11

### Added

- Framework-independent `TrainingSession` core with registered settings, actions,
  goals, and event streams.
- Explicit apply-before-advance control points for Hugging Face Trainer, patched
  optimizers, and custom loops.
- Plan/act/reflect LLM agent with bounded permissions and an explicit cross-round
  journal.
- HTTP/WebSocket transport, Aim transport, and customized Aim Live workspace.
- Multi-round reference semantics, deterministic-round helpers, DDP control broadcast,
  checkpoint bookkeeping, and action-result auditing.
- BERT, data-mixing, layerwise GPT, Muon–AdamW, RLVR, and GAN examples.
- Five committed seed-42 journals and reproducible control-trace plotting tools.
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
