#!/usr/bin/env bash
# Portable shell setup for running the repository from a checkout.

_IT2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_IT2_ROOT}/src:${_IT2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "[interactive-training] project root: ${_IT2_ROOT}"

unset _IT2_ROOT
