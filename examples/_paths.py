"""Shared output conventions for the example experiments.

Each example's logs (memory JSONL, checkpoints, and wandb runs) live under
`logs/<example>/`, and every run gets its own timestamped subfolder, so
different runs of the same example never overwrite or mix with each other.
"""
from __future__ import annotations

import os
import time

LOGS_DIR = "logs"


def logs_path(*parts: str) -> str:
    """A path under the logs dir, e.g. logs_path('gan_cifar10_memory.jsonl')."""
    return os.path.join(LOGS_DIR, *parts)


def setup_logs(example: str, run_id: str | None = None) -> tuple[str, str]:
    """Create logs/<example>/<run_id>/ and route wandb's run files into it (unless the
    caller already set WANDB_DIR). Call once at the start of an example's main(), and
    use the returned run_dir for that run's checkpoints/memory file.

    Returns (run_dir, run_id).
    """
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOGS_DIR, example, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", run_dir)
    return run_dir, run_id
