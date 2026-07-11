"""Best-effort RNG seeding so each round can reproduce the same baseline trajectory."""
from __future__ import annotations

DEFAULT_SEED = 42


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy, and torch RNGs to the same value.

    Imports are lazy so the core keeps no hard dependency on numpy/torch (plan §3.5).
    Re-seeding to the *same* value at the start of every round means a fresh round
    starts from an identical RNG state (weight init, data shuffling, sampling), so any
    trajectory difference between rounds is attributable to the agent's actions rather
    than sampling noise. This is best-effort: GPU kernels can still be non-deterministic.
    """
    import os
    import random
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Seeding the RNGs alone is not enough on GPU: cuBLAS autotuning and many
        # atomicAdd-based backward kernels are non-deterministic across executions,
        # so identical seeds + identical weights still diverge after the first step.
        # Pin the deterministic code paths too (best-effort; warn_only keeps ops that
        # have no deterministic implementation from raising).
        # NB: CUBLAS_WORKSPACE_CONFIG only takes effect if it is set *before* CUDA
        # initializes cuBLAS, so launchers should also export it at process start.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 matmuls vary by hardware path; disable so a run is reproducible.
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        # Flash / memory-efficient attention have non-deterministic backward kernels;
        # force the math SDPA path (transformers uses sdpa attention by default).
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass
