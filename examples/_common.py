"""Shared helpers for the example experiments (deduped from each example's main)."""
from __future__ import annotations

import logging
import os

from interactive_training.agents.agent import OpenAIClient

# This project's top-level packages; everything else (HF, aim, urllib3, ...) stays quiet.
_PROJECT_LOGGERS = ("examples", "core", "agents", "transport", "integrations",
                    "recipes", "__main__")


class _TqdmSafeHandler(logging.StreamHandler):
    """Emit through `tqdm.write` so log lines don't shred an active progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=self.stream)
        except ImportError:
            super().emit(record)
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    """One-call console logging for the examples (replaces per-file basicConfig).

    Third-party libraries log at WARNING so HF Trainer's own metric lines and its
    tqdm progress bar stay readable, while this project's loggers emit at `level`
    (INFO by default) so per-step metric lines actually reach the terminal.
    Idempotent: safe to call from an example that imports another example.
    """
    root = logging.getLogger()
    if not any(isinstance(h, _TqdmSafeHandler) for h in root.handlers):
        handler = _TqdmSafeHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(logging.WARNING)
    for name in _PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(level)


def build_client(args) -> OpenAIClient:
    """Build the babysitter LLM client for the chosen provider.

    - openai: native OpenAI request to api.openai.com using OPENAI_API_KEY, with
      `reasoning_effort` passed as a first-class parameter (api="responses", required for
      gpt-5.x when combining tools with reasoning effort).
    - openrouter: OpenRouter endpoint using OPENROUTER_API_KEY, with reasoning effort
      forwarded via `extra_body` (OpenRouter's schema).

    The model slug is read from `--agent-model` if present, else `--model`.

    The API key is optional here: a missing key yields a client with
    ``api_key_set == False`` so the run can launch agent-configurable from the frontend
    (the human supplies provider/model/key via a `configure_agent` action, §3.8). The
    `provider` is stored on the client so it surfaces in GET /state.agent.
    """
    model = getattr(args, "agent_model", None) or args.model
    if args.provider == "openai":
        return OpenAIClient(model=model, base_url=args.base_url,
                            api_key=os.environ.get("OPENAI_API_KEY"),
                            reasoning_effort=args.reasoning_effort, api="responses",
                            provider="openai")
    return OpenAIClient(model=model,
                        base_url=args.base_url or "https://openrouter.ai/api/v1",
                        api_key=os.environ.get("OPENROUTER_API_KEY"),
                        extra_body={"reasoning": {"effort": args.reasoning_effort}},
                        provider="openrouter")
