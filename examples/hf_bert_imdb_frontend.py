"""Multiround BERT/IMDB finetuning driven from the Aim frontend.
"""
from __future__ import annotations

import argparse
import logging
import os


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from interactive_training.agents.agent import LLMAgent
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from examples._common import build_client, setup_logging
from examples._paths import setup_logs
from interactive_training.integrations.hf_trainer import make_interactive

setup_logging()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frontend-controlled multiround BERT/IMDB finetuning")
    p.add_argument("--model-ckpt", default="bert-base-uncased")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=16000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=1000, help="train steps per round")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    p.add_argument("--aim-repo", default=None,
                   help="Aim repo path (defaults to logs/hf_bert_imdb_frontend/<run-id>/aim)")
    p.add_argument("--no-ui", action="store_true",
                   help="do not serve the Aim web UI automatically (run `aim up` yourself)")
    p.add_argument("--ui-port", type=int, default=43800, help="port for the Aim web UI")
    p.add_argument("--preflight", action="store_true",
                   help="start paused at step 0: configure knobs/context "
                        "in the UI, then send `resume` (Start) to begin training")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--model", default="gpt-5.5", help="babysitter model slug")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--agent-every", type=int, default=100, help="agent acts every N training steps")
    return p.parse_args()


_HF_SCHEDULERS = {"linear", "cosine", "cosine_with_restarts", "polynomial",
                  "constant", "constant_with_warmup", "inverse_sqrt"}


def scheduler_kwargs(cfg: dict) -> dict:
    """Map the babysitter's schedule config onto TrainingArguments kwargs."""
    raw = cfg.get("lr_scheduler_type")
    name = raw.strip().lower() if isinstance(raw, str) else None
    if raw is not None and name not in _HF_SCHEDULERS:
        logging.getLogger(__name__).warning(
            "unrecognized lr_scheduler_type %r; falling back to a constant LR", raw)
    out = {"lr_scheduler_type": name if name in _HF_SCHEDULERS else "constant"}
    warmup = cfg.get("warmup_steps")
    if warmup is not None:
        out["warmup_steps"] = int(warmup)
    return out


def build_context(args) -> str:
    """Run background surfaced to the babysitter at plan/act/reflect time.

    Describes the task + fixed setup so planning is informed, and pins the exact
    initial-config keys and lr_scheduler_type vocabulary the trainer actually honors,
    so the agent chooses from valid options instead of inventing schedule names.
    """
    allowed = ", ".join(sorted(_HF_SCHEDULERS))
    return (
        f"Task: fine-tune {args.model_ckpt} for binary IMDB sentiment classification "
        "(2 labels) with AdamW, minimizing eval cross-entropy loss (lower is better).\n"
        f"Data: {args.n_train} train / {args.n_eval} eval examples, max sequence length 256.\n"
        f"Fixed every round: max_steps={args.max_steps}, per-device batch size "
        f"{args.batch_size}, evaluate + log every {args.eval_steps} steps.\n"
        "Initial config: reply with only these keys -- 'learning_rate' (float, peak LR), "
        f"'lr_scheduler_type' (EXACTLY one of: {allowed}), and optional 'warmup_steps' (int). "
        "Do NOT invent schedule names (e.g. 'cosine_decay', 'piecewise_linear') or pass "
        "milestone lists/objects; any unrecognized choice falls back to a constant LR.\n"
        "Babysitting: during the run you steer via knobs set at control points -- lr, "
        "weight_decay, adam_beta1, eps, max_grad_norm -- and control actions such as "
        "save_checkpoint, evaluate, and stop. Set weight_decay through the knob (it is not "
        "read from the initial config). Setting 'lr' rescales the active schedule's peak; "
        "the schedule shape from lr_scheduler_type is preserved."
    )


def main():
    args = parse_args()
    run_dir, run_id = setup_logs("hf_bert_imdb_frontend")
    aim_repo = args.aim_repo or os.path.join(run_dir, "aim")

    agent = LLMAgent(every=args.agent_every, name=args.model, client=build_client(args))
    session = TrainingSession(
        goal="eval_loss", memory=os.path.join(run_dir, "memory.jsonl"),
        agent=agent, max_rounds=args.max_rounds, context=build_context(args), seed=args.seed,
        frontend={"repo": aim_repo, "experiment": f"hf_bert_imdb_frontend-{run_id}",
                  "up": not args.no_ui, "ui_port": args.ui_port})
    session.start()

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)
    data = datasets.load_dataset("stanfordnlp/imdb").map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=256),
        batched=True, remove_columns=["text"]).rename_column("label", "labels")
    train_ds = data["train"].shuffle(seed=args.seed).select(range(args.n_train))
    eval_ds = data["test"].shuffle(seed=args.seed).select(range(args.n_eval))

    InteractiveTrainer = make_interactive(Trainer)

    def train_round(session, ctx):
        session.plan_round(ctx, apply=False)
        cfg = (ctx.plan.config if ctx.plan else {}) or {}
        lr = float(cfg.get("learning_rate", cfg.get("lr", args.learning_rate)))
        sched = scheduler_kwargs(cfg)
        InteractiveTrainer(
            model=AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=2),
            args=TrainingArguments(
                output_dir=os.path.join(run_dir, f"round_{ctx.round}"),
                report_to=[], eval_strategy="steps",
                eval_steps=args.eval_steps, logging_steps=args.eval_steps,
                seed=args.seed, data_seed=args.seed, learning_rate=lr, **sched,
                max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size),
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer, data_collator=collator, session=session).train()

    if args.preflight:
        print("[frontend] pre-flight: configure in the UI, then click Start (resume) to train")
        session.wait_until_resumed()

    session.run_rounds(train_round)
    best = session.memory.best(direction="min")
    print(f"Best round {best['round']}: eval_loss={best['score']:.4f}")


if __name__ == "__main__":
    main()
