"""Multi-round LLM-babysat BERT/IMDB finetuning (plan §3.6, §7.2).

A GPT-5.5 agent over OpenRouter plans -> babysits online -> reflects each round,
minimizing validation loss. Set OPENROUTER_API_KEY, then run in the Apptainer env:

    srun --jobid <ID> python -m examples.hf_bert_imdb_multiround --max-rounds 10
"""
from __future__ import annotations

import argparse
import os
import time

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

import logging
setup_logging()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-round LLM-babysat BERT/IMDB finetuning")
    p.add_argument("--model-ckpt", default="bert-base-uncased")
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--n-train", type=int, default=16000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="human-defined initial learning rate used for round 0")
    p.add_argument("--no-agent", action="store_true")
    p.add_argument("--max-steps", type=int, default=1000, help="train steps per round")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed (re-applied each round for a reproducible baseline)")
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/hf_bert_imdb_multiround/<run-id>/imdb_bert")
    p.add_argument("--memory-path", default=None,
                   help="defaults to logs/hf_bert_imdb_multiround/<run-id>/bert_imdb_memory.jsonl")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"],
                   help="native OpenAI (api.openai.com) or OpenRouter")
    p.add_argument("--model", default="gpt-5.5",
                   help="babysitter model slug (e.g. 'gpt-5.5' for openai, 'openai/gpt-5.5' for openrouter)")
    p.add_argument("--base-url", default=None,
                   help="override the provider's default endpoint")
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--agent-every", type=int, default=100, help="agent acts every N training steps")
    p.add_argument("--wandb-project", default="interactive_training_v2",
                   help="wandb project name (defaults to $WANDB_PROJECT)")
    p.add_argument("--wandb-experiment", default="hf_bert_imdb_multiround",
                   help="experiment name; rounds of one launch share a group derived from this")
    p.add_argument("--wandb-run-id", default=None,
                   help="unique id appended to the group/run names (default: launch timestamp)")
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    p.add_argument("--frontend", nargs="?", const=True, default=None, metavar="AIM_REPO",
                   help="serve the Aim frontend transport; optional value = Aim repo path "
                        "(default: logs/<experiment>/<run-id>/aim)")
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
    run_id = args.wandb_run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("hf_bert_imdb_multiround", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "imdb_bert")
    memory_path = args.memory_path or os.path.join(run_dir, "bert_imdb_memory.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)
    data = datasets.load_dataset("stanfordnlp/imdb").map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=256),
        batched=True, remove_columns=["text"]).rename_column("label", "labels")
    train_ds = data["train"].shuffle(seed=args.seed).select(range(args.n_train))
    eval_ds = data["test"].shuffle(seed=args.seed).select(range(args.n_eval))

    agent = None if args.no_agent else LLMAgent(every=args.agent_every, name=args.model,
                                                client=build_client(args))
    frontend = args.frontend
    if frontend is True:
        frontend = {"repo": os.path.join(run_dir, "aim"),
                    "experiment": f"{args.wandb_experiment}-{run_id}"}
    session = TrainingSession(goal="eval_loss", memory=memory_path,
                              agent=agent, max_rounds=args.max_rounds,
                              context=build_context(args), seed=args.seed,
                              frontend=frontend)
    InteractiveTrainer = make_interactive(Trainer)
    use_wandb = not args.no_wandb

    wandb_group = f"{args.wandb_experiment}-{run_id}"
    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_GROUP"] = wandb_group
        print(f"[wandb] project={args.wandb_project} group={wandb_group}")

    def train_round(session, ctx):
        # The babysitter sometimes names the initial LR "lr" instead of "learning_rate";
        # accept either so the round doesn't silently fall back to the human default.
        session.plan_round(ctx, apply=False)
        cfg = (ctx.plan.config if ctx.plan else {}) or {}
        lr = float(cfg.get("learning_rate", cfg.get("lr", args.learning_rate)))
        sched = scheduler_kwargs(cfg)  # honor the plan's lr schedule/warmup, mapped to HF
        run_name = f"{wandb_group}-round{ctx.round}"
        InteractiveTrainer(
            model=AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=2),
            args=TrainingArguments(
                output_dir=f"{output_dir}/round_{ctx.round}",
                report_to=["wandb"] if use_wandb else [],
                run_name=run_name,
                eval_strategy="steps", eval_steps=args.eval_steps, logging_steps=args.eval_steps,
                seed=args.seed, data_seed=args.seed,
                learning_rate=lr, **sched,
                max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=4),
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer, data_collator=collator, session=session).train()
        if use_wandb:
            import wandb
            if wandb.run is not None:
                score = session.goal.score(session.history) if session.goal else None
                wandb.log({"round": ctx.round, "round_eval_loss": score, "round_lr": lr})
                wandb.config.update({"round": ctx.round, "learning_rate": lr,
                                     "babysitter_model": args.model, "run_name": run_name},
                                    allow_val_change=True)
                wandb.finish()

    session.run_rounds(train_round)
    best = session.memory.best(direction="min")
    print(f"Best round {best['round']}: eval_loss={best['score']:.4f} (see {session.memory.metrics_path})")


if __name__ == "__main__":
    main()
