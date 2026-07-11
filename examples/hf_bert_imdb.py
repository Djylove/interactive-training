"""Single-round interactive BERT/IMDB finetuning (plan §7.2) — the old-repo analog.

Like the original `train_bert_imdb.py`: one run, steered live from a frontend (change
lr, pause/resume, save/load, evaluate). Run in the Apptainer env, then point a frontend
at the served port:

    srun --jobid <ID> python -m examples.hf_bert_imdb --port 9876
"""
from __future__ import annotations

import argparse

import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from examples._paths import setup_logs
from interactive_training.integrations.hf_trainer import make_interactive
from interactive_training.transport.server import HttpTransport


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-round interactive BERT/IMDB finetuning")
    p.add_argument("--model-ckpt", default="bert-base-uncased")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/hf_bert_imdb/<run-id>/imdb_bert")
    p.add_argument("--port", type=int, default=9876, help="serve HTTP/WS for the frontend")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir, _ = setup_logs("hf_bert_imdb")
    output_dir = args.output_dir or run_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)
    data = datasets.load_dataset("stanfordnlp/imdb").map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=256),
        batched=True, remove_columns=["text"]).rename_column("label", "labels")
    train_ds = data["train"].shuffle(seed=args.seed).select(range(args.n_train))
    eval_ds = data["test"].shuffle(seed=args.seed).select(range(args.n_eval))

    session = TrainingSession(transport=HttpTransport(port=args.port), seed=args.seed)
    trainer_cls = make_interactive(Trainer)
    trainer = trainer_cls(
        model=AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=2),
        args=TrainingArguments(
            output_dir=output_dir, report_to=[], eval_strategy="steps",
            eval_steps=args.eval_steps, logging_steps=args.eval_steps, max_steps=args.max_steps,
            seed=args.seed, data_seed=args.seed,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size),
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, data_collator=collator, session=session)
    trainer.train()


if __name__ == "__main__":
    main()
