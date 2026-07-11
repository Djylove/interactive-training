from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interactive_training import Action, Plan, TrainingSession
from interactive_training.transport import AimTransport, CompositeTransport, HttpTransport


MODEL_REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
DATASET_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"


class ScriptedReferenceOperator:
    """Deterministic public-demo policy; never calls an external LLM."""

    name = "scripted-demo"
    every = 25

    def __init__(self) -> None:
        self._acted_rounds: set[int] = set()

    def describe(self) -> dict:
        return {
            "provider": "local-script",
            "model": "deterministic-policy",
            "base_url": None,
            "reasoning_effort": None,
            "api_key_set": False,
            "every": self.every,
        }

    def plan(self, ctx) -> Plan:
        return Plan(
            config={"lr": 2e-5, "weight_decay": 0.01},
            strategy=(
                "Use a lower learning rate and modest weight decay after the fixed "
                "baseline. This is a deterministic public-demo policy, not an LLM call."
            ),
        )

    def act(self, obs) -> list[Action]:
        if obs.round in self._acted_rounds or not obs.metrics_history:
            return []
        step = int(obs.metrics_history[-1].get("step", 0))
        if step < 25:
            return []
        self._acted_rounds.add(obs.round)
        return [
            Action(
                type="set_knob",
                payload={"name": "lr", "value": 1.5e-5},
                source=self.name,
            )
        ]

    def reflect(self, ctx, trajectory: list[dict], score: float) -> str:
        return (
            f"Round {ctx.round} reached eval_loss={score:.4f}. "
            "The public demo records this deterministic lesson in session memory."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny public CPU sentiment sandbox")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--aim-repo", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    return parser.parse_args()


def load_data(tokenizer, seed: int, revision: str):
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/imdb", revision=revision)
    train = dataset["train"].shuffle(seed=seed).select(range(512))
    valid = dataset["test"].shuffle(seed=seed).select(range(128))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    columns = ["text"]
    train = train.map(tokenize, batched=True, remove_columns=columns)
    valid = valid.map(tokenize, batched=True, remove_columns=columns)
    train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    valid.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return train, valid


@torch.no_grad()
def evaluate(model, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        total += float(output.loss) * batch["input_ids"].shape[0]
        count += batch["input_ids"].shape[0]
    model.train()
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    args.session_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed = 42
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    train_data, valid_data = load_data(tokenizer, seed, args.dataset_revision)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

    http = HttpTransport(host="127.0.0.1", port=args.port)
    aim = AimTransport(
        repo=args.aim_repo,
        experiment=f"public-cpu-{args.session_id}",
        control_url=lambda: http.url,
    )
    transport = CompositeTransport([http, aim])
    operator = ScriptedReferenceOperator()
    session = TrainingSession(
        goal="eval_loss",
        transport=transport,
        run_id=f"public-cpu-{args.session_id}",
        memory=str(args.session_dir / "memory.jsonl"),
        agent=operator,
        max_rounds=1,
        context=(
            "Public CPU micro-demo: tiny BERT sentiment classification on a fixed "
            "IMDB subset. The attached operator is deterministic and makes no LLM calls."
        ),
        seed=seed,
        watch_metrics=["eval_loss"],
    )
    for forbidden_action in (
        "load_checkpoint",
        "pause",
        "resume",
        "reset_module",
        "note",
        "set_agent",
        "configure_agent",
        "set_context",
    ):
        session.registry.unregister(forbidden_action)

    def train_round(active_session: TrainingSession, ctx) -> None:
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            revision=args.model_revision,
            num_labels=2,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
        config = {"max_grad_norm": 1.0}
        active_session.register_optimizer_lr(optimizer)
        active_session.register_knob(
            "weight_decay",
            get=lambda: float(optimizer.param_groups[0]["weight_decay"]),
            set=lambda value: [
                group.__setitem__("weight_decay", float(value))
                for group in optimizer.param_groups
            ],
            min=0.0,
            max=0.2,
            description="AdamW weight decay",
        )
        active_session.register_knob(
            "max_grad_norm",
            get=lambda: config["max_grad_norm"],
            set=lambda value: config.__setitem__("max_grad_norm", float(value)),
            min=0.1,
            max=5.0,
            description="gradient clipping norm",
        )
        active_session.bind_model(model)
        active_session.plan_round(ctx)

        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            train_data,
            batch_size=8,
            shuffle=True,
            generator=generator,
        )
        iterator = iter(loader)
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            output.loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
            )
            optimizer.step()
            metrics = {
                "loss": float(output.loss.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm,
            }
            if step % args.eval_every == 0 or step == args.steps:
                metrics["eval_loss"] = evaluate(model, valid_loader)
            control = active_session.step(metrics, step=step)
            if control.evaluate:
                active_session.report_eval(
                    {"eval_loss": evaluate(model, valid_loader)}
                )
            if control.save:
                path = args.session_dir / f"round-{ctx.round}-step-{step}.pt"
                torch.save(model.state_dict(), path)
                active_session.checkpoint_saved(str(path), step, tag=control.tag)
            if control.stop:
                break
            if args.delay:
                time.sleep(args.delay)

    memory = session.run_rounds(train_round)
    best = memory.best("min")
    print(
        f"Public CPU session complete: best round={best['round']} "
        f"eval_loss={best['score']:.5f}"
    )


if __name__ == "__main__":
    main()
