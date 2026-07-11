"""Multi-round LLM-babysat cross-domain 3-class sentiment data mixing (HF Trainer).
"""
from __future__ import annotations

import argparse
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import datasets
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from interactive_training.agents.agent import LLMAgent
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED, seed_everything
from interactive_training.core.goals import Goal
from examples._common import build_client, setup_logging
from examples._paths import setup_logs
from interactive_training.integrations.hf_trainer import make_interactive
from interactive_training.recipes._common import bind_dict_knob

import logging
setup_logging()
logger = logging.getLogger(__name__)

# Complementary-deficient by design: each source is missing one axis (domain fit or
# class coverage), so no single source suffices for the 3-class target -- the optimum
# is a combination. ('movie'/SST-5 is intentionally NOT a default source: it is both
# domain-close and class-complete, which makes it near-sufficient alone.)
SOURCES = ["tweet", "finance", "product"]
COVERAGE = {  # honest operator knowledge, surfaced to the agent in the context
    "movie": "movie-review snippets labeled on a graded scale (all three classes)",
    "tweet": "tweets covering all three classes",
    "finance": "financial news sentences, mostly neutral",
    "product": "product reviews with only negative/positive rows (no neutral at all)",
}
CLASSES = ["negative", "neutral", "positive"]
# 5-point scales (SST-5 sentiment strength, Yelp 1-5 stars) -> 3 classes.
_FIVE_TO_THREE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

def _cap(ds: datasets.Dataset, limit: int, seed: int) -> datasets.Dataset:
    """Shuffle then keep at most `limit` rows. Applied to the RAW source BEFORE the
    per-row normalization map, so huge sources (amazon 3.6M, yelp 560k) don't get
    fully mapped/materialized just to discard 99% of the rows downstream."""
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(len(ds), limit)))


def _normalize(source: str, limit: int, seed: int) -> datasets.Dataset:
    """Load one training source (capped to ~`limit` rows) and normalize to columns
    text (str) + labels (0=negative, 1=neutral, 2=positive). Class coverage varies
    by source ON PURPOSE (that imbalance is part of the experiment): product is
    polar-only, finance is neutral-heavy, movie/tweet cover all three."""
    if source == "movie":  # SST-5 movie snippets, 5-point scale -> 3 classes
        ds = _cap(datasets.load_dataset("SetFit/sst5", split="train"), limit, seed)
        return ds.map(lambda b: {"text": b["text"], "labels": _FIVE_TO_THREE[b["label"]]},
                      remove_columns=ds.column_names)
    if source == "tweet":  # labels already 0=neg 1=neutral 2=pos
        ds = _cap(datasets.load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train"),
                  limit, seed)
        return ds.map(lambda b: {"text": b["text"], "labels": b["label"]},
                      remove_columns=ds.column_names)
    if source == "finance":  # labels already 0=neg 1=neutral 2=pos (tiny; no cap needed)
        ds = datasets.load_dataset("warwickai/financial_phrasebank_mirror", split="train")
        return ds.map(lambda b: {"text": b["sentence"], "labels": b["label"]},
                      remove_columns=ds.column_names)
    if source == "product":  # binary by construction: no neutral coverage at all
        ds = _cap(datasets.load_dataset("fancyzhx/amazon_polarity", split="train"), limit, seed)
        return ds.map(lambda b: {"text": b["title"] + ". " + b["content"],
                                 "labels": 0 if b["label"] == 0 else 2},
                      remove_columns=ds.column_names)
    raise ValueError(f"unknown source {source!r}")


def _flip_labels(ds: datasets.Dataset, frac: float, seed: int) -> datasets.Dataset:
    """Relabel a `frac` fraction of rows to a uniformly random OTHER class
    (synthetic noise trap)."""
    rng = random.Random(seed)
    offset = [rng.randrange(1, len(CLASSES)) if rng.random() < frac else 0
              for _ in range(len(ds))]
    return ds.map(lambda b, i: {"labels": (b["labels"] + offset[i]) % len(CLASSES)},
                  with_indices=True)

class MixtureDataset(torch.utils.data.IterableDataset):
    """Infinite stream; each example is drawn from a source in proportion to the
    LIVE mixture weights read from `cfg` (the dict the w_<source> knobs write to).
    With `class_balance=True` (the standard non-LLM baseline) rows are drawn
    uniformly over the classes each source actually covers, instead of at the
    source's natural class frequencies."""

    def __init__(self, sources: dict, cfg: dict, seed: int, class_balance: bool = False):
        self.sources, self.cfg = sources, cfg
        self.rng = random.Random(seed)
        self.buckets = None
        if class_balance:  # per-source class -> row-index buckets, built once
            self.buckets = {}
            for n, ds in sources.items():
                b: dict[int, list[int]] = {}
                for i, y in enumerate(ds["labels"]):
                    b.setdefault(int(y), []).append(i)
                self.buckets[n] = b

    def __iter__(self):
        names = list(self.sources)
        while True:
            w = [max(float(self.cfg[f"w_{n}"]), 0.0) for n in names]
            if sum(w) <= 0:  # guard: agent drove everything to zero
                w = [1.0] * len(names)
            n = self.rng.choices(names, weights=w, k=1)[0]
            src = self.sources[n]
            if self.buckets is None:
                yield src[self.rng.randrange(len(src))]
            else:
                rows = self.buckets[n][self.rng.choice(list(self.buckets[n]))]
                yield src[rows[self.rng.randrange(len(rows))]]


class HeuristicMixCallback(TrainerCallback):
    """weight_s proportional to (latest eval_<s>_loss)^2, renormalized. The standard
    'feed the worst-off source' rule; intentionally fixed, do not tune."""

    def __init__(self, cfg):
        self.cfg = cfg

    def on_evaluate(self, args, state, control, **kwargs):
        losses = {}
        for s in SOURCES:  # scan log_history backwards for each key
            key = f"eval_{s}_loss"
            losses[s] = next((h[key] for h in reversed(state.log_history) if key in h), None)
        if any(v is None for v in losses.values()):
            return  # first evals still in flight (dict eval sets evaluate one at a time)
        w = {s: max(v, 1e-6) ** 2 for s, v in losses.items()}
        total = sum(w.values())
        for s in SOURCES:
            self.cfg[f"w_{s}"] = w[s] / total


class KnobLogCallback(TrainerCallback):
    """Inject normalized knob/w_<source> and knob/cw_<class> values into each
    logged dict so mixture/class-weight trajectories plot in wandb."""

    def __init__(self, cfg):
        self.cfg = cfg

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        w = {s: max(float(self.cfg[f"w_{s}"]), 0.0) for s in SOURCES}
        total = sum(w.values()) or 1.0
        for s in SOURCES:
            logs[f"knob/w_{s}"] = w[s] / total
        for c in CLASSES:
            logs[f"knob/cw_{c}"] = max(float(self.cfg[f"cw_{c}"]), 0.0)


class QuietProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        return


class SummaryCallback(TrainerCallback):
    """Collapse HF's per-eval-dataset log spam (one dict per source + target, plus a
    train-loss line every logging step) into a single line per eval point."""

    def __init__(self, cfg):
        self.cfg = cfg

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not any(k.startswith("eval_target") for k in (metrics or {})):
            return

        def latest(key):
            return next((h[key] for h in reversed(state.log_history) if key in h), None)

        val = " ".join(f"{s}={latest(f'eval_{s}_loss'):.3f}" for s in SOURCES
                       if latest(f"eval_{s}_loss") is not None)
        w = {s: max(float(self.cfg[f"w_{s}"]), 0.0) for s in SOURCES}
        tot = sum(w.values()) or 1.0
        mix = " ".join(f"{s}={w[s] / tot:.2f}" for s in SOURCES)
        cw = " ".join(f"{c[:3]}={max(float(self.cfg[f'cw_{c}']), 0.0):.2f}" for c in CLASSES)
        f1s = " ".join(f"{c[:3]}={latest(f'eval_target_f1_{c}'):.3f}" for c in CLASSES
                       if latest(f"eval_target_f1_{c}") is not None)
        loss, tgt, acc = latest("loss"), latest("eval_target_macro_f1"), latest("eval_target_accuracy")
        logger.info("step %d | train_loss=%s | val[%s] | target_macro_f1=%.4f acc=%.4f f1[%s] | mix[%s] | cw[%s]",
                    state.global_step, f"{loss:.3f}" if loss is not None else "n/a",
                    val, tgt, acc, f1s, mix, cw)


def compute_metrics(p):
    preds, labels = p.predictions.argmax(-1), p.label_ids
    out = {"accuracy": float((preds == labels).mean())}
    f1s = []
    for c, name in enumerate(CLASSES):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"f1_{name}"], out[f"recall_{name}"] = float(f1), float(rec)
        f1s.append(f1)
    out["macro_f1"] = float(sum(f1s) / len(f1s))
    return out

def build_context(args) -> str:
    return (
        f"Task: fine-tune {args.model} for 3-class sentiment classification "
        f"(labels: {', '.join(CLASSES)}) on a mixture of {len(SOURCES)} labeled text "
        f"sources ({', '.join(SOURCES)}). Every batch samples each example from a "
        f"source in proportion to the mixture-weight knobs 'w_<source>' (relative, "
        f">=0, renormalized on the fly). The training loss is cross-entropy with "
        f"per-class weights set by the 'cw_<class>' knobs (relative, >=0).\n"
        f"Sources: " + "; ".join(f"{s} = {COVERAGE[s]}" for s in SOURCES) + ".\n"
        f"Goal metric 'eval_target_macro_f1' (higher is better): macro-averaged F1 "
        f"over the three classes on a held-out evaluation set from a domain that is "
        f"not one of the training sources; 'eval_target_accuracy' and per-class "
        f"'eval_target_f1_<class>' / 'eval_target_recall_<class>' are reported on the "
        f"same set. Per-source validation losses are reported as 'eval_<source>_loss'. "
        f"All are measured every {args.eval_steps} steps.\n"
        f"Algorithm: AdamW with a constant learning-rate schedule; under control are "
        f"the mixture weights, the class weights, and the learning-rate knob 'lr'. "
        f"{args.max_steps} steps per round, batch size {args.batch_size}.\n"
        f"Data caveat: the sources differ in domain relevance to the target AND in "
        f"class coverage/balance -- some sources are missing or under-represent some "
        f"classes, so source weights and class balance interact.\n"
        f"Initial config: reply with only these keys -- 'learning_rate' (float) and "
        f"optionally 'w_<source>' starting weights (floats in [0,1]) and 'cw_<class>' "
        f"starting class weights (floats in [0,5])."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-round LLM-babysat sentiment data mixing (HF Trainer)")
    p.add_argument("--mode", default="babysat",
                   choices=["babysat", "no-agent", "uniform", "heuristic", "class-balanced"],
                   help="which controller to run (each is one experiment arm); "
                        "'class-balanced' = no agent + uniform class sampling within each source")
    p.add_argument("--no-agent", action="store_true",
                   help="fixed uniform-mixture baseline, no LLM babysitter (alias for --mode no-agent)")
    p.add_argument("--sources", nargs="+", default=None, choices=list(COVERAGE),
                   help="override the training sources (subsets for transfer probes, or "
                        "re-adding 'movie'); default: SOURCES")
    p.add_argument("--model", default="bert-base-uncased",
                   help="HF checkpoint to fine-tune (the babysitter LLM is --agent-model)")
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2000, help="train steps per round")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1.5e-5,
                   help="round-0 default; the plan may override per round")
    # Fixed optimizer hyperparameters (NOT knobs: only lr stays agent-tunable).
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (fixed)")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient-norm clip (fixed)")
    p.add_argument("--adam-beta1", type=float, default=0.9, help="AdamW beta1 (fixed)")
    p.add_argument("--adam-beta2", type=float, default=0.999, help="AdamW beta2")
    p.add_argument("--adam-epsilon", type=float, default=1e-8, help="AdamW epsilon (fixed)")
    p.add_argument("--per-source-cap", type=int, default=20000, help="max train examples per source")
    p.add_argument("--val-per-source", type=int, default=500, help="val examples held out per source")
    p.add_argument("--target-eval-n", type=int, default=3000,
                   help="held-out target (Yelp full) eval examples")
    p.add_argument("--noise-source", default=None, nargs="+", choices=SOURCES,
                   help="relabel a fraction of one or more sources' train rows to a random "
                        "other class (synthetic noise trap)")
    p.add_argument("--noise-frac", type=float, nargs="+", default=[0.35],
                   help="fraction of train labels to corrupt: a single shared value, or one "
                        "value per --noise-source (aligned by order)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed (re-applied each round for a reproducible baseline)")
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/data_mixing_sentiment/<run-id>/sentiment")
    p.add_argument("--memory-path", default=None,
                   help="defaults to logs/data_mixing_sentiment/<run-id>/sentiment_memory.jsonl")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"],
                   help="native OpenAI (api.openai.com) or OpenRouter")
    p.add_argument("--agent-model", default="gpt-5.5",
                   help="babysitter model slug (e.g. 'gpt-5.5' for openai, 'openai/gpt-5.5' for openrouter)")
    p.add_argument("--base-url", default=None, help="override the provider's default endpoint")
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--agent-every", type=int, default=100, help="agent acts every N training steps")
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="data_mixing_sentiment")
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    return p.parse_args()


def main():
    global SOURCES
    args = parse_args()
    if args.no_agent:
        args.mode = "no-agent"
    if args.sources:  # e.g. single-source transfer probes; callbacks read the global
        SOURCES = list(dict.fromkeys(args.sources))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("data_mixing_sentiment", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "sentiment")
    memory_path = args.memory_path or os.path.join(run_dir, "sentiment_memory.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)

    def tokenize(ds, max_length):
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                      batched=True, remove_columns=["text"])

    # Resolve the per-source noise fraction: one shared value, or one aligned per source.
    noise = {}
    if args.noise_source:
        fracs = args.noise_frac
        if len(fracs) == 1:
            fracs = fracs * len(args.noise_source)
        if len(fracs) != len(args.noise_source):
            raise SystemExit("--noise-frac must be a single value or one per --noise-source")
        noise = {s: f for s, f in zip(args.noise_source, fracs)}

    train_sources, val_ds = {}, {}
    cap = args.per_source_cap + args.val_per_source  # only load/normalize what we keep
    for s in SOURCES:
        ds = _normalize(s, cap, args.seed).shuffle(seed=args.seed)
        n_val = min(args.val_per_source, len(ds) // 2)
        val = ds.select(range(n_val))
        train = ds.select(range(n_val, min(len(ds), n_val + args.per_source_cap)))
        if noise.get(s, 0.0) > 0:
            # offset the seed per source so multiple noised sources flip independently;
            # never flip val/target rows.
            train = _flip_labels(train, noise[s], args.seed + SOURCES.index(s))
        train_sources[s] = tokenize(train, max_length=128)
        val_ds[s] = tokenize(val, max_length=128)

    # Held-out target domain: Yelp reviews, 1-5 stars mapped to the 3 classes
    # (1-2=neg, 3=neutral, 4-5=pos). Naturally class-imbalanced, never trained on.
    yelp = _cap(datasets.load_dataset("Yelp/yelp_review_full", split="test"),
                args.target_eval_n, args.seed)
    yelp = yelp.map(lambda b: {"text": b["text"], "labels": _FIVE_TO_THREE[b["label"]]},
                    remove_columns=yelp.column_names)
    target_eval = tokenize(yelp, max_length=256)  # reviews are long; collator pads per batch
    eval_dataset = {**{s: val_ds[s] for s in SOURCES}, "target": target_eval}

    agent = None if args.mode != "babysat" else LLMAgent(every=args.agent_every, name=args.agent_model,
                                                          client=build_client(args))
    session = TrainingSession(goal=Goal(name="target_macro_f1", metric="eval_target_macro_f1",
                                        direction="max"),
                              memory=memory_path,
                              agent=agent, max_rounds=args.max_rounds,
                              context=build_context(args), seed=args.seed,
                              watch_metrics=[f"eval_{s}_loss" for s in SOURCES]
                              + [f"eval_target_f1_{c}" for c in CLASSES]
                              + ["eval_target_accuracy"])
    InteractiveTrainer = make_interactive(Trainer, extra_optimizer_knobs=False)

    class WeightedLossTrainer(InteractiveTrainer):
        """Training-time cross-entropy with LIVE per-class weights read from the
        cw_<class> knobs; eval keeps plain CE so eval_*_loss stays comparable."""

        def __init__(self, *targs, class_weight_cfg: dict, **tkwargs):
            super().__init__(*targs, **tkwargs)
            self._cw_cfg = class_weight_cfg

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weight = None
            if model.training:
                w = [max(float(self._cw_cfg[f"cw_{c}"]), 0.0) for c in CLASSES]
                if sum(w) <= 0:  # guard: agent drove everything to zero
                    w = [1.0] * len(CLASSES)
                weight = torch.tensor(w, device=outputs.logits.device,
                                      dtype=outputs.logits.dtype)
                weight = weight * (len(CLASSES) / weight.sum())  # mean 1: loss scale ~ lr-stable
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss
    use_wandb = not args.no_wandb

    wandb_group = f"{args.wandb_experiment}-{run_id}"
    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_GROUP"] = wandb_group
        print(f"[wandb] project={args.wandb_project} group={wandb_group}")

    def train_round(session, ctx):
        cfg = {**{f"w_{s}": 1.0 for s in SOURCES}, **{f"cw_{c}": 1.0 for c in CLASSES}}
        for s in SOURCES:
            bind_dict_knob(session, cfg, f"w_{s}", min=0.0, max=1.0,
                           description=f"relative sampling weight for the {s} source")
        for c in CLASSES:
            bind_dict_knob(session, cfg, f"cw_{c}", min=0.0, max=5.0,
                           description=f"relative training cross-entropy weight for the {c} class")

        session.plan_round(ctx, apply=False)  # w_/cw_ knobs already bound -> shown in plan
        plan_cfg = (ctx.plan.config if ctx.plan else {}) or {}
        lr = float(plan_cfg.get("learning_rate", plan_cfg.get("lr", args.learning_rate)))
        for s in SOURCES:  # planned starting mixture, if any
            if f"w_{s}" in plan_cfg:
                cfg[f"w_{s}"] = min(max(float(plan_cfg[f"w_{s}"]), 0.0), 1.0)
        for c in CLASSES:  # planned starting class weights, if any
            if f"cw_{c}" in plan_cfg:
                cfg[f"cw_{c}"] = min(max(float(plan_cfg[f"cw_{c}"]), 0.0), 5.0)

        extra_callbacks = [SummaryCallback(cfg)]  # one consolidated line per eval point
        if args.mode == "heuristic":
            extra_callbacks.append(HeuristicMixCallback(cfg))
        if use_wandb:
            extra_callbacks.append(KnobLogCallback(cfg))

        seed_everything(args.seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                                   num_labels=len(CLASSES))
        trainer = WeightedLossTrainer(
            model=model,
            class_weight_cfg=cfg,
            args=TrainingArguments(
                output_dir=f"{output_dir}/round_{ctx.round}",
                report_to=["wandb"] if use_wandb else [],
                run_name=f"{wandb_group}-round{ctx.round}",
                eval_strategy="steps", eval_steps=args.eval_steps, logging_steps=args.eval_steps,
                seed=args.seed, data_seed=args.seed,
                learning_rate=lr, lr_scheduler_type="constant",
                weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
                adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
                adam_epsilon=args.adam_epsilon,  # round-0 starts; then tuned via knobs
                max_steps=args.max_steps,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=64,
                save_strategy="no",
                dataloader_num_workers=0),
            train_dataset=MixtureDataset(train_sources, cfg, seed=args.seed + ctx.round,
                                         class_balance=args.mode == "class-balanced"),
            eval_dataset=eval_dataset,
            processing_class=tokenizer, data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=extra_callbacks,
            session=session)
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(QuietProgressCallback)
        trainer.train()

        if use_wandb:
            import wandb
            if wandb.run is not None:
                score = session.goal.score(session.history) if session.goal else None
                wandb.log({"round": ctx.round, "round_macro_f1": score})
                wandb.config.update({"mode": args.mode, "round": ctx.round, "learning_rate": lr,
                                     "babysat": agent is not None,
                                     "babysitter_model": args.agent_model},
                                    allow_val_change=True)
                wandb.finish()

    session.run_rounds(train_round)
    best = session.memory.best(direction="max")
    baseline = next((r for r in session.memory.rounds if r["round"] == 0), None)
    if baseline:
        print(f"Baseline (round 0) eval_target_macro_f1={baseline['score']:.4f}")
    print(f"Best round {best['round']}: eval_target_macro_f1={best['score']:.4f} (see {session.memory.metrics_path})")


if __name__ == "__main__":
    main()
