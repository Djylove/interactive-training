"""LLM-babysat per-layer learning rates for a deep LLM trained from scratch.

Inspired by LLR (openreview vhhwY0AVgu): different layers of a deep Transformer want
different learning rates. Here an LLM babysitter owns one LR knob per decoder layer of a
fresh deep-but-narrow Qwen (RoPE/RMSNorm/SwiGLU/GQA), reads the loss + per-layer grad norms
each control point, and decides each layer's LR on the fly. Goal is 'val_loss';
`--no-agent` is the uniform-LR baseline.

Run (inside apptainer, after `source init.sh`):
    python -m examples.layerwise_lr_gpt --max-rounds 4             # babysat
    python -m examples.layerwise_lr_gpt --no-agent --max-rounds 1  # uniform-LR baseline
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (see core.determinism)

import datasets
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from interactive_training.agents.agent import LLMAgent, Observation
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from examples._common import build_client, setup_logging
from examples._paths import setup_logs

setup_logging()
logger = logging.getLogger(__name__)


class FineWeb(IterableDataset):
    PATH, NAME, TEXT = "HuggingFaceFW/fineweb-edu", "sample-10BT", "text"

    def __init__(self, tok, seq_len, eval_docs):
        self.tok, self.seq_len, self.eval_docs = tok, seq_len, eval_docs

    def _stream(self, skip=None, take=None):
        ds = datasets.load_dataset(path=self.PATH, name=self.NAME, split="train", streaming=True)
        return ds.skip(skip).take(take) if skip and take else (ds.skip(skip) if skip else
                                                               ds.take(take) if take else ds)

    def _chunks(self, raw):
        buf, eos = [], self.tok.eos_token_id
        for ex in raw:
            if not (text := ex.get(self.TEXT)):
                continue
            buf += self.tok(text, add_special_tokens=False).input_ids + [eos]
            while len(buf) >= self.seq_len:
                yield torch.tensor(buf[:self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len:]

    def __iter__(self):
        while True:  # restart the stream when the shard is exhausted
            yield from self._chunks(self._stream(skip=self.eval_docs))

    def eval_blocks(self, n_blocks):
        blocks = []
        for b in self._chunks(self._stream(take=self.eval_docs)):
            blocks.append(b)
            if len(blocks) >= n_blocks:
                break
        return blocks

    @staticmethod
    def collate(blocks):
        ids = torch.stack(blocks, 0)
        return {"input_ids": ids, "labels": ids.clone()}  # HF shifts labels internally


def build_param_groups(model):
    """One param group per decoder layer, plus separate groups for the token embeddings
    ('lr_embed'), the LM head ('lr_head'), and any remaining params ('lr_other', e.g. the
    final norm). Embeddings and head are only distinct groups because build_model unties
    their weights; a tied model shares one tensor and could not carry two learning rates."""
    groups, names, seen = [], [], set()
    for i, layer in enumerate(model.model.layers):
        params = list(layer.parameters())
        groups.append({"params": params})
        names.append(f"lr_block_{i}")
        seen.update(id(p) for p in params)

    def take(module, name):
        params = [p for p in (module.parameters() if module is not None else []) if id(p) not in seen]
        if params:
            groups.append({"params": params})
            names.append(name)
            seen.update(id(p) for p in params)

    take(model.get_input_embeddings(), "lr_embed")
    take(model.get_output_embeddings(), "lr_head")
    other = [p for p in model.parameters() if id(p) not in seen]
    if other:
        groups.append({"params": other})
        names.append("lr_other")
    return groups, names


def register_block_lrs(session, names, base_lr):
    """Register one LR knob per group. Knobs hold a *base* LR; the applied optimizer LR is
    base x a shared warmup+cosine factor, so the agent tunes per-layer amplitude, not shape."""
    base = {name: base_lr for name in names}

    labels = {"lr_embed": "token embeddings", "lr_head": "LM head (output projection)",
              "lr_other": "final norm / other params"}

    def make(idx, name):
        # keep the per-knob description tiny; the shared "how to tune LR" guidance lives once
        # in the run context, so it isn't repeated 25x in the tool schema.
        where = labels.get(name, f"decoder layer {idx}")
        session.register_knob(
            name, get=lambda n=name: base[n], set=lambda v, n=name: base.__setitem__(n, float(v)),
            min=0.0, max=10 * base_lr, description=where)

    for i, name in enumerate(names):
        make(i, name)
    return base


def apply_lr(groups, names, base, factor):
    for g, name in zip(groups, names):
        g["lr"] = base[name] * factor


def sched_factor(step, warmup, max_steps, min_ratio=0.1):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(1, max_steps - warmup), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def block_grad_norms(groups, names):
    out = {}
    for g, name in zip(groups, names):
        sq = sum(float(p.grad.pow(2).sum()) for p in g["params"] if p.grad is not None)
        out[name] = math.sqrt(sq)
    return out


def _latest(history, key):
    for row in reversed(history):
        if key in row and row[key] == row[key]:
            return row[key]
    return None


def _fmt_num(v):
    if v is None:
        return "--"
    if not math.isfinite(float(v)):
        return "--"
    v = float(v)
    return f"{v:.2e}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.3g}"


def _downsample_points(points, max_points=6):
    """Evenly subsample a history while preserving endpoints."""
    n = len(points)
    if n <= max_points:
        return points
    step = (n - 1) / (max_points - 1)
    idxs = sorted({round(i * step) for i in range(max_points)})
    return [points[i] for i in idxs]


def _metric_history(history, key, max_points=6):
    points = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            points.append((row.get("step", "?"), float(value)))
    points = _downsample_points(points, max_points=max_points)
    return ",".join(f"s{step}:{_fmt_num(value)}" for step, value in points) or "--"


class LayerwiseAgent(LLMAgent):
    """LLM babysitter that sees compact per-layer LR and grad-norm histories."""

    def __init__(self, block_names, **kw):
        super().__init__(**kw)
        self.block_names = block_names

    def _render(self, obs: Observation) -> str:
        history = obs.metrics_history or obs.recent_metrics
        render_obs = obs.model_copy(update={"knobs": {}})
        rows = ["Per-layer history (downsampled step:value; lr=base knob, gn=grad norm):"]
        for name in self.block_names:
            lr = obs.knobs[name].value if name in obs.knobs else None
            gn = _latest(history, f"gnorm/{name}")
            lr_hist = _metric_history(history, f"base_lr/{name}")
            gn_hist = _metric_history(history, f"gnorm/{name}")
            rows.append(f"  {name}: cur_lr={_fmt_num(lr)} cur_gn={_fmt_num(gn)} "
                        f"lr=[{lr_hist}] gn=[{gn_hist}]")
        prompt = super()._render(render_obs).replace("Knobs: (none)", "Knobs: see per-layer history below")
        return prompt + "\n" + "\n".join(rows)


def build_context(args) -> str:
    return (
        f"Task: train a deep modern LLM (Qwen arch: RoPE, RMSNorm, SwiGLU, GQA) with "
        f"{args.n_layer} decoder layers, hidden {args.hidden_size}, from scratch on FineWeb-Edu "
        f"(streamed), {args.max_steps} steps/round, batch {args.batch_size} x {args.block_size} tokens.\n"
        f"Goal metric 'val_loss' (lower is better) = cross-entropy on held-out blocks, every "
        f"{args.eval_steps} steps.\n"
        f"You control a SEPARATE base learning rate per part of the model: one knob per decoder "
        f"layer ('lr_block_0'..'lr_block_{args.n_layer - 1}'), plus 'lr_embed' (token embeddings), "
        f"'lr_head' (LM head), and 'lr_other' (final norm). "
        f"Applied LR = base x a shared warmup+cosine factor, so you set per-layer amplitude, not "
        f"the schedule shape. Uniform baseline: every knob = {args.lr}.\n"
        f"Each control point you also see a per-layer grad_norm. Use the loss trend, the grad "
        f"norms, and the prior-round history to decide which parts of the model to speed up or "
        f"slow down; keep a knob unchanged when you believe it is already working."
    )


def build_model(args):
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_hidden_layers = args.n_layer
    lt = getattr(cfg, "layer_types", None)  # Qwen3 stores a per-layer list; resize to new depth
    if lt:
        cfg.layer_types = [lt[i % len(lt)] for i in range(args.n_layer)]
    cfg.hidden_size = args.hidden_size
    cfg.intermediate_size = args.intermediate_size
    cfg.num_attention_heads = args.n_head
    cfg.num_key_value_heads = args.n_kv_head
    if hasattr(cfg, "head_dim"):
        cfg.head_dim = args.hidden_size // args.n_head
    cfg.use_cache = False
    cfg.tie_word_embeddings = False
    # params stay fp32 (optimizer stability); autocast feeds bf16 activations to the attn kernel.
    return AutoModelForCausalLM.from_config(cfg, attn_implementation=args.attn_impl)


def main(args):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("layerwise_lr_gpt", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "layerwise_lr_gpt")
    memory_path = args.memory_path or os.path.join(run_dir, "layerwise_lr_gpt_memory.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data = FineWeb(tok, args.block_size, args.eval_docs)
    eval_loader = DataLoader(data.eval_blocks(args.eval_blocks),
                             batch_size=args.batch_size, collate_fn=FineWeb.collate)

    block_names = [f"lr_block_{i}" for i in range(args.n_layer)] + ["lr_embed", "lr_head", "lr_other"]
    agent = None if args.no_agent else LayerwiseAgent(
        block_names, every=args.agent_every, name=args.agent_model, client=build_client(args))
    session = TrainingSession(goal="val_loss", memory=memory_path, agent=agent,
                              max_rounds=args.max_rounds, context=build_context(args),
                              seed=args.seed, watch_metrics=["val_loss"])
    use_wandb = not args.no_wandb
    wandb_group = f"{args.wandb_experiment}-{run_id}"

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
            else torch.autocast(device_type="cpu", enabled=False)

    @torch.no_grad()
    def eval_loss(model):
        model.eval()
        total, n = 0.0, 0
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                total += float(model(**batch).loss) * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
        model.train()
        return total / max(n, 1)

    def train_round(session, ctx):
        model = build_model(args).to(device)
        model.train()
        groups, names = build_param_groups(model)
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        base = register_block_lrs(session, names, args.lr)
        session.plan_round(ctx)

        loader = iter(DataLoader(data, batch_size=args.batch_size, collate_fn=FineWeb.collate))
        babysat = agent is not None
        run = None
        if use_wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, group=wandb_group,
                             name=f"{wandb_group}-round{ctx.round}", reinit=True,
                             config={"round": ctx.round, "babysat": babysat, "lr": args.lr,
                                     "n_layer": args.n_layer, "hidden_size": args.hidden_size,
                                     "max_steps": args.max_steps, "batch_size": args.batch_size,
                                     "block_size": args.block_size})

        for it in range(args.max_steps):
            apply_lr(groups, names, base, sched_factor(it, args.warmup, args.max_steps))
            batch = {k: v.to(device) for k, v in next(loader).items()}
            with autocast():
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorms = block_grad_norms(groups, names)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            metrics = {"loss": float(loss.detach())}
            metrics.update({f"base_lr/{name}": base[name] for name in names})
            metrics.update({f"gnorm/{name}": gnorms[name] for name in names})
            if (it + 1) % args.eval_steps == 0 or it + 1 == args.max_steps:
                metrics["val_loss"] = eval_loss(model)

            if run is not None:
                run.log({**metrics, **{f"lr/{name}": g["lr"] for g, name in zip(groups, names)}}, step=it)
            logger.info("round %d step %d | loss=%.3f%s", ctx.round, it, metrics["loss"],
                        f" val_loss={metrics['val_loss']:.3f}" if "val_loss" in metrics else "")

            if session.step(metrics, step=it, act=babysat).stop:
                break

        model.save_pretrained(f"{output_dir}/round_{ctx.round}")
        if run is not None:
            run.finish()

    session.run_rounds(train_round)
    best = session.memory.best(direction="min")
    baseline = next((r for r in session.memory.rounds if r["round"] == 0), None)
    if baseline:
        print(f"Baseline (round 0) val_loss={baseline['score']:.4f}")
    print(f"Best round {best['round']} val_loss={best['score']:.4f} (see {session.memory.metrics_path})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM-babysat per-layer learning rates for a deep LLM from scratch")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="config/tokenizer source (weights are re-init)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=3000, help="training steps per round")
    p.add_argument("--n-layer", type=int, default=24, help="decoder layers (deep on purpose)")
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-kv-head", type=int, default=4, help="GQA key/value heads")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--attn-impl", default="sdpa",
                   help="attention kernel. 'sdpa' is deterministic (honors core.determinism); "
                        "'flash_attention_2' is faster but has a non-deterministic backward, so "
                        "identical seeds/configs will NOT reproduce the same loss.")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=6e-4, help="uniform base LR (baseline; per-layer start)")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-docs", type=int, default=2000, help="docs reserved for held-out eval")
    p.add_argument("--eval-blocks", type=int, default=200, help="held-out blocks scored")
    p.add_argument("--agent-every", type=int, default=300, help="agent acts every N steps")
    p.add_argument("--no-agent", action="store_true", help="uniform-LR baseline (no babysitter)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="deprecated no-op: round 0 is always a no-agent baseline")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--memory-path", default=None)
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="layerwise_lr_gpt")
    p.add_argument("--no-wandb", action="store_true")
    main(p.parse_args())
