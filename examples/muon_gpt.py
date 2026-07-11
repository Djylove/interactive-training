"""LLM-babysat Muon optimizer for a small LLM trained from scratch.

Demonstrates that an LLM babysitter tuning the Muon optimizer (orthogonalized-momentum
updates on hidden 2D matrices, AdamW on everything else) beats fixed defaults on a small
from-scratch Qwen pretraining run.

Run (inside apptainer, after `source init.sh`):
    python -m examples.muon_gpt --max-rounds 4                      # babysat Muon
    python -m examples.muon_gpt --no-agent --max-rounds 1           # fixed-default Muon
    python -m examples.muon_gpt --no-agent --momentum-warmup --max-rounds 1  # hand warmup
    python -m examples.muon_gpt --no-agent --optimizer adamw --max-rounds 1  # AdamW reference

Compare best-round val_loss across arms with the same --seed; repeat with 3 seeds
(--seed 1234 2345 3456) and report mean±std.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (see core.determinism)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from interactive_training.agents.agent import LLMAgent
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from examples._common import build_client, setup_logging
from examples._paths import setup_logs
from examples.layerwise_lr_gpt import FineWeb, build_model, sched_factor
from interactive_training.recipes._common import bind_dict_knob

setup_logging()
logger = logging.getLogger(__name__)


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate UV^T (orthogonalization) of G via quintic Newton-Schulz in bf16."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon with nesterov momentum and decoupled weight decay, for 2D params only.
    Update RMS is matched to AdamW's (~0.2) so lr transfers across the two groups."""

    def __init__(self, params, lr, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"])
                o = zeropower_via_newtonschulz5(g)
                o.mul_(0.2 * math.sqrt(max(p.size(0), p.size(1))))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o, alpha=-group["lr"])


def split_params(model):
    """Muon gets the 2D matrices inside decoder layers (attention/MLP projections);
    AdamW gets everything else (embeddings, LM head, norms, any 1D params)."""
    hidden = [p for p in model.model.layers.parameters() if p.ndim == 2]
    ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return hidden, rest


@torch.no_grad()
def mean_weight_rms(mats):
    return float(sum(p.pow(2).mean().sqrt() for p in mats) / len(mats))


def build_context(args) -> str:
    return (
        f"Task: train a modern LLM (Qwen arch: RoPE, RMSNorm, SwiGLU, GQA) with "
        f"{args.n_layer} decoder layers, hidden {args.hidden_size}, from scratch on "
        f"FineWeb-Edu (streamed), {args.max_steps} steps/round, batch {args.batch_size} x "
        f"{args.block_size} tokens.\n"
        f"Goal metric 'val_loss' (lower is better) = cross-entropy on held-out blocks, "
        f"every {args.eval_steps} steps.\n"
        f"Optimizer: a hybrid. 'Muon' updates the hidden 2D weight matrices inside the "
        f"decoder layers: nesterov momentum ('muon_momentum'), then the momentum matrix is "
        f"orthogonalized via Newton-Schulz before the update; update RMS is rescaled to "
        f"match AdamW's, so 'muon_lr' and 'adam_lr' are on the same scale. AdamW "
        f"(betas 0.9/0.95) updates everything else: embeddings, LM head, norms. "
        f"'weight_decay' is decoupled decay applied to the Muon matrices only. Applied LR "
        f"= base knob x a shared warmup({args.warmup} steps)+cosine factor; you set the "
        f"amplitudes, not the schedule shape. Defaults: muon_lr={args.lr}, "
        f"adam_lr={args.lr}, muon_momentum={args.muon_momentum}, "
        f"weight_decay={args.weight_decay}.\n"
        f"Each control point you also see 'grad_norm' (global grad norm before clipping "
        f"at {args.grad_clip}) and 'weight_rms' (mean RMS of the Muon-updated matrices). "
        f"Use the loss trend, these diagnostics, and the prior-round history to decide; "
        f"keep a knob unchanged when you believe it is already working."
    )


def main(args):
    if args.optimizer == "adamw" and not args.no_agent:
        raise SystemExit("--optimizer adamw is a reference arm; add --no-agent")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("muon_gpt", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "muon_gpt")
    memory_path = args.memory_path or os.path.join(run_dir, "muon_gpt_memory.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data = FineWeb(tok, args.block_size, args.eval_docs)
    eval_loader = DataLoader(data.eval_blocks(args.eval_blocks),
                             batch_size=args.batch_size, collate_fn=FineWeb.collate)

    agent = None if args.no_agent else LLMAgent(
        every=args.agent_every, name=args.agent_model, client=build_client(args))
    session = TrainingSession(goal="val_loss", memory=memory_path, agent=agent,
                              max_rounds=args.max_rounds, context=build_context(args),
                              seed=args.seed,
                              watch_metrics=["val_loss", "weight_rms", "grad_norm"])
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
        hidden, rest = split_params(model)

        if args.optimizer == "adamw":
            opt_muon = None
            opt_adam = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                         weight_decay=0.0)
            cfg = None
        else:
            opt_muon = Muon(hidden, lr=args.lr, momentum=args.muon_momentum,
                            weight_decay=args.weight_decay)
            opt_adam = torch.optim.AdamW(rest, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
            cfg = {"muon_lr": args.lr, "adam_lr": args.lr,
                   "muon_momentum": args.muon_momentum, "weight_decay": args.weight_decay}
            bind_dict_knob(session, cfg, "muon_lr", min=0.0, max=10 * args.lr,
                           description="base LR for the Muon group (hidden 2D matrices); applied "
                                       "LR = base x a shared warmup+cosine factor")
            bind_dict_knob(session, cfg, "adam_lr", min=0.0, max=10 * args.lr,
                           description="base LR for the AdamW group (embeddings, LM head, norms); "
                                       "applied LR = base x the same warmup+cosine factor")
            bind_dict_knob(session, cfg, "muon_momentum", min=0.5, max=0.99,
                           description="Muon nesterov momentum, applied directly each step")
            bind_dict_knob(session, cfg, "weight_decay", min=0.0, max=0.2,
                           description="decoupled weight decay on the Muon-updated matrices only")

        session.plan_round(ctx)

        loader = iter(DataLoader(data, batch_size=args.batch_size, collate_fn=FineWeb.collate))
        babysat = agent is not None
        run = None
        if use_wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, group=wandb_group,
                             name=f"{wandb_group}-round{ctx.round}", reinit=True,
                             config={"round": ctx.round, "babysat": babysat, "optimizer": args.optimizer,
                                     "momentum_warmup": args.momentum_warmup, "lr": args.lr,
                                     "muon_momentum": args.muon_momentum, "weight_decay": args.weight_decay,
                                     "n_layer": args.n_layer, "hidden_size": args.hidden_size,
                                     "intermediate_size": args.intermediate_size, "n_head": args.n_head,
                                     "n_kv_head": args.n_kv_head, "max_steps": args.max_steps,
                                     "batch_size": args.batch_size, "block_size": args.block_size})

        for it in range(args.max_steps):
            if args.momentum_warmup and agent is None and cfg is not None:
                cfg["muon_momentum"] = 0.85 + 0.10 * min(it / 300.0, 1.0)

            if cfg is not None:
                factor = sched_factor(it, args.warmup, args.max_steps)
                for g in opt_muon.param_groups:
                    g["lr"] = float(cfg["muon_lr"]) * factor
                    g["momentum"] = float(cfg["muon_momentum"])
                    g["weight_decay"] = float(cfg["weight_decay"])
                for g in opt_adam.param_groups:
                    g["lr"] = float(cfg["adam_lr"]) * factor
            else:
                factor = sched_factor(it, args.warmup, args.max_steps)
                for g in opt_adam.param_groups:
                    g["lr"] = args.lr * factor

            batch = {k: v.to(device) for k, v in next(loader).items()}
            with autocast():
                loss = model(**batch).loss
            opt_adam.zero_grad(set_to_none=True)
            if opt_muon is not None:
                opt_muon.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            if opt_muon is not None:
                opt_muon.step()
            opt_adam.step()

            metrics = {"loss": float(loss.detach()), "grad_norm": gnorm,
                       "weight_rms": mean_weight_rms(hidden)}
            if cfg is not None:
                metrics.update({"muon_lr": float(cfg["muon_lr"]), "adam_lr": float(cfg["adam_lr"]),
                                "muon_momentum": float(cfg["muon_momentum"]),
                                "weight_decay": float(cfg["weight_decay"])})
            if (it + 1) % args.eval_steps == 0 or it + 1 == args.max_steps:
                metrics["val_loss"] = eval_loss(model)

            if run is not None:
                run.log(metrics, step=it)
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
    p = argparse.ArgumentParser(description="LLM-babysat Muon optimizer for a small LLM from scratch")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="config/tokenizer source (weights are re-init)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=3000, help="training steps per round")
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--intermediate-size", type=int, default=2048)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-kv-head", type=int, default=4, help="GQA key/value heads")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--attn-impl", default="sdpa",
                   help="attention kernel. 'sdpa' is deterministic (honors core.determinism); "
                        "'flash_attention_2' is faster but has a non-deterministic backward, so "
                        "identical seeds/configs will NOT reproduce the same loss.")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=6e-4,
                   help="shared default base LR for both the Muon and AdamW groups")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="round-0 default for the Muon-group decay knob (vanilla Muon)")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"],
                   help="'adamw' is a no-knob reference arm; requires --no-agent")
    p.add_argument("--momentum-warmup", action="store_true",
                   help="no-agent reference arm: linear muon_momentum 0.85->0.95 over "
                        "the first 300 steps")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-docs", type=int, default=2000, help="docs reserved for held-out eval")
    p.add_argument("--eval-blocks", type=int, default=200, help="held-out blocks scored")
    p.add_argument("--agent-every", type=int, default=300, help="agent acts every N steps")
    p.add_argument("--no-agent", action="store_true", help="fixed-default Muon baseline (no babysitter)")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--memory-path", default=None)
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="muon_gpt")
    p.add_argument("--no-wandb", action="store_true")
    main(p.parse_args())
