"""Multi-round LLM-babysat data mixing for GPT-2 small.

Continued pretraining of GPT-2 (124M) on a mixture of text domains (web / wiki /
code / math) streamed from HuggingFace. Each batch samples sequences per domain in
proportion to mixture-weight knobs (a simplex). The babysitter shifts the mixture
online toward lagging domains, since the compute-optimal mixture is non-stationary:
easy domains saturate early while harder ones still have headroom.

The goal is 'worst_excess' = max over domains of (val_loss_d - base_loss_d): the
least-improved domain's excess loss vs the initial GPT-2 (lower is better), so a
starved domain is penalized and the agent is pushed toward balanced improvement.

Babysitting is a minimal addition to a vanilla loop: register the knobs, then call
session.step(metrics) at each iteration. Pass --no-agent for the plain baseline.

Run (inside apptainer, after `source init.sh`):
    python -m examples.data_mixing_gpt2 --max-rounds 10            # babysat
    python -m examples.data_mixing_gpt2 --no-agent --max-rounds 1  # fixed-mixture baseline
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (see core.determinism)

import datasets
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from interactive_training.agents.agent import LLMAgent
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from examples._common import build_client, setup_logging
from examples._paths import setup_logs
from interactive_training.recipes._common import bind_dict_knob

setup_logging()
logger = logging.getLogger(__name__)

DOMAINS = {
    "web": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "text": "text"},
    "wiki": {"path": "wikimedia/wikipedia", "name": "20231101.en", "text": "text"},
    "code": {"path": "nick007x/github-code-2025", "text": "content"},
    "math": {"path": "HuggingFaceTB/finemath", "name": "finemath-4plus", "text": "text"},
}
NAMES = list(DOMAINS)


def build_context(args) -> str:
    # Mechanics only (action space, objective definition, cadence). Deliberately no
    # strategy guidance: prescribing a policy here (e.g. "upweight lagging domains")
    # leaks the intended behavior into the prompt and taints any comparison between
    # the babysitter and rule-based baselines. The agent must derive its policy from
    # the observed metrics and its own priors.
    return (
        f"Task: continued pretraining of GPT-2 small (124M) on a mixture of {len(NAMES)} "
        f"text domains ({', '.join(NAMES)}), streamed from HuggingFace. Every batch samples "
        f"each sequence from a domain in proportion to the mixture-weight knobs "
        f"'w_<domain>' (relative, >=0, renormalized to a simplex each step).\n"
        f"Per-domain excess loss = val_loss_d - base_loss_d, where base_loss_d is the "
        f"initial GPT-2's loss on that domain's held-out blocks (lower is better). "
        f"Goal metric 'objective' (lower is better) = "
        f"(1-{args.balance_weight})*mean_excess + {args.balance_weight}*worst_excess "
        f"(mean over domains blended with the max over domains). Measured every "
        f"{args.eval_steps} steps on held-out blocks per domain, reported as 'loss_<domain>'.\n"
        f"Algorithm: standard causal-LM cross-entropy with AdamW; only the data mixture and "
        f"the 'lr' knob (AdamW learning rate) are under control. {args.max_steps} steps per "
        f"round, batch {args.batch_size} x {args.seq_len} tokens."
    )


def load_stream(spec, take=None, skip=None):
    kw = {"path": spec["path"], "split": spec.get("split", "train"), "streaming": True}
    if spec.get("name"):
        kw["name"] = spec["name"]
    if spec.get("trust"):
        kw["trust_remote_code"] = True
    ds = datasets.load_dataset(**kw)
    if skip:
        ds = ds.skip(skip)
    if take:
        ds = ds.take(take)
    return ds


def iter_blocks(raw, text, tok, seq_len):
    buf, eos = [], tok.eos_token_id
    for ex in raw:
        t = ex.get(text)
        if not t:
            continue
        buf += tok(t, add_special_tokens=False).input_ids + [eos]
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def eval_blocks(spec, tok, seq_len, n_docs, n_blocks):
    out = []
    for b in iter_blocks(load_stream(spec, take=n_docs), spec["text"], tok, seq_len):
        out.append(b)
        if len(out) >= n_blocks:
            break
    return out


def train_gen(spec, tok, seq_len, skip):
    while True:  # restart the stream if exhausted (finite domains like wiki)
        yield from iter_blocks(load_stream(spec, skip=skip), spec["text"], tok, seq_len)


@torch.no_grad()
def eval_loss(model, blocks, batch_size, device):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        x = torch.tensor(blocks[i:i + batch_size], dtype=torch.long, device=device)
        total += model(input_ids=x, labels=x).loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(n, 1)


def sample_batch(gens, weights, batch_size, device):
    order = random.choices(range(len(NAMES)), weights=weights, k=batch_size)
    rows = [next(gens[NAMES[i]]) for i in order]
    return torch.tensor(rows, dtype=torch.long, device=device)


def load_model(name, device):
    kw = {"torch_dtype": torch.bfloat16}
    kw["attn_implementation"] = "flash_attention_2"
    return GPT2LMHeadModel.from_pretrained(name, **kw).to(device)


def main(args):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("data_mixing_gpt2", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "data_mixing_gpt2")
    memory_path = args.memory_path or os.path.join(run_dir, "data_mixing_gpt2_memory.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    evals = {d: eval_blocks(DOMAINS[d], tok, args.seq_len, args.eval_docs, args.eval_blocks)
             for d in NAMES}
    base_model = load_model(args.model, device)
    base = {d: eval_loss(base_model, evals[d], args.batch_size, device) for d in NAMES}
    del base_model
    torch.cuda.empty_cache()
    logger.info("base losses: %s", {d: round(v, 4) for d, v in base.items()})

    agent = None if args.no_agent else LLMAgent(every=args.agent_every, name=args.agent_model,
                                                client=build_client(args))
    session = TrainingSession(goal="objective", memory=memory_path,
                              agent=agent, max_rounds=args.max_rounds,
                              context=build_context(args), seed=args.seed,
                              watch_metrics=[f"loss_{d}" for d in NAMES] + ["mean_excess", "worst_excess"])
    use_wandb = not args.no_wandb
    wandb_group = f"{args.wandb_experiment}-{run_id}"

    def train_round(session, ctx):
        model = load_model(args.model, device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        cfg = {f"w_{d}": 1.0 for d in NAMES}

        session.register_optimizer_lr(opt)
        for d in NAMES:
            bind_dict_knob(session, cfg, f"w_{d}", min=0.0, max=1.0,
                           description=f"relative sampling weight for the {d} domain")
        session.bind_model(model)
        session.plan_round(ctx)

        gens = {d: train_gen(DOMAINS[d], tok, args.seq_len, args.eval_docs) for d in NAMES}
        babysat = agent is not None
        run = None
        if use_wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, group=wandb_group,
                             name=f"{wandb_group}-round{ctx.round}", reinit=True,
                             config={"round": ctx.round, "babysat": babysat, "lr": args.lr,
                                     "max_steps": args.max_steps, "batch_size": args.batch_size,
                                     "seq_len": args.seq_len})

        for it in range(args.max_steps):
            weights = [max(float(cfg[f"w_{d}"]), 0.0) for d in NAMES]
            if sum(weights) <= 0:
                weights = [1.0] * len(NAMES)
            x = sample_batch(gens, weights, args.batch_size, device)
            loss = model(input_ids=x, labels=x).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            metrics = {"loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"]}
            if (it + 1) % args.eval_steps == 0 or it + 1 == args.max_steps:
                excess = {}
                for d in NAMES:
                    ld = eval_loss(model, evals[d], args.batch_size, device)
                    metrics[f"loss_{d}"] = ld
                    excess[d] = ld - base[d]
                mean_excess = sum(excess.values()) / len(excess)
                worst_excess = max(excess.values())
                metrics["mean_loss"] = sum(metrics[f"loss_{d}"] for d in NAMES) / len(NAMES)
                metrics["mean_excess"] = mean_excess
                metrics["worst_excess"] = worst_excess
                # blended goal: overall improvement (mean) + worst-domain balance (alpha)
                metrics["objective"] = (1 - args.balance_weight) * mean_excess + args.balance_weight * worst_excess

            if run is not None:
                total_w = sum(weights)
                knob_log = {f"knob/w_{d}": weights[i] / total_w for i, d in enumerate(NAMES)}
                knob_log["knob/lr"] = opt.param_groups[0]["lr"]
                run.log({**metrics, **knob_log}, step=it)

            if "objective" in metrics:
                per_domain = " ".join(f"{d}={metrics[f'loss_{d}']:.3f}" for d in NAMES)
                tail = (f" | val[{per_domain}] overall={metrics['mean_loss']:.3f}"
                        f" mean_excess={metrics['mean_excess']:.3f}"
                        f" worst_excess={metrics['worst_excess']:.3f} obj={metrics['objective']:.3f}")
            else:
                tail = ""
            logger.info("round %d step %d | loss=%.3f lr=%.1e%s",
                        ctx.round, it, metrics["loss"], metrics["lr"], tail)

            if session.step(metrics, step=it, act=babysat).stop:
                break

        model.save_pretrained(f"{output_dir}/round_{ctx.round}")
        if run is not None:
            run.finish()

    session.run_rounds(train_round)
    best = session.memory.best(direction="min")
    baseline = next((r for r in session.memory.rounds if r["round"] == 0), None)
    if baseline:
        print(f"Baseline (round 0) objective={baseline['score']:.4f}")
    print(f"Best round {best['round']} objective={best['score']:.4f} (see {session.memory.metrics_path})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-round LLM-babysat data mixing (GPT-2 small)")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed (re-applied each round for a reproducible baseline)")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2000, help="training steps per round")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--balance-weight", type=float, default=0.5,
                   help="objective = (1-w)*mean_excess + w*worst_excess; 0=overall only, 1=balance only")
    p.add_argument("--eval-steps", type=int, default=200, help="evaluate per-domain loss every N steps")
    p.add_argument("--eval-docs", type=int, default=20000, help="docs reserved per domain for held-out eval")
    p.add_argument("--eval-blocks", type=int, default=256, help="held-out blocks scored per domain")
    p.add_argument("--agent-every", type=int, default=200, help="agent acts every N steps")
    p.add_argument("--no-agent", action="store_true", help="fixed uniform-mixture baseline, no babysitter")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/data_mixing_gpt2/<run-id>/data_mixing_gpt2")
    p.add_argument("--memory-path", default=None,
                   help="defaults to logs/data_mixing_gpt2/<run-id>/data_mixing_gpt2_memory.jsonl")
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="data_mixing_gpt2")
    p.add_argument("--no-wandb", action="store_true")
    main(p.parse_args())
