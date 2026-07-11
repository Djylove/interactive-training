"""Multi-round LLM-babysat GRPO RL on Countdown with reward-weight tuning.

Countdown arithmetic with a shaped reward whose default weights are exploitable: the
policy can maximize format and length credits without solving instances. The babysitter
tunes w_format, w_numbers, and w_length round by round while eval_acc counts only
correctness.

Run (inside apptainer, after `source init.sh`):
    python -m examples.rlvr_reward_hack --max-rounds 5                 # babysat
    python -m examples.rlvr_reward_hack --no-agent --max-rounds 1      # hacky-default baseline
    python -m examples.rlvr_reward_hack --no-agent --max-rounds 1 \
        --w-format 0.05 --w-numbers 0.1 --w-length 0                   # hand-tuned reference

Compare best-round eval_acc across arms with the same --seed; repeat with 3 seeds and
report mean±std.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # V1 engine core needs spawn, not fork
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (see core.determinism)

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

from interactive_training.agents.agent import LLMAgent
from interactive_training.core import TrainingSession
from interactive_training.core.determinism import DEFAULT_SEED
from interactive_training.core.goals import Accuracy
from examples._common import build_client, setup_logging
from examples._paths import setup_logs
from examples.rlvr_grpo_countdown import (LEVELS, analyze, evaluate, logp_of,
                                          make_microbatch, make_pool, _instance_key)
from interactive_training.recipes._common import bind_dict_knob
from interactive_training.recipes.rlvr import rlvr

import logging
setup_logging()


def shaped_reward(text: str, numbers: list[int], target: int, n_tokens: int,
                  max_tokens: int, w: dict) -> tuple[float, float, float, float, float]:
    """(total, correct, r_format, r_numbers, r_length); eval uses only `correct`."""
    parseable, numbers_ok, val = analyze(text, numbers)
    correct = 1.0 if numbers_ok and val is not None and abs(val - target) < 1e-6 else 0.0
    r_f = float(w["w_format"]) if parseable else 0.0
    r_n = float(w["w_numbers"]) if numbers_ok else 0.0
    r_l = float(w["w_length"]) * min(n_tokens, max_tokens) / max_tokens
    return correct + r_f + r_n + r_l, correct, r_f, r_n, r_l


def build_context(args) -> str:
    s = LEVELS[args.level]
    return (
        f"Task: Countdown arithmetic. Each prompt gives {s['n_min']}-{s['n_max']} integers in "
        f"[{s['lo']},{s['hi']}] and a target up to {s['target_max']}; the model must emit one "
        f"expression using each number exactly once (operators + - * /, parentheses) equal to "
        f"the target, written after '#### '. Targets are guaranteed solvable.\n"
        f"Model: {args.model} (small, fast) trained with a LoRA adapter (rank {args.lora_rank}); "
        f"the frozen base acts as the GRPO reference policy.\n"
        f"Algorithm: GRPO (PPO-style clip + KL-to-reference) with vLLM rollout. Per iteration: "
        f"{args.n_prompts} prompts x {args.group_size} sampled completions, group-relative "
        f"advantages, then {args.ppo_epochs} update epoch(s) (so clip_range bites). "
        f"{args.max_steps} rollout iterations per round.\n"
        f"Reward per completion: 1.0 * correct + w_format * parseable + w_numbers * "
        f"uses_given_numbers + w_length * (completion_tokens / {args.max_resp_len}). 'correct' "
        f"means the expression uses each given number exactly once and equals the target; "
        f"'parseable' means the line after '#### ' is a well-formed arithmetic expression; "
        f"'uses_given_numbers' additionally requires the digits to match the given numbers. "
        f"w_format, w_numbers, w_length are knobs. Per-component means are logged as "
        f"'r_format', 'r_numbers', 'r_length'; 'train_acc' is the rollout mean of 'correct'. "
        f"The goal metric 'eval_acc' counts ONLY 'correct' (no shaping) on {args.eval_n} "
        f"held-out instances decoded greedily, measured every {args.eval_steps} iterations.\n"
        f"Knob guidance: 'lr' is applied directly each step (no scheduler). 'kl_coef' pulls the "
        f"policy toward the base model (too high stalls learning, too low risks reward hacking / "
        f"KL blow-ups). 'temperature' controls rollout exploration. The PPO update uses a DAPO "
        f"decoupled clip: 'clip_range' is the lower bound (eps_low) and 'clip_range_high' is the "
        f"upper bound (eps_high). Raising clip_range_high above clip_range ('clip-higher') gives "
        f"low-probability tokens more room to grow, boosting exploration and fighting entropy "
        f"collapse; keep eps_high >= eps_low. 'grad_clip' is the gradient max-norm. 'entropy_coef' "
        f"adds an exploration bonus. "
        f"Watch for KL spikes and collapsing entropy/response length, which precede eval_acc "
        f"plateaus or regressions; favor early-stopping (the 'stop' action) once eval_acc "
        f"plateaus rather than training to the end."
    )


def main(args):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("rlvr_reward_hack", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "countdown_reward_hack")
    memory_path = args.memory_path or os.path.join(run_dir, "countdown_reward_hack_memory.jsonl")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    train_items = make_pool(tokenizer, 3000, seed=args.seed, max_len=args.max_prompt_len,
                            spec=LEVELS[args.level])
    train_keys = {_instance_key(nums, tgt) for _, nums, tgt in train_items}
    eval_items = make_pool(tokenizer, args.eval_n, seed=args.seed, max_len=args.max_prompt_len,
                           exclude=train_keys, spec=LEVELS[args.level])

    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=True, max_lora_rank=args.lora_rank,
              max_loras=1, gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_prompt_len + args.max_resp_len, enforce_eager=True,
              enable_sleep_mode=True)

    agent = None if args.no_agent else LLMAgent(every=args.agent_every, name=args.agent_model,
                                                client=build_client(args))
    session = TrainingSession(goal=Accuracy("eval_acc"), memory=memory_path,
                              agent=agent, max_rounds=args.max_rounds, seed=args.seed,
                              context=build_context(args),
                              watch_metrics=["train_acc", "r_format", "r_numbers", "r_length", "resp_len"])
    use_wandb = not args.no_wandb
    wandb_group = f"{args.wandb_experiment}-{run_id}"

    def train_round(session, ctx):
        model = AutoModelForCausalLM.from_pretrained(args.model).to(torch.bfloat16).to(device)
        model.config.use_cache = False
        policy = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        policy.train()
        params = [p for p in policy.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=args.lr)
        policy.to("cpu")

        cfg = {"kl_coef": args.kl_coef, "temperature": args.temperature, "clip_range": args.clip_range,
               "clip_range_high": args.clip_range_high, "grad_clip": args.grad_clip, "entropy_coef": 0.0,
               "w_format": args.w_format, "w_numbers": args.w_numbers, "w_length": args.w_length}
        rlvr(session, cfg, optim=optim)
        bind_dict_knob(session, cfg, "clip_range_high", min=0.0,
                       description="DAPO clip-higher: upper PPO/GRPO clip (eps_high), decoupled from "
                                   "clip_range (eps_low) to let low-prob tokens grow and curb entropy collapse")
        bind_dict_knob(session, cfg, "grad_clip", min=0.0, description="gradient clipping max-norm")
        bind_dict_knob(session, cfg, "entropy_coef", min=0.0, description="entropy bonus weight (exploration)")
        for name in ("w_format", "w_numbers", "w_length"):
            bind_dict_knob(session, cfg, name, min=0.0, max=1.0,
                           description=f"reward weight '{name}' (see reward definition in the run background)")
        session.bind_model(policy)
        session.plan_round(ctx)

        adapter_dir = f"{output_dir}/round_{ctx.round}/adapter"
        babysat = agent is not None
        lora_req, version = None, 0
        run = None
        if use_wandb:
            import wandb
            run = wandb.init(project=args.wandb_project, group=wandb_group,
                             name=f"{wandb_group}-round{ctx.round}", reinit=True,
                             config={"round": ctx.round, "babysat": babysat, "model": args.model,
                                     "lr": args.lr, "kl_coef": args.kl_coef,
                                     "temperature": args.temperature, "clip_range": args.clip_range,
                                     "clip_range_high": args.clip_range_high,
                                     "grad_clip": args.grad_clip, "n_prompts": args.n_prompts,
                                     "group_size": args.group_size, "ppo_epochs": args.ppo_epochs,
                                     "max_steps": args.max_steps, "lora_rank": args.lora_rank,
                                     "level": args.level, "w_format": args.w_format,
                                     "w_numbers": args.w_numbers, "w_length": args.w_length})

        sampler = torch.Generator().manual_seed(args.seed)
        for it in range(args.max_steps):
            idx = torch.randint(0, len(train_items), (args.n_prompts,), generator=sampler).tolist()
            prompts = [TokensPrompt(prompt_token_ids=train_items[i][0]) for i in idx]
            sp = SamplingParams(n=args.group_size, temperature=float(cfg["temperature"]),
                                top_p=1.0, max_tokens=args.max_resp_len, seed=args.seed + it)
            outs = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)

            seqs, rewards, corrects, lengths = [], [], [], []
            sum_rf, sum_rn, sum_rl = 0.0, 0.0, 0.0
            for i, out in zip(idx, outs):
                _, nums, tgt = train_items[i]
                rs = [shaped_reward(c.text, nums, tgt, len(c.token_ids), args.max_resp_len, cfg)
                      for c in out.outputs]
                r = torch.tensor([x[0] for x in rs])
                adv = (r - r.mean()) / (r.std() + 1e-6)
                for c, a in zip(out.outputs, adv.tolist()):
                    seqs.append((train_items[i][0] + list(c.token_ids), len(train_items[i][0]), a))
                    lengths.append(len(c.token_ids))
                rewards += [x[0] for x in rs]
                corrects += [x[1] for x in rs]
                sum_rf += sum(x[2] for x in rs)
                sum_rn += sum(x[3] for x in rs)
                sum_rl += sum(x[4] for x in rs)

            llm.sleep(level=1)
            policy.to(device)

            seqs.sort(key=lambda s: len(s[0]))
            mbs = [make_microbatch(seqs[k:k + args.micro_bs], pad_id, device)
                   for k in range(0, len(seqs), args.micro_bs)]

            old_lp, ref_lp, kl_sum, ent_sum, tok_sum = [], [], 0.0, 0.0, 0.0
            with torch.no_grad():
                for input_ids, attn, mask, _ in mbs:
                    lp, ent = logp_of(policy, input_ids, attn, with_entropy=True)
                    old_lp.append(lp)
                    with policy.disable_adapter():
                        rlp = logp_of(policy, input_ids, attn)
                    ref_lp.append(rlp)
                    kd = rlp - lp
                    kl_sum += float(((torch.exp(kd) - kd - 1.0) * mask).sum())
                    ent_sum += float((ent * mask).sum())
                    tok_sum += float(mask.sum())

            klc = float(cfg["kl_coef"])
            eps_lo, eps_hi = float(cfg["clip_range"]), float(cfg["clip_range_high"])
            entc, gc = float(cfg["entropy_coef"]), float(cfg["grad_clip"])
            last_loss = gnorm = 0.0
            n_comp = len(rewards)
            for _ in range(args.ppo_epochs):
                for (input_ids, attn, mask, adv), old, ref in zip(mbs, old_lp, ref_lp):
                    new, ent = logp_of(policy, input_ids, attn, with_entropy=True)
                    ratio = torch.exp(new - old)
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * adv)
                    kd = ref - new
                    kl = torch.exp(kd) - kd - 1.0
                    loss = ((pg + klc * kl - entc * ent) * mask).sum() / mask.sum().clamp(min=1.0)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    gnorm = float(torch.nn.utils.clip_grad_norm_(params, gc))
                    optim.step()
                    last_loss = float(loss.detach())

            policy.save_pretrained(adapter_dir)
            policy.to("cpu")
            torch.cuda.empty_cache()
            llm.wake_up()
            version += 1
            lora_req = LoRARequest("policy", version, adapter_dir)

            metrics = {"reward": sum(rewards) / len(rewards), "train_acc": sum(corrects) / len(corrects),
                       "kl": kl_sum / max(tok_sum, 1.0), "entropy": ent_sum / max(tok_sum, 1.0),
                       "resp_len": sum(lengths) / len(lengths), "loss": last_loss,
                       "grad_norm": gnorm, "lr": optim.param_groups[0]["lr"],
                       "r_format": sum_rf / n_comp, "r_numbers": sum_rn / n_comp,
                       "r_length": sum_rl / n_comp}
            if (it + 1) % args.eval_steps == 0 or it + 1 == args.max_steps:
                metrics["eval_acc"] = evaluate(llm, eval_items, lora_req, args.max_resp_len)
            if run is not None:
                knob_log = {f"knob/{k}": float(cfg[k]) for k in cfg}
                knob_log["knob/lr"] = optim.param_groups[0]["lr"]
                run.log({**metrics, **knob_log}, step=it)

            logging.getLogger(__name__).info(
                "round %d step %d | reward=%.3f acc=%.3f kl=%.4f ent=%.3f len=%.0f loss=%.3f lr=%.1e%s",
                ctx.round, it, metrics["reward"], metrics["train_acc"], metrics["kl"],
                metrics["entropy"], metrics["resp_len"], metrics["loss"], metrics["lr"],
                f" eval_acc={metrics['eval_acc']:.3f}" if "eval_acc" in metrics else "")

            ctrl = session.step(metrics, step=it, act=babysat)
            if ctrl.stop:
                break

        if run is not None:
            run.finish()

    session.run_rounds(train_round)
    best = session.memory.best(direction="max")
    base = next((r for r in session.memory.rounds if r["round"] == 0), None)
    print(f"Baseline (round 0) eval_acc={base['score'] if base else float('nan'):.4f}")
    print(f"Best round {best['round']} eval_acc={best['score']:.4f} (see {session.memory.metrics_path})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM-babysat reward-weight tuning on Countdown GRPO")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="re-applied each round (data, weight init, rollout)")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200, help="rollout iterations per round")
    p.add_argument("--n-prompts", type=int, default=16, help="prompts sampled per iteration")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt (verl rollout.n)")
    p.add_argument("--ppo-epochs", type=int, default=2, help="update passes per rollout (makes clip_range matter)")
    p.add_argument("--micro-bs", type=int, default=16, help="sequences per forward/backward micro-batch")
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--max-resp-len", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-6, help="verl actor_lr default (the baseline)")
    p.add_argument("--kl-coef", type=float, default=1e-3, help="verl kl_loss_coef default")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--clip-range", type=float, default=0.2, help="lower PPO/GRPO clip (DAPO eps_low)")
    p.add_argument("--clip-range-high", type=float, default=0.28,
                   help="DAPO 'clip-higher': upper PPO/GRPO clip (eps_high), decoupled from clip_range")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=500, help="held-out instances scored for the goal")
    p.add_argument("--level", type=int, default=2, choices=range(len(LEVELS)))
    p.add_argument("--w-format", type=float, default=0.4)
    p.add_argument("--w-numbers", type=float, default=0.2)
    p.add_argument("--w-length", type=float, default=0.3)
    p.add_argument("--gpu-mem-util", type=float, default=0.85,
                   help="vLLM share; reclaimed for training each step via sleep + CPU offload")
    p.add_argument("--agent-every", type=int, default=10, help="agent acts every N rollout iterations")
    p.add_argument("--no-agent", action="store_true", help="pure GRPO baseline, no LLM babysitter (offline)")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/rlvr_reward_hack/<run-id>/countdown_reward_hack")
    p.add_argument("--memory-path", default=None,
                   help="defaults to logs/rlvr_reward_hack/<run-id>/countdown_reward_hack_memory.jsonl")
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="rlvr_reward_hack")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    main(args)
