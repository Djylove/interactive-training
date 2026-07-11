"""Multi-round LLM-babysat GRPO RL on Countdown with Qwen3-0.6B.

Countdown: given a few numbers and a target, write an arithmetic expression using
each number exactly once that equals the target. Instances are generated
procedurally (no dataset download). The babysitter acts as a curriculum teacher:
`difficulty` is a knob controlling which level rollout prompts are drawn from, while
`eval_acc` is always measured on held-out hardest-level instances. Fixed-hard training
stalls because zero-variance GRPO groups (identical rewards within a group) carry no
gradient.

Run (inside apptainer, after `source init.sh`):
    python -m examples.rlvr_grpo_countdown --max-rounds 4            # babysat teacher
    python -m examples.rlvr_grpo_countdown --no-agent --max-rounds 1 # fixed-hard baseline
    python -m examples.rlvr_grpo_countdown --no-agent --difficulty 0 --max-rounds 1  # fixed-easy
    python -m examples.rlvr_grpo_countdown --no-agent --ramp --max-rounds 1          # hand ramp

Compare best-round eval_acc across arms with the same --seed; repeat with 3 seeds
(--seed 1234 2345 3456) and report mean±std.
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import re
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # V1 engine core needs spawn, not fork
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (see core.determinism)

import torch
import torch.nn.functional as F
from tqdm import tqdm
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
from interactive_training.recipes._common import bind_dict_knob
from interactive_training.recipes.rlvr import rlvr

import logging
setup_logging()
logger = logging.getLogger(__name__)

SYSTEM = ("You are playing Countdown. Given a list of numbers and a target, use each "
          "number exactly once with the operators + - * / and parentheses to build an "
          "expression equal to the target. Write the final expression on its own line "
          "after '#### '.")
_EXPR_OK = re.compile(r"^[0-9+\-*/() ]+$")
# Gentle ladder: adjacent levels differ by at most one extra number and a modest range
# bump. The lower rungs are quickly learnable, but the top rungs (5-6 numbers in a wide
# range, large targets) push a 0.6B model into a big combinatorial search that fixed-hard
# GRPO stalls on -- that gap is what makes the curriculum worth climbing.
LEVELS = [
    dict(n_min=2, n_max=3, lo=1, hi=10, target_max=30),
    dict(n_min=3, n_max=3, lo=1, hi=20, target_max=60),
    dict(n_min=3, n_max=4, lo=1, hi=25, target_max=150),
    dict(n_min=4, n_max=4, lo=1, hi=30, target_max=300),
    dict(n_min=4, n_max=5, lo=1, hi=40, target_max=500),
    dict(n_min=5, n_max=6, lo=1, hi=50, target_max=800),
]
MAX_DIFFICULTY = float(len(LEVELS) - 1)  # difficulty knob ceiling; eval always uses LEVELS[-1]


def build_context(args) -> str:
    """Static run background handed to the babysitting agent so it can reason about the
    model/task/algorithm instead of treating the knobs as opaque numbers."""
    lvl_desc = "; ".join(
        f"L{i}: {s['n_min']}-{s['n_max']} numbers in [{s['lo']},{s['hi']}], target<={s['target_max']}"
        for i, s in enumerate(LEVELS))
    return (
        f"Task: Countdown arithmetic. Each prompt gives a list of integers and a target; the "
        f"model must emit one expression using each number exactly once (operators + - * /, "
        f"parentheses) equal to the target, written after '#### '. Targets are guaranteed "
        f"solvable. Rollouts run in the model's thinking mode, so responses carry a "
        f"chain-of-thought before the answer and resp_len counts those tokens (cap "
        f"{args.max_resp_len}).\n"
        f"Difficulty levels: {lvl_desc}. Rollout prompts are drawn at the level set by the "
        f"'difficulty' knob (see its description). Every {args.eval_steps} iterations the policy "
        f"is scored on a held-out pool for EVERY level, logged as 'eval_acc_L0'..'eval_acc_L"
        f"{int(MAX_DIFFICULTY)}' ({args.eval_n_per_level} instances each, {args.eval_n} for the "
        f"hardest), plus 'eval_acc_mean' (their average). The goal metric 'eval_acc' is exact-match "
        f"accuracy on the {args.eval_n} held-out level-{int(MAX_DIFFICULTY)} (hardest) instances; "
        f"the per-level scores show how far up the curriculum the policy has climbed. Evaluation "
        f"levels never change.\n"
        f"Model: {args.model} (small, fast) trained with a LoRA adapter (rank {args.lora_rank}); "
        f"the frozen base acts as the GRPO reference policy.\n"
        f"Algorithm: GRPO (PPO-style clip + KL-to-reference) with vLLM rollout. Per iteration: "
        f"{args.n_prompts} prompts x {args.group_size} sampled completions, group-relative "
        f"advantages, then {args.ppo_epochs} update epoch(s) (so clip_range bites). "
        f"{args.max_steps} rollout iterations per round. 'zero_adv_frac' is the fraction of prompt "
        f"groups in an iteration whose {args.group_size} sampled completions all received "
        f"identical reward; such groups have zero group-relative advantage and contribute no "
        f"gradient.\n"
        f"Reward: staircase 0.0 (no valid expression) -> 0.1 (valid arithmetic, wrong "
        f"numbers/value) -> 1.1 (uses each number once and equals target).\n"
        f"Knob guidance (semantics are in each knob's description): 'lr' has no scheduler "
        f"(applied as-is each step); too-high 'kl_coef' stalls learning, too-low risks reward "
        f"hacking / KL blow-ups; keep clip_range_high >= clip_range. Watch for KL spikes and "
        f"collapsing entropy/response length, which precede eval_acc plateaus or regressions; "
        f"favor early-stopping (the 'stop' action) once eval_acc plateaus rather than training "
        f"to the end."
    )


def _random_tree_value(rng: random.Random, nums: list[int]) -> int:
    """Fold the numbers together in a random *binary tree* (not left-to-right) using
    + - *, so the resulting target generally needs operator precedence / parentheses to
    hit -- harder than a simple sequential fold. Guaranteed solvable: the tree itself is a
    valid expression using each number exactly once."""
    nodes = list(nums)
    while len(nodes) > 1:
        a = nodes.pop(rng.randrange(len(nodes)))
        b = nodes.pop(rng.randrange(len(nodes)))
        op = rng.choice("+-*")
        nodes.append(a + b if op == "+" else (a - b if op == "-" else a * b))
    return nodes[0]


def gen_instance(rng: random.Random, n_min=4, n_max=4, lo=1, hi=30,
                 target_max=300) -> tuple[list[int], int]:
    """Sample numbers and a guaranteed-solvable positive-integer target, built from a
    random +-* expression tree of the numbers (the solver may use / and parentheses too).
    Defaults match the hardest level (LEVELS[-1])."""
    while True:
        nums = [rng.randint(lo, hi) for _ in range(rng.randint(n_min, n_max))]
        val = _random_tree_value(rng, nums)
        if 0 < val <= target_max:
            return nums, val


def analyze(text: str, numbers: list[int]) -> tuple[bool, bool, float | None]:
    """(parseable, numbers_ok, value) for the '#### expr' tail; value None if eval fails."""
    seg = text.split("####")[-1].strip()
    expr = seg.splitlines()[0].strip() if seg else ""
    if not expr or not _EXPR_OK.match(expr) or "**" in expr:
        return False, False, None
    numbers_ok = sorted(int(x) for x in re.findall(r"\d+", expr)) == sorted(numbers)
    try:
        val = eval(expr, {"__builtins__": {}})  # safe: chars restricted to digits/ops/parens
    except SyntaxError:
        return False, False, None  # e.g. '+++': passes the char filter but isn't arithmetic
    except Exception:
        return True, numbers_ok, None  # valid arithmetic that fails to evaluate (e.g. 1/0)
    if not isinstance(val, (int, float)):
        return True, numbers_ok, None
    try:
        return True, numbers_ok, float(val)  # huge int products can overflow float
    except OverflowError:
        return True, numbers_ok, None


def reward_of(text: str, numbers: list[int], target: int) -> tuple[float, float]:
    """Return (shaped_reward, correct): 0 (no valid expression) -> 0.1 (valid arithmetic)
    -> 1.1 (uses each number once and equals target). The staircase gives early signal."""
    parseable, numbers_ok, val = analyze(text, numbers)
    if not parseable:
        return 0.0, 0.0
    if not numbers_ok or val is None:
        return 0.1, 0.0
    correct = 1.0 if abs(val - target) < 1e-6 else 0.0
    return correct + 0.1, correct


def prompt_ids(tokenizer, numbers: list[int], target: int, max_len: int) -> list[int]:
    user = f"Numbers: {', '.join(map(str, numbers))}\nTarget: {target}"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    enc = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=True, return_dict=True)
    return enc["input_ids"][-max_len:]


def _instance_key(nums: list[int], tgt: int) -> tuple:
    """Canonical key (number-multiset + target) so eval can be held out even against
    reorderings of the same problem."""
    return (tuple(sorted(nums)), tgt)


def make_pool(tokenizer, k: int, seed: int, max_len: int,
              exclude: set | None = None, spec: dict | None = None,
              desc: str | None = None, unique: bool = True) -> list[tuple[list[int], list[int], int]]:
    """Build k Countdown instances. Any key in `exclude` is skipped so the eval pool can be
    generated disjoint from train (a true held-out set). With `unique=True` the k instances
    are distinct; low levels (e.g. L0) have a tiny instance space that can be smaller than k,
    so train pools pass `unique=False` (rollout samples with replacement anyway)."""
    rng = random.Random(seed)
    exclude = set(exclude or ())
    seen, pool = set(), []
    with tqdm(total=k, desc=desc or "make_pool", unit="inst", dynamic_ncols=True) as pbar:
        while len(pool) < k:
            nums, tgt = gen_instance(rng, **(spec or {}))
            key = _instance_key(nums, tgt)
            if key in exclude or (unique and key in seen):
                continue
            seen.add(key)
            pool.append((prompt_ids(tokenizer, nums, tgt, max_len), nums, tgt))
            pbar.update(1)
    return pool


def _apply_liger_kernel(model_name: str) -> bool:
    try:
        transformers_mod = importlib.import_module("liger_kernel.transformers")
        apply_liger_kernel_to_qwen3 = getattr(transformers_mod, "apply_liger_kernel_to_qwen3")
    except ImportError:
        logger.warning("--use-liger requested but liger-kernel is not installed; continuing without it")
        return False
    except Exception as exc:
        logger.warning("could not import Liger Qwen3 kernel patch: %s; continuing without it", exc)
        return False

    if "qwen3" not in model_name.lower():
        logger.warning("--use-liger currently only patches Qwen3 here; model=%s", model_name)
        return False
    apply_liger_kernel_to_qwen3()
    logger.info("enabled Liger kernels for Qwen3")
    return True


def _causal_lm_parts(model):
    """Return the wrapped CausalLM, decoder, and lm_head for PEFT or plain HF models."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
        return None
    return causal_lm, causal_lm.model, causal_lm.lm_head


def _chunked_lm_head_logp(hidden, targets, lm_head, with_entropy: bool, chunk_tokens: int):
    # GRPO needs unreduced per-token log-probs for ratios and KL. Liger's fused
    # linear CE is memory efficient for scalar CE, but is not a drop-in replacement here.
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    logps, ents = [], []
    chunk_tokens = max(int(chunk_tokens), 1)
    for start in range(0, flat_hidden.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, flat_hidden.shape[0])
        # upcast to fp32 before softmax/CE: bf16 logits over a ~150k vocab make the
        # per-token logprob noisy, and that noise is amplified by exp(new - old).
        logits = F.linear(flat_hidden[start:end], lm_head.weight, lm_head.bias).float()
        logps.append(-F.cross_entropy(logits, flat_targets[start:end], reduction="none"))
        if with_entropy:
            lsm = torch.log_softmax(logits, dim=-1)
            ents.append(-(lsm.exp() * lsm).sum(-1))
    logp = torch.cat(logps).view_as(targets)
    if with_entropy:
        return logp, torch.cat(ents).view_as(targets)
    return logp


def logp_of(model, input_ids, attn, with_entropy: bool = False, logits_chunk_tokens: int = 1024):
    """Per-token log-prob of each next token (B, L-1); optionally also entropy.

    Entropy requires a full-vocab log-softmax tensor, so callers should request it only for
    metrics or when an entropy bonus is active.
    """
    parts = _causal_lm_parts(model)
    if parts is not None and logits_chunk_tokens > 0:
        _, decoder, lm_head = parts
        hidden = decoder(input_ids=input_ids, attention_mask=attn, use_cache=False,
                         return_dict=True).last_hidden_state[:, :-1, :]
        return _chunked_lm_head_logp(hidden, input_ids[:, 1:], lm_head, with_entropy,
                                     logits_chunk_tokens)

    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = -F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    if with_entropy:
        lsm = torch.log_softmax(logits, dim=-1)
        return logp, -(lsm.exp() * lsm).sum(-1)
    return logp


def make_microbatch(chunk, pad_id, device):
    """Pad a chunk of sequences and return (input_ids, attn, completion-token mask, adv)."""
    maxlen = max(len(ids) for ids, _, _ in chunk)
    n = len(chunk)
    input_ids = torch.full((n, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((n, maxlen), dtype=torch.long)
    cmask = torch.zeros((n, maxlen), dtype=torch.float32)
    adv = torch.zeros((n, 1), dtype=torch.float32)
    for j, (ids, start, a) in enumerate(chunk):
        L = len(ids)
        input_ids[j, :L] = torch.tensor(ids)
        attn[j, :L] = 1
        cmask[j, start:L] = 1.0
        adv[j, 0] = a
    mask = (cmask[:, 1:] * attn[:, 1:].float()).to(device)  # align to next-token targets
    return input_ids.to(device), attn.to(device), mask, adv.to(device)


@torch.no_grad()
def evaluate(llm, eval_items, lora_req, max_resp, seed: int = DEFAULT_SEED) -> float:
    # default seed keeps imported callers (examples/rlvr_reward_hack.py) working
    # Qwen3 thinking mode degenerates under greedy decoding; use the recommended
    # sampling (T=0.6, top_p=0.95, top_k=20) with a fixed seed so evals stay comparable.
    sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20,
                        max_tokens=max_resp, seed=seed)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _, _ in eval_items]
    outs = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
    correct = sum(reward_of(o.outputs[0].text, nums, tgt)[1]
                  for o, (_, nums, tgt) in zip(outs, eval_items))
    return correct / max(len(eval_items), 1)


def main(args):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir, _ = setup_logs("rlvr_grpo_countdown", run_id)
    output_dir = args.output_dir or os.path.join(run_dir, "countdown_grpo")
    memory_path = args.memory_path or os.path.join(run_dir, "countdown_grpo_memory.jsonl")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    train_pools = [make_pool(tokenizer, args.pool_per_level, seed=args.seed + 101 * lv,
                             max_len=args.max_prompt_len, spec=spec, desc=f"train pool L{lv}",
                             unique=False)
                   for lv, spec in enumerate(LEVELS)]
    train_keys = {_instance_key(nums, tgt) for pool in train_pools for _, nums, tgt in pool}
    # One held-out eval pool per difficulty level so we can watch the whole curriculum, not
    # just the top rung. The hardest level (the goal) gets the full args.eval_n instances;
    # the lower rungs get a cheaper args.eval_n_per_level sample (they are diagnostics only).
    eval_pools = [
        make_pool(tokenizer, args.eval_n if lv == len(LEVELS) - 1 else args.eval_n_per_level,
                  seed=args.seed + 7 * lv, max_len=args.max_prompt_len,
                  exclude=train_keys, spec=spec, desc=f"eval pool L{lv}")
        for lv, spec in enumerate(LEVELS)]

    # CUDA graphs (enforce_eager=False) matter a lot here: a 0.6B model decoding ~128
    # sequences is launch-overhead-bound, and graphs survive sleep/wake cycles.
    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=True, max_lora_rank=args.lora_rank,
              max_loras=1, gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_prompt_len + args.max_resp_len,
              enforce_eager=args.enforce_eager, enable_sleep_mode=True)

    agent = None if args.no_agent else LLMAgent(every=args.agent_every, name=args.agent_model,
                                                client=build_client(args))
    session = TrainingSession(goal=Accuracy("eval_acc"), memory=memory_path,
                              agent=agent, max_rounds=args.max_rounds, seed=args.seed,
                              context=build_context(args),
                              watch_metrics=["train_acc", "zero_adv_frac", "difficulty", "resp_len",
                                             "eval_acc_mean"])
    use_wandb = not args.no_wandb
    wandb_group = f"{args.wandb_experiment}-{run_id}"

    def train_round(session, ctx):
        if args.use_liger:
            _apply_liger_kernel(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation=args.attn_impl
        ).to(torch.bfloat16).to(device)
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        policy = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        policy.train()
        params = [p for p in policy.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=args.lr)
        policy.to("cpu")  # rollout runs first with vLLM holding the GPU

        cfg = {"kl_coef": args.kl_coef, "temperature": args.temperature, "clip_range": args.clip_range,
               "clip_range_high": args.clip_range_high, "grad_clip": args.grad_clip, "entropy_coef": 0.0,
               "difficulty": args.difficulty}
        rlvr(session, cfg, optim=optim)  # registers lr / kl_coef / temperature / clip_range knobs
        bind_dict_knob(session, cfg, "clip_range_high", min=0.0,
                       description="DAPO clip-higher: upper PPO/GRPO clip (eps_high), decoupled from "
                                   "clip_range (eps_low) to let low-prob tokens grow and curb entropy collapse")
        bind_dict_knob(session, cfg, "grad_clip", min=0.0,
                       description="gradient clipping max-norm (0 disables clipping)")
        bind_dict_knob(session, cfg, "entropy_coef", min=0.0, description="entropy bonus weight (exploration)")
        bind_dict_knob(session, cfg, "difficulty", min=0.0, max=MAX_DIFFICULTY,
                       description="training-instance difficulty level for rollout prompts: level = "
                                   "floor(value), +1 with prob frac(value); evaluation is always "
                                   f"level {int(MAX_DIFFICULTY)}")
        session.bind_model(policy)
        session.plan_round(ctx)  # rounds 1+: seed the planned config (round 0 keeps default)

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
                                     "difficulty": args.difficulty, "ramp": args.ramp})

        # dedicated RNG so prompt draws are reproducible and decoupled from model-op RNG
        sampler = torch.Generator().manual_seed(args.seed)
        for it in range(args.max_steps):
            if args.ramp and agent is None:
                cfg["difficulty"] = min(MAX_DIFFICULTY,
                                        MAX_DIFFICULTY * it / max(1, int(args.max_steps * 0.6)))
            d = min(max(float(cfg["difficulty"]), 0.0), MAX_DIFFICULTY)
            base = min(int(d), len(LEVELS) - 1)
            frac = d - base
            lvls = [min(base + int(torch.rand((), generator=sampler).item() < frac), len(LEVELS) - 1)
                    for _ in range(args.n_prompts)]
            picks = [train_pools[lv][int(torch.randint(0, len(train_pools[lv]), (1,), generator=sampler))]
                     for lv in lvls]
            prompts = [TokensPrompt(prompt_token_ids=pids) for pids, _, _ in picks]
            sp = SamplingParams(n=args.group_size, temperature=float(cfg["temperature"]),
                                top_p=1.0, max_tokens=args.max_resp_len, seed=args.seed + it)
            # per-step seed keeps vLLM's sampler reproducible (the global seed can't reach it)
            outs = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)

            seqs, rewards, corrects, lengths, zero_groups = [], [], [], [], 0
            for (pids, nums, tgt), out in zip(picks, outs):
                rs = [reward_of(c.text, nums, tgt) for c in out.outputs]
                r = torch.tensor([x[0] for x in rs])
                rewards += [x[0] for x in rs]
                corrects += [x[1] for x in rs]
                lengths += [len(c.token_ids) for c in out.outputs]
                # zero-variance groups have zero group-relative advantage: they carry no
                # policy-gradient signal, so exclude them from the update (otherwise they
                # only contribute KL steps that drag the policy back toward the reference).
                if float(r.std()) < 1e-8:
                    zero_groups += 1
                    continue
                adv = (r - r.mean()) / (r.std() + 1e-6)
                for c, a in zip(out.outputs, adv.tolist()):
                    seqs.append((pids + list(c.token_ids), len(pids), a))

            metrics = {"reward": sum(rewards) / len(rewards), "train_acc": sum(corrects) / len(corrects),
                       "resp_len": sum(lengths) / len(lengths), "lr": optim.param_groups[0]["lr"],
                       "zero_adv_frac": zero_groups / len(picks), "difficulty": d}

            # every group zero-variance -> no gradient signal at all: skip the whole
            # sleep/train/save cycle (the adapter is unchanged) and omit the update
            # diagnostics rather than logging a misleading kl=0 / entropy=0.
            if seqs:
                llm.sleep(level=1)
                policy.to(device)

                seqs.sort(key=lambda s: len(s[0]))  # length-bucketed micro-batches cut padding waste
                mbs = [make_microbatch(seqs[k:k + args.micro_bs], pad_id, device)
                       for k in range(0, len(seqs), args.micro_bs)]

                # frozen reference logprobs + behavior-policy ("old") logprobs, computed once.
                old_lp, ref_lp = [], []
                with torch.no_grad():
                    for input_ids, attn, mask, _ in mbs:
                        old_lp.append(logp_of(policy, input_ids, attn,
                                              logits_chunk_tokens=args.logits_chunk_tokens))
                        with policy.disable_adapter():
                            ref_lp.append(logp_of(policy, input_ids, attn,
                                                  logits_chunk_tokens=args.logits_chunk_tokens))

                klc = float(cfg["kl_coef"])
                eps_lo, eps_hi = float(cfg["clip_range"]), float(cfg["clip_range_high"])
                entc, gc = float(cfg["entropy_coef"]), float(cfg["grad_clip"])
                train_tok = sum(float(mask.sum()) for _, _, mask, _ in mbs)
                mb_rng = random.Random(args.seed + it)  # reproducible per-step microbatch shuffle
                last_loss = gnorm = 0.0
                for _ in range(args.ppo_epochs):
                    order = list(range(len(mbs)))
                    mb_rng.shuffle(order)  # break the fixed length-sorted order across epochs
                    optim.zero_grad(set_to_none=True)
                    epoch_loss = 0.0
                    for idx in order:
                        (input_ids, attn, mask, adv), old, ref = mbs[idx], old_lp[idx], ref_lp[idx]
                        if entc > 0.0:
                            new, ent = logp_of(policy, input_ids, attn, with_entropy=True,
                                               logits_chunk_tokens=args.logits_chunk_tokens)
                        else:
                            new = logp_of(policy, input_ids, attn,
                                          logits_chunk_tokens=args.logits_chunk_tokens)
                            ent = torch.zeros_like(new)
                        # clamp the log-ratio / KL delta before exp() so a single collapsed
                        # token can't overflow the loss and hijack the update direction.
                        ratio = torch.exp((new - old).clamp(-20.0, 20.0))
                        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * adv)
                        kd = (ref - new).clamp(-20.0, 20.0)
                        kl = torch.exp(kd) - kd - 1.0
                        # accumulate gradients over the whole rollout with a global token-mean
                        # normalizer, then take one optimizer step per epoch (effective batch =
                        # the full rollout, not a single length-bucketed micro-batch).
                        loss = ((pg + klc * kl - entc * ent) * mask).sum() / max(train_tok, 1.0)
                        loss.backward()
                        epoch_loss += float(loss.detach())
                    # gc <= 0 disables clipping (returns the norm without scaling grads to 0)
                    max_norm = gc if gc > 0.0 else float("inf")
                    gnorm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
                    optim.step()
                    last_loss = epoch_loss

                # post-update KL(policy||ref) + entropy so the logged diagnostics reflect the
                # policy after this step's updates (a pre-update measurement understates drift).
                kl_sum = ent_sum = tok_sum = 0.0
                with torch.no_grad():
                    for (input_ids, attn, mask, _), ref in zip(mbs, ref_lp):
                        lp, ent = logp_of(policy, input_ids, attn, with_entropy=True,
                                          logits_chunk_tokens=args.logits_chunk_tokens)
                        kd = (ref - lp).clamp(-20.0, 20.0)
                        kl_sum += float(((torch.exp(kd) - kd - 1.0) * mask).sum())
                        ent_sum += float((ent * mask).sum())
                        tok_sum += float(mask.sum())

                policy.save_pretrained(adapter_dir)  # sync trained weights -> vLLM via hot-swap
                policy.to("cpu")
                torch.cuda.empty_cache()
                llm.wake_up()
                version += 1
                lora_req = LoRARequest("policy", version, adapter_dir)

                metrics.update({"kl": kl_sum / max(tok_sum, 1.0),
                                "entropy": ent_sum / max(tok_sum, 1.0),
                                "loss": last_loss, "grad_norm": gnorm})
            if (it + 1) % args.eval_steps == 0 or it + 1 == args.max_steps:
                # Score every level's held-out pool: eval_acc_L{lv} exposes the whole
                # curriculum, eval_acc_mean averages across levels, and the goal metric
                # eval_acc is the hardest level (LEVELS[-1]) as before.
                lvl_accs = [evaluate(llm, pool, lora_req, args.max_resp_len, seed=args.seed)
                            for pool in eval_pools]
                for lv, a in enumerate(lvl_accs):
                    metrics[f"eval_acc_L{lv}"] = a
                metrics["eval_acc_mean"] = sum(lvl_accs) / len(lvl_accs)
                metrics["eval_acc"] = lvl_accs[-1]
            if run is not None:
                # also surface the live (agent-tunable) knob values so the babysitter's
                # adjustments are visible alongside the metrics in wandb.
                knob_log = {f"knob/{k}": float(cfg[k]) for k in cfg}
                knob_log["knob/lr"] = optim.param_groups[0]["lr"]
                run.log({**metrics, **knob_log}, step=it)

            nan = float("nan")
            eval_str = ""
            if "eval_acc" in metrics:
                per_lvl = " ".join(f"L{lv}={metrics[f'eval_acc_L{lv}']:.2f}" for lv in range(len(LEVELS)))
                eval_str = f" eval_acc={metrics['eval_acc']:.3f} (mean={metrics['eval_acc_mean']:.3f} | {per_lvl})"
            logger.info(
                "round %d step %d | reward=%.3f acc=%.3f kl=%.4f ent=%.3f len=%.0f loss=%.3f lr=%.1e%s",
                ctx.round, it, metrics["reward"], metrics["train_acc"], metrics.get("kl", nan),
                metrics.get("entropy", nan), metrics["resp_len"], metrics.get("loss", nan),
                metrics["lr"], eval_str)

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
    p = argparse.ArgumentParser(description="Multi-round LLM-babysat GRPO on Countdown (Qwen3-0.6B)")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="re-applied each round (data, weight init, rollout)")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100, help="rollout iterations per round")
    p.add_argument("--n-prompts", type=int, default=16, help="prompts sampled per iteration")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt (verl rollout.n)")
    p.add_argument("--ppo-epochs", type=int, default=2, help="update passes per rollout (makes clip_range matter)")
    p.add_argument("--micro-bs", type=int, default=16,
                   help="sequences per forward/backward micro-batch; raise only if GPU memory allows")
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--max-resp-len", type=int, default=2048,
                   help="completion token budget; thinking-mode CoT needs room before '#### '")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="HF training attention backend; flash_attention_2 is fastest but nondeterministic")
    p.add_argument("--use-liger", action=argparse.BooleanOptionalAction, default=True,
                   help="patch supported HF model kernels with liger-kernel when installed")
    p.add_argument("--logits-chunk-tokens", type=int, default=1024,
                   help="tokens per LM-head chunk for logprob/entropy; 0 falls back to full logits")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="trade compute for lower activation memory during PPO updates")
    p.add_argument("--lr", type=float, default=1e-6, help="verl actor_lr default (the baseline)")
    p.add_argument("--kl-coef", type=float, default=1e-3, help="verl kl_loss_coef default")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--clip-range", type=float, default=0.2, help="lower PPO/GRPO clip (DAPO eps_low)")
    p.add_argument("--clip-range-high", type=float, default=0.28,
                   help="DAPO 'clip-higher': upper PPO/GRPO clip (eps_high), decoupled from clip_range")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=500,
                   help="held-out instances scored on the hardest level for the goal metric")
    p.add_argument("--eval-n-per-level", type=int, default=200,
                   help="held-out instances scored per non-goal level for the eval_acc_L* diagnostics")
    p.add_argument("--difficulty", type=float, default=MAX_DIFFICULTY,
                   help=f"starting difficulty knob in [0,{int(MAX_DIFFICULTY)}]; "
                        f"{int(MAX_DIFFICULTY)} = hardest (the eval level)")
    p.add_argument("--pool-per-level", type=int, default=2000)
    p.add_argument("--ramp", action="store_true",
                   help=f"no-agent reference arm: linear difficulty ramp 0->{int(MAX_DIFFICULTY)} "
                        "over 60%% of steps")
    p.add_argument("--gpu-mem-util", type=float, default=0.85,
                   help="vLLM share; reclaimed for training each step via sleep + CPU offload")
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                   help="disable vLLM CUDA graphs; slower rollout but faster startup and "
                        "less GPU memory (the old behavior)")
    p.add_argument("--agent-every", type=int, default=10, help="agent acts every N rollout iterations")
    p.add_argument("--no-agent", action="store_true", help="pure GRPO baseline, no LLM babysitter (offline)")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--agent-model", default="gpt-5.5")
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--output-dir", default=None,
                   help="defaults to logs/rlvr_grpo_countdown/<run-id>/countdown_grpo")
    p.add_argument("--memory-path", default=None,
                   help="defaults to logs/rlvr_grpo_countdown/<run-id>/countdown_grpo_memory.jsonl")
    p.add_argument("--wandb-project", default="interactive_training_v2")
    p.add_argument("--wandb-experiment", default="rlvr_grpo_countdown")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    main(args)
