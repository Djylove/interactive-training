#!/usr/bin/env python3
"""Regenerate all paper frontier panels and their provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

from plot_memory_scores import frontier_indices, load_memory, plot_frontier


REPO = Path(__file__).resolve().parents[1]
PAPER_REPO = REPO


@dataclass(frozen=True)
class RunSpec:
    setting: str
    memory: str
    output: str
    direction: str
    ylabel: str


RUNS = (
    RunSpec(
        "BERT/IMDB",
        "logs/hf_bert_imdb_multiround/seed42/bert_imdb_memory.jsonl",
        "imdb_multi_round_frontier.png",
        "min",
        "Validation loss",
    ),
    RunSpec(
        "Sentiment mixing",
        "logs/data_mixing_sentiment/babysat_seed42/sentiment_memory.jsonl",
        "sentiment_frontier.png",
        "max",
        "Yelp macro-F1",
    ),
    RunSpec(
        "Layerwise GPT",
        "logs/layerwise_lr_gpt/seed42/layerwise_lr_gpt_memory.jsonl",
        "layerwise_lr_gpt_frontier.png",
        "min",
        "Validation loss",
    ),
    RunSpec(
        "Muon--AdamW GPT",
        "logs/muon_gpt/seed42/muon_gpt_memory.jsonl",
        "muon_gpt_frontier.png",
        "min",
        "Validation loss",
    ),
    RunSpec(
        "RLVR Countdown",
        "logs/rlvr_grpo_countdown/seed42/countdown_memory.jsonl",
        "countdown_frontier.png",
        "max",
        "Hardest-level accuracy",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def summarize(spec: RunSpec, memory: Path, output: Path, rounds: list[dict]) -> dict:
    scores = [float(row["score"]) for row in rounds]
    pick = min if spec.direction == "min" else max
    best_row = pick(rounds, key=lambda row: float(row["score"]))
    final_usage = rounds[-1].get("agent_usage") or {}
    return {
        "setting": spec.setting,
        "memory": str(memory.relative_to(REPO)),
        "memory_sha256": sha256(memory),
        "output": (
            str(output.relative_to(PAPER_REPO))
            if output.is_relative_to(PAPER_REPO)
            else str(Path("figures") / output.name)
        ),
        "output_sha256": sha256(output),
        "direction": spec.direction,
        "ylabel": spec.ylabel,
        "rounds": len(rounds),
        "agent_rounds": sum(not bool(row.get("baseline")) for row in rounds),
        "frontier_improvements_after_baseline": len(
            frontier_indices(scores, direction=spec.direction)
        ),
        "baseline_score": scores[0],
        "best_score": float(best_row["score"]),
        "best_round": int(best_row["round"]),
        "total_recorded_actions": sum(len(row.get("actions") or []) for row in rounds),
        "final_cumulative_agent_usage": final_usage,
        "command": (
            f"python scripts/plot_memory_scores.py {spec.memory} "
            f"-o figures/{spec.output} --direction {spec.direction} "
            f'--ylabel "{spec.ylabel}"'
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all five paper frontier panels from committed memory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "figures",
        help="Directory for panel PNGs and frontier_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in RUNS:
        memory = REPO / spec.memory
        output = output_dir / spec.output
        rounds = load_memory(memory)
        plot_frontier(
            rounds,
            output=output,
            title=None,
            ylabel=spec.ylabel,
            direction=spec.direction,
        )
        records.append(summarize(spec, memory, output, rounds))
        print(f"Wrote {output}")

    manifest = {
        "schema_version": 1,
        "source_repository": "https://github.com/yuntian-group/interactive-training",
        "source_commit": git_commit(),
        "matplotlib_version": matplotlib.__version__,
        "plotter": "scripts/plot_memory_scores.py",
        "runs": records,
        "totals": {
            "agent_rounds": sum(row["agent_rounds"] for row in records),
            "recorded_actions": sum(row["total_recorded_actions"] for row in records),
            "input_tokens": sum(
                row["final_cumulative_agent_usage"].get("input_tokens", 0)
                for row in records
            ),
            "output_tokens": sum(
                row["final_cumulative_agent_usage"].get("output_tokens", 0)
                for row in records
            ),
            "cost_usd": round(
                sum(
                    row["final_cumulative_agent_usage"].get("cost_usd", 0.0)
                    for row in records
                ),
                6,
            ),
        },
    }
    manifest_path = output_dir / "frontier_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
