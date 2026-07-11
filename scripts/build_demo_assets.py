#!/usr/bin/env python3
"""Build static, truthful demo assets from committed round-memory ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/muon_gpt/seed42/muon_gpt_memory.jsonl"
OUTPUT_DIR = ROOT / "demo/assets/muon_paper_trace"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda row: int(row["round"]))


def main() -> None:
    rows = load_rows()
    running_best = None
    normalized = []
    for row in rows:
        score = float(row["score"])
        improved = running_best is None or score < running_best
        running_best = score if improved else running_best
        plan = row.get("plan") or {}
        normalized.append(
            {
                "round": int(row["round"]),
                "baseline": bool(row.get("baseline")),
                "score": score,
                "best_step": row.get("best_step"),
                "frontier_improvement": bool(improved and not row.get("baseline")),
                "running_best": running_best,
                "config": row.get("config") or {},
                "strategy": plan.get("strategy", ""),
                "planned_config": plan.get("config") or {},
                "actions": row.get("actions") or [],
                "reflection": row.get("reflection", ""),
                "agent_usage": row.get("agent_usage") or {},
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds_path = OUTPUT_DIR / "rounds.json"
    rounds_path.write_text(json.dumps(normalized, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "mode": "round_memory",
        "fidelity": "committed_round_memory",
        "title": "Muon–AdamW 3,000-Step Paper Trace",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "rounds_sha256": sha256(rounds_path),
        "rounds": len(rows),
        "baseline_rounds": 1,
        "steps_per_round": 3000,
        "baseline_score": float(rows[0]["score"]),
        "best_score": min(float(row["score"]) for row in rows),
        "notice": "This is the 11-round paper trace, not the separate five-round video run. It contains round-level memory and summarized actions, not per-step metric curves."
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
