#!/usr/bin/env python3
"""Export reproducible LaTeX tables from the five committed memory ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_all_frontiers import REPO, RUNS
from plot_memory_scores import frontier_indices, load_memory


TRANSITIONS = {
    "Sentiment mixing": (
        (8, 9, 10),
        "Round 8 used a product-free, tweet-dominant mixture and regressed. Its "
        "reflection proposed balanced tweet/finance weights and asymmetric class "
        "weights; Round 9 records those choices explicitly.",
    ),
    "Layerwise GPT": (
        (2, 3, 8),
        "Round 2 raised many deeper-block rates toward 0.001 and regressed. Round 3 "
        "caps block rates at 0.0009 while increasing embedding and output-head rates; "
        "later plans preserve that separation.",
    ),
    "Muon--AdamW GPT": (
        (4, 5),
        "The Round 4 reflection identifies momentum as an untested control. The "
        "Round 5 plan lowers Muon momentum from 0.95 to 0.90, preceding the next "
        "recorded score.",
    ),
    "RLVR Countdown": (
        (5, 6, 7),
        "Round 5 starts at the hardest curriculum level and stalls. Its reflection "
        "proposes wider clipping and more early exploration; the Round 6 and 7 plans "
        "record those changes.",
    ),
}


def score_text(value: float) -> str:
    return f"{value:.5g}"


def load_all() -> dict[str, tuple[object, list[dict]]]:
    return {
        spec.setting: (spec, load_memory(REPO / spec.memory))
        for spec in RUNS
    }


def render_ledger(data: dict[str, tuple[object, list[dict]]]) -> str:
    rows = []
    totals = {
        "rounds": 0,
        "agent_rounds": 0,
        "improvements": 0,
        "actions": 0,
        "input": 0,
        "output": 0,
        "cost": 0.0,
    }
    for setting, (spec, rounds) in data.items():
        scores = [float(row["score"]) for row in rounds]
        agent_rounds = sum(not bool(row.get("baseline")) for row in rounds)
        improvements = len(frontier_indices(scores, direction=spec.direction))
        actions = sum(len(row.get("actions") or []) for row in rounds)
        usage = rounds[-1].get("agent_usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cost = float(usage.get("cost_usd", 0.0))
        rows.append(
            f"{setting} & {len(rounds)} & {agent_rounds} & {improvements} & "
            f"{actions} & {input_tokens / 1000:.1f}k / {output_tokens / 1000:.1f}k "
            f"& \\${cost:.2f} \\\\"
        )
        totals["rounds"] += len(rounds)
        totals["agent_rounds"] += agent_rounds
        totals["improvements"] += improvements
        totals["actions"] += actions
        totals["input"] += input_tokens
        totals["output"] += output_tokens
        totals["cost"] += cost

    rows.append("\\midrule")
    rows.append(
        f"Total & {totals['rounds']} & {totals['agent_rounds']} & "
        f"{totals['improvements']} & {totals['actions']} & "
        f"{totals['input'] / 1000:.1f}k / {totals['output'] / 1000:.1f}k "
        f"& \\${totals['cost']:.2f} \\\\"
    )
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
\textbf{{Setting}} & \textbf{{Rounds}} & \textbf{{LLM rounds}} &
\textbf{{New-best rounds}} & \textbf{{Actions}} &
\textbf{{Input/output tokens}} & \textbf{{Cumulative cost}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{\textbf{{Committed session journal.}}
Token and cost fields are cumulative at the end of each session, as recorded by the
runtime client. ``Actions'' counts summarized successful within-round setting and
control actions; it is not a measure of decision quality.}}
\label{{tab:memory-ledger}}
\end{{table*}}
"""


def render_transitions(data: dict[str, tuple[object, list[dict]]]) -> str:
    rows = []
    for setting, (round_ids, lesson) in TRANSITIONS.items():
        _, rounds = data[setting]
        by_round = {int(row["round"]): row for row in rounds}
        transition = " $\\rightarrow$ ".join(
            f"R{round_id}: {score_text(float(by_round[round_id]['score']))}"
            for round_id in round_ids
        )
        rows.append(f"{setting} & {transition} & {lesson} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{@{{}}l l Y@{{}}}}
\toprule
\textbf{{Setting}} & \textbf{{Recorded score transition}} &
\textbf{{Journal entry carried forward}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\caption{{\textbf{{Representative cross-round journal transitions.}}
These examples are selected from the committed JSONL ledgers to illustrate how
failed or plateaued rounds appear explicitly in the next plan.}}
\label{{tab:memory-transitions}}
\end{{table*}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "generated",
        help="Directory for generated_memory_ledger.tex and generated_memory_transitions.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all()
    outputs = {
        output_dir / "generated_memory_ledger.tex": render_ledger(data),
        output_dir / "generated_memory_transitions.tex": render_transitions(data),
    }
    for path, text in outputs.items():
        path.write_text(text)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
