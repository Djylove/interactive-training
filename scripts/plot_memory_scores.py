#!/usr/bin/env python3
"""Plot neutral multi-round control traces from session-memory JSONL files.

Example:
    python scripts/plot_memory_scores.py logs/.../memory.jsonl --direction min
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Blue / gray / white palette aligned with the interactive training UI.
BLUE = "#2563eb"
BLUE_DARK = "#1d4ed8"
GRAY_DISCARDED = "#c4c9d4"
GRAY_BASELINE = "#9ca3af"
GRAY_GRID = "#e5e7eb"
GRAY_TEXT = "#374151"
WHITE = "#ffffff"


def load_memory(path: Path) -> list[dict]:
    rounds: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rounds.append(json.loads(line))
    if not rounds:
        raise SystemExit(f"No rounds found in {path}")
    return sorted(rounds, key=lambda r: r["round"])


def frontier_indices(scores: list[float], *, direction: str) -> list[int]:
    """Return post-baseline indices where the running best strictly improves."""
    kept: list[int] = []
    if direction == "min":
        best = scores[0]
        for i, score in enumerate(scores[1:], start=1):
            if score < best:
                best = score
                kept.append(i)
    else:
        best = scores[0]
        for i, score in enumerate(scores[1:], start=1):
            if score > best:
                best = score
                kept.append(i)
    return kept


def _format_score(value: float) -> str:
    if value >= 1:
        return f"{value:.4f}"
    if value >= 0.1:
        return f"{value:.5f}"
    return f"{value:.6f}"


def running_best(scores: list[float], *, direction: str) -> list[float]:
    if direction == "min":
        best = scores[0]
        out = [best]
        for score in scores[1:]:
            best = min(best, score)
            out.append(best)
        return out
    best = scores[0]
    out = [best]
    for score in scores[1:]:
        best = max(best, score)
        out.append(best)
    return out


def _legend_location(*, direction: str) -> str:
    """Pick a legend corner that stays clear of the frontier curve."""
    if direction == "max":
        # Frontier improvements climb toward the top-right; anchor legend elsewhere.
        return "upper left"
    # For minimization, the running best ends near the bottom.
    return "upper right"


def plot_frontier(
    rounds: list[dict],
    *,
    output: Path,
    title: str | None,
    ylabel: str,
    direction: str,
) -> None:
    xs = np.array([r["round"] for r in rounds], dtype=float)
    ys = np.array([float(r["score"]) for r in rounds], dtype=float)

    kept_idx = set(frontier_indices(ys.tolist(), direction=direction))
    baseline = ys[0]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    # Baseline from round 0.
    ax.axhline(
        baseline,
        color=GRAY_BASELINE,
        linestyle="--",
        linewidth=1.5,
        zorder=1,
        label="Baseline (round 0)",
    )
    ax.scatter(
        [xs[0]],
        [ys[0]],
        marker="D",
        s=58,
        color=GRAY_BASELINE,
        edgecolors=WHITE,
        linewidths=1.0,
        zorder=4,
    )

    # Attempts after the baseline that did not push the frontier.
    discarded_x = [x for i, x in enumerate(xs) if i != 0 and i not in kept_idx]
    discarded_y = [y for i, y in enumerate(ys) if i != 0 and i not in kept_idx]
    if discarded_x:
        ax.scatter(
            discarded_x,
            discarded_y,
            s=36,
            color=GRAY_DISCARDED,
            alpha=0.85,
            edgecolors="none",
            zorder=2,
            label="Non-improving round",
        )

    # Frontier improvements.
    kept_x = [x for i, x in enumerate(xs) if i in kept_idx]
    kept_y = [y for i, y in enumerate(ys) if i in kept_idx]
    ax.scatter(
        kept_x,
        kept_y,
        s=90,
        color=BLUE,
        edgecolors=WHITE,
        linewidths=1.5,
        zorder=4,
        label="Frontier improvement",
    )

    for x, y, color in [(xs[0], ys[0], GRAY_TEXT)] + [
        (x, y, BLUE_DARK) for x, y in zip(kept_x, kept_y)
    ]:
        ax.annotate(
            _format_score(y),
            xy=(x, y),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=color,
            zorder=5,
        )

    # Step line for running best (frontier).
    best_curve = running_best(ys.tolist(), direction=direction)
    ax.step(
        xs,
        best_curve,
        where="post",
        color=BLUE_DARK,
        linewidth=2.5,
        zorder=3,
        label="Best observed so far",
    )

    n_rounds = len(rounds)
    n_improvements = len(kept_idx)
    if title is None:
        title = (
            f"Multi-round control trace: {n_rounds} rounds, "
            f"{n_improvements} frontier improvements"
        )

    better = "lower is better" if direction == "min" else "higher is better"
    ax.set_title(title, fontsize=15, fontweight="bold", color=GRAY_TEXT, pad=14)
    ax.set_xlabel("Round", fontsize=12, fontweight="bold", color=GRAY_TEXT)
    ax.set_ylabel(f"{ylabel} ({better})", fontsize=12, fontweight="bold", color=GRAY_TEXT)

    ax.set_xticks(xs)
    ax.tick_params(colors=GRAY_TEXT)
    ax.grid(True, axis="both", color=GRAY_GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY_GRID)
    ax.spines["bottom"].set_color(GRAY_GRID)

    ymin, ymax = float(ys.min()), float(ys.max())
    ypad = max((ymax - ymin) * 0.12, 0.008)
    if direction == "max":
        # Leave extra headroom above the final kept point and its label.
        ypad = max(ypad, ymax * 0.08)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlim(xs.min() - 0.4, xs.max() + 0.4)

    ax.legend(
        loc=_legend_location(direction=direction),
        frameon=True,
        facecolor=WHITE,
        edgecolor=GRAY_GRID,
        fontsize=10,
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot autoresearch-style frontier figures from memory JSONL files."
    )
    p.add_argument("memory", type=Path, help="Path to memory JSONL file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: figures/<stem>_frontier.png)",
    )
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--ylabel", default="Score", help="Y-axis label")
    p.add_argument(
        "--direction",
        choices=("min", "max"),
        default="min",
        help="Whether lower or higher score is better (default: min)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    memory_path = args.memory.resolve()
    if not memory_path.exists():
        raise SystemExit(f"Memory file not found: {memory_path}")

    stem = memory_path.stem.replace("_results", "").replace("_memory", "")
    output = args.output or Path("figures") / f"{stem}_frontier.png"

    plot_frontier(
        load_memory(memory_path),
        output=output,
        title=args.title,
        ylabel=args.ylabel,
        direction=args.direction,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
