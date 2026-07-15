from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import OUTPUT_ROOT


MODEL_ORDER = [
    "rule-agent",
    "qwen-flash",
    "qwen3.5-flash",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]
MODEL_LABELS = {
    "rule-agent": "Rule Agent",
    "qwen-flash": "Qwen Flash",
    "qwen3.5-flash": "Qwen 3.5 Flash",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}
MODEL_COLORS = {
    "rule-agent": "#7A8793",
    "qwen-flash": "#9EC5E5",
    "qwen3.5-flash": "#5B9BD5",
    "deepseek-v4-flash": "#E6A15A",
    "deepseek-v4-pro": "#B84A3A",
}


def load_summary(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_model = {row["model"]: row for row in rows}
    return [by_model[model] for model in MODEL_ORDER if model in by_model]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#DDE3E8",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def quality_bars(rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = [
        ("schema_completeness_mean", "Schema completeness"),
        ("top1_root_mentioned_mean", "Top-1 grounding"),
        ("candidate_coverage_mean", "Candidate coverage"),
        ("citation_precision_mean", "Citation precision"),
    ]
    x = np.arange(len(rows))
    width = 0.18
    metric_colors = ["#315B7D", "#4E86A6", "#79A9B8", "#C08B54"]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for idx, (key, label) in enumerate(metrics):
        values = [row.get(key) if row.get("completed", 0) else np.nan for row in rows]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=metric_colors[idx])
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    for idx, row in enumerate(rows):
        if not row.get("completed", 0):
            ax.text(idx, 0.04, "Not run", ha="center", va="bottom", color="#8A3B32", fontsize=8, rotation=90)
    ax.set_title("Structured Output Quality and Evidence Grounding")
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1.16)
    ax.set_xticks(x, [MODEL_LABELS[row["model"]] for row in rows], rotation=15, ha="right")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig01_quality_comparison")


def quality_score(row: dict[str, Any]) -> float | None:
    if not row.get("completed", 0):
        return None
    hallucinations = float(row.get("sensor_hallucination_count_mean") or 0)
    hallucination_control = 1.0 / (1.0 + hallucinations)
    return (
        0.25 * float(row.get("schema_completeness_mean") or 0)
        + 0.25 * float(row.get("top1_root_mentioned_mean") or 0)
        + 0.20 * float(row.get("candidate_coverage_mean") or 0)
        + 0.20 * float(row.get("citation_precision_mean") or 0)
        + 0.10 * hallucination_control
    )


def pareto_scatter(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    completed = 0
    for row in rows:
        score = quality_score(row)
        if score is None:
            continue
        completed += 1
        latency_seconds = float(row.get("latency_ms_mean") or 0) / 1000.0
        tokens = float(row.get("input_tokens") or 0) + float(row.get("output_tokens") or 0)
        size = 90 + min(650, tokens / max(1, row.get("completed", 1)) * 0.15)
        model = row["model"]
        ax.scatter(
            latency_seconds,
            score,
            s=size,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            MODEL_LABELS[model],
            (latency_seconds, score),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=8.5,
        )
    missing = [MODEL_LABELS[row["model"]] for row in rows if not row.get("completed", 0)]
    if missing:
        ax.text(
            0.98,
            0.05,
            "Not run:\n" + "\n".join(missing),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#6D7780",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F6F7", "edgecolor": "#D6DCE0"},
        )
    if completed == 1:
        ax.text(
            0.02,
            0.05,
            "API models will appear after running the paid benchmark.",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#6D7780",
        )
    ax.set_title("Quality–Latency Trade-off")
    ax.set_xlabel("Mean latency (seconds, lower is better)")
    ax.set_ylabel("Composite quality score (higher is better)")
    ax.set_ylim(0, 1.08)
    ax.margins(x=0.20)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig02_quality_latency_pareto")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the two model benchmark figures.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_ROOT / "model_benchmark" / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "model_benchmark" / "artifacts",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary(args.summary)
    setup_style()
    quality_bars(rows, args.output_dir)
    pareto_scatter(rows, args.output_dir)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
