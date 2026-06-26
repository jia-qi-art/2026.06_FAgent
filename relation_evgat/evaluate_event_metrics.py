from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_smd_edge_gat import best_f1_threshold, point_metrics
from run_top_ready_relation_gat import apply_consecutive_filter, smooth_score


def spans(binary: np.ndarray) -> list[tuple[int, int]]:
    binary = binary.astype(int)
    result = []
    start = None
    for idx, value in enumerate(binary):
        if value and start is None:
            start = idx
        if (not value or idx == len(binary) - 1) and start is not None:
            end = idx if not value else idx + 1
            result.append((start, end))
            start = None
    return result


def event_metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    true_events = spans(labels)
    pred_events = spans(pred)
    detected = 0
    delays = []
    for start, end in true_events:
        hits = np.where(pred[start:end] == 1)[0]
        if len(hits):
            detected += 1
            delays.append(int(hits[0]))
    matched_pred = 0
    for start, end in pred_events:
        if labels[start:end].any():
            matched_pred += 1
    precision = matched_pred / (len(pred_events) + 1e-9)
    recall = detected / (len(true_events) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "true_events": len(true_events),
        "pred_events": len(pred_events),
        "detected_events": detected,
        "matched_pred_events": matched_pred,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "mean_detection_delay_points": float(np.mean(delays)) if delays else None,
    }


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="WaDI_A2_ds10")
    parser.add_argument("--score-name", choices=["node", "joint"], default="joint")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-consecutive", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.project / "outputs" / "top_ready_relation_gat" / args.dataset
    summary = load_summary(out_dir / "summary.json")
    data_root = args.project / "data" / "top_ready" / args.dataset
    labels_full = np.load(data_root / "test_label.npy").astype(int)

    # Reconstruct the aligned score from saved plot inputs is not possible, so rerun a lightweight parse is avoided.
    # Instead, this script expects score arrays saved by future runs.
    score_file = out_dir / f"{args.score_name}_score.npy"
    times_file = out_dir / "times.npy"
    if not score_file.exists() or not times_file.exists():
        raise FileNotFoundError(
            f"Missing {score_file.name} or times.npy. Re-run run_top_ready_relation_gat.py after score-array saving is enabled."
        )
    score = np.load(score_file)
    times = np.load(times_file)
    labels = labels_full[times].astype(int)
    smoothed = smooth_score(score, args.smooth_window)
    threshold, best_metrics = best_f1_threshold(labels, smoothed)
    pred = apply_consecutive_filter((smoothed > threshold).astype(int), args.min_consecutive)
    result = {
        "dataset": args.dataset,
        "score_name": args.score_name,
        "threshold": float(threshold),
        "point_metrics": point_metrics(labels, pred),
        "point_best_f1_before_filter": best_metrics,
        "event_metrics": event_metrics(labels, pred),
    }
    path = out_dir / f"event_metrics_{args.score_name}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

