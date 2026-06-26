from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from run_smd_edge_gat import build_edge_vectors, standardize
from run_smd_relation_degradation_gat import relation_degradation_scores


def robust_column_scale(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    lo = np.quantile(reference, 0.05, axis=0, keepdims=True)
    hi = np.quantile(reference, 0.995, axis=0, keepdims=True)
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, None)


def robust_self_scale(values: np.ndarray) -> np.ndarray:
    lo = np.quantile(values, 0.05, axis=0, keepdims=True)
    hi = np.quantile(values, 0.995, axis=0, keepdims=True)
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, None)


def contiguous_events(labels: np.ndarray, times: np.ndarray) -> list[tuple[int, int, int, int]]:
    events = []
    start_idx = None
    for idx, value in enumerate(labels.astype(int)):
        if value == 1 and start_idx is None:
            start_idx = idx
        if (value == 0 or idx == len(labels) - 1) and start_idx is not None:
            end_idx = idx - 1 if value == 0 else idx
            events.append((len(events) + 1, start_idx, end_idx, int(times[start_idx])))
            start_idx = None
    return events


def names_for(indices: np.ndarray, columns: list[str]) -> list[str]:
    return [columns[int(i)] if int(i) < len(columns) else f"D{int(i) + 1}" for i in indices]


def save_root_cause_plots(
    out_dir: Path,
    dataset: str,
    rows: list[dict[str, object]],
    columns: list[str],
    joint_root: np.ndarray,
    aligned_labels: np.ndarray,
    events: list[tuple[int, int, int, int]],
    top_k: int,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if aligned_labels.any():
        global_score = joint_root[aligned_labels == 1].mean(axis=0)
    else:
        global_score = joint_root.mean(axis=0)
    ranking = np.argsort(global_score)[::-1][:top_k]
    labels = names_for(ranking, columns)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(range(len(ranking)), global_score[ranking], color="#7d3c98")
    ax.set_xticks(range(len(ranking)), labels, rotation=55, ha="right")
    ax.set_ylabel("Mean joint root score")
    ax.set_title(f"{dataset}: global Top-{top_k} root-cause candidates")
    fig.tight_layout()
    path = out_dir / "global_root_cause_topk.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["global_topk"] = str(path)

    if events:
        event_scores = []
        for _, start_idx, end_idx, _ in events:
            event_scores.append(joint_root[start_idx : end_idx + 1].mean(axis=0)[ranking])
        matrix = np.asarray(event_scores)
        fig, ax = plt.subplots(figsize=(12, max(4.0, 0.35 * len(events) + 2.2)))
        image = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(ranking)), labels, rotation=55, ha="right")
        ax.set_yticks(range(len(events)), [f"E{event_id}" for event_id, *_ in events])
        ax.set_title(f"{dataset}: event-level joint root scores")
        ax.set_xlabel("Candidate sensor")
        ax.set_ylabel("Anomaly event")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Joint root score")
        fig.tight_layout()
        path = out_dir / "event_root_cause_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["event_heatmap"] = str(path)

        row_min = matrix.min(axis=1, keepdims=True)
        row_max = matrix.max(axis=1, keepdims=True)
        row_norm = (matrix - row_min) / (row_max - row_min + 1e-8)
        fig, ax = plt.subplots(figsize=(12, max(4.0, 0.35 * len(events) + 2.2)))
        image = ax.imshow(row_norm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(ranking)), labels, rotation=55, ha="right")
        ax.set_yticks(range(len(events)), [f"E{event_id}" for event_id, *_ in events])
        ax.set_title(f"{dataset}: row-normalized event root-cause heatmap")
        ax.set_xlabel("Candidate sensor")
        ax.set_ylabel("Anomaly event")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Within-event normalized score")
        fig.tight_layout()
        path = out_dir / "event_root_cause_heatmap_row_normalized.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["event_heatmap_row_normalized"] = str(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="WaDI_A2_ds10")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--edge-weight", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    project = args.project
    data_dir = project / "data" / "top_ready" / args.dataset
    run_dir = args.run_dir or project / "outputs" / "top_ready_relation_gat" / args.dataset / "full_joint"
    out_dir = project / "outputs" / "root_cause" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = np.load(data_dir / "train.npy").astype(np.float32)
    test_raw = np.load(data_dir / "test.npy").astype(np.float32)
    test_label = np.load(data_dir / "test_label.npy").astype(int)
    columns = json.loads((data_dir / "columns.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    times = np.load(run_dir / "times.npy").astype(int)
    node_errors = np.load(run_dir / "node_errors.npy").astype(np.float32)
    aligned_labels = test_label[times].astype(int)
    events = contiguous_events(aligned_labels, times)

    edge_weight = args.edge_weight
    if edge_weight is None:
        edge_weight = float(summary.get("edge_weight", 0.10))
    max_lag = int(summary.get("max_lag", 5))
    relation_window = int(summary.get("relation_window", 40))
    important_edge_quantile = float(summary.get("important_edge_quantile", 0.90))

    train, test = standardize(train_raw, test_raw)
    normal_edge = build_edge_vectors(train, max_lag)

    # The original model run saved only global edge degradation. Recompute the
    # node-level relation degradation here so root-cause ranking can use both
    # node and edge evidence without retraining the model.
    train_times = np.arange(summary.get("window", 30), len(train), max(int(summary.get("eval_stride", 8)) * 2, 16))
    _, train_edge_node = relation_degradation_scores(
        train,
        normal_edge,
        train_times,
        relation_window,
        max_lag,
        important_edge_quantile,
    )
    _, edge_node = relation_degradation_scores(
        test,
        normal_edge,
        times,
        relation_window,
        max_lag,
        important_edge_quantile,
    )

    node_norm = robust_self_scale(node_errors)
    edge_norm = robust_column_scale(train_edge_node, edge_node)
    joint_root = (1.0 - edge_weight) * node_norm + edge_weight * edge_norm

    np.save(out_dir / "node_root_score.npy", node_norm)
    np.save(out_dir / "edge_root_score.npy", edge_norm)
    np.save(out_dir / "joint_root_score.npy", joint_root)
    np.save(out_dir / "times.npy", times)

    rows = []
    for event_id, start_idx, end_idx, raw_start_time in events:
        mask = slice(start_idx, end_idx + 1)
        node_event = node_norm[mask].mean(axis=0)
        edge_event = edge_norm[mask].mean(axis=0)
        joint_event = joint_root[mask].mean(axis=0)
        ranking = np.argsort(joint_event)[::-1][: args.top_k]
        node_ranking = np.argsort(node_event)[::-1][: args.top_k]
        edge_ranking = np.argsort(edge_event)[::-1][: args.top_k]
        row = {
            "dataset": args.dataset,
            "event_id": event_id,
            "aligned_start_index": start_idx,
            "aligned_end_index": end_idx,
            "raw_start_time": raw_start_time,
            "raw_end_time": int(times[end_idx]),
            "duration_aligned_points": end_idx - start_idx + 1,
            "duration_raw_points_approx": int(times[end_idx] - times[start_idx] + 1),
            "top1_joint": names_for(ranking[:1], columns)[0],
            "top3_joint": ";".join(names_for(ranking[:3], columns)),
            "top5_joint": ";".join(names_for(ranking[:5], columns)),
            "top10_joint": ";".join(names_for(ranking, columns)),
            "top5_node_only": ";".join(names_for(node_ranking[:5], columns)),
            "top5_edge_only": ";".join(names_for(edge_ranking[:5], columns)),
            "top10_joint_indices": ";".join(str(int(i)) for i in ranking),
            "top10_joint_scores": ";".join(f"{float(joint_event[i]):.6f}" for i in ranking),
            "top10_node_scores": ";".join(f"{float(node_event[i]):.6f}" for i in ranking),
            "top10_edge_scores": ";".join(f"{float(edge_event[i]):.6f}" for i in ranking),
        }
        rows.append(row)

    csv_path = out_dir / "event_root_cause_candidates.csv"
    json_path = out_dir / "event_root_cause_candidates.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    figures = save_root_cause_plots(
        out_dir,
        args.dataset,
        rows,
        columns,
        joint_root,
        aligned_labels,
        events,
        args.top_k,
    )

    global_joint = joint_root[aligned_labels == 1].mean(axis=0) if aligned_labels.any() else joint_root.mean(axis=0)
    global_ranking = np.argsort(global_joint)[::-1][: args.top_k]
    report = {
        "dataset": args.dataset,
        "run_dir": str(run_dir),
        "num_events": len(events),
        "edge_weight": edge_weight,
        "relation_window": relation_window,
        "important_edge_quantile": important_edge_quantile,
        "top_global_joint": names_for(global_ranking, columns),
        "csv": str(csv_path),
        "json": str(json_path),
        "figures": figures,
    }
    (out_dir / "root_cause_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
