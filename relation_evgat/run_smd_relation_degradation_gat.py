from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from run_smd_edge_gat import (
    Config as BaseConfig,
    EdgeVectorGAT,
    WindowDataset,
    build_edge_vectors,
    best_f1_threshold,
    load_smd,
    parse_interpretation,
    point_metrics,
    roc_pr_auc,
    standardize,
    threshold_from_train,
)


@dataclass
class TopConfig(BaseConfig):
    relation_window: int = 40
    relation_stride: int = 4
    edge_weight: float = 0.35
    important_edge_quantile: float = 0.90
    epochs: int = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=0, keepdims=True).T @ (b * b).sum(axis=0, keepdims=True)) + 1e-8
    return (a.T @ b) / denom


def dynamic_edge_vector(window_data: np.ndarray, max_lag: int) -> np.ndarray:
    n_nodes = window_data.shape[1]
    best_corr = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    best_abs = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    best_lag = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for lag in range(max_lag + 1):
        a = window_data[:-lag] if lag else window_data
        b = window_data[lag:] if lag else window_data
        if len(a) < 3:
            continue
        corr = corr_matrix(a, b).astype(np.float32)
        abs_corr = np.abs(corr)
        update = abs_corr > best_abs
        best_corr[update] = corr[update]
        best_abs[update] = abs_corr[update]
        best_lag[update] = lag / max_lag

    edge = np.zeros((n_nodes, n_nodes, 4), dtype=np.float32)
    edge[..., 0] = best_corr
    edge[..., 1] = best_abs
    edge[..., 2] = best_lag
    edge[..., 3] = np.where(best_corr >= 0, 1.0, -1.0)
    diag = np.eye(n_nodes, dtype=bool)
    edge[diag] = 0.0
    return edge


def relation_degradation_scores(
    data: np.ndarray,
    normal_edge: np.ndarray,
    times: np.ndarray,
    relation_window: int,
    max_lag: int,
    important_edge_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    global_scores = []
    node_scores = []
    n_nodes = data.shape[1]
    off_diag = ~np.eye(n_nodes, dtype=bool)
    cutoff = np.quantile(normal_edge[..., 1][off_diag], important_edge_quantile)
    mask = off_diag & (normal_edge[..., 1] >= cutoff)

    for time in times:
        end = int(time)
        start = max(0, end - relation_window)
        window = data[start:end]
        if len(window) < max_lag + 3:
            window = data[: max(end, max_lag + 3)]
        current_edge = dynamic_edge_vector(window, max_lag)
        delta = np.linalg.norm(current_edge - normal_edge, axis=-1)
        delta[~mask] = 0.0
        global_scores.append(float(delta[mask].mean()))
        in_count = mask.sum(axis=0).clip(min=1)
        out_count = mask.sum(axis=1).clip(min=1)
        in_score = delta.sum(axis=0) / in_count
        out_score = delta.sum(axis=1) / out_count
        node_scores.append((in_score + out_score) / 2.0)

    return np.asarray(global_scores, dtype=np.float32), np.asarray(node_scores, dtype=np.float32)


def z_from_train(train_score: np.ndarray, score: np.ndarray) -> np.ndarray:
    return (score - train_score.mean()) / (train_score.std() + 1e-8)


def robust_minmax(train_score: np.ndarray, score: np.ndarray) -> np.ndarray:
    lo = np.quantile(train_score, 0.05)
    hi = np.quantile(train_score, 0.995)
    return np.clip((score - lo) / (hi - lo + 1e-8), 0.0, None)


def train_model(
    cfg: TopConfig,
    train: np.ndarray,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> tuple[EdgeVectorGAT, list[float]]:
    loader = DataLoader(
        WindowDataset(train, cfg.window, cfg.max_train_windows),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = EdgeVectorGAT(train.shape[1], cfg.window, edge_attr.shape[-1], cfg.hidden, cfg.edge_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    losses = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred, _ = model(x, edge_attr)
            loss = torch.nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(x)
            count += len(x)
        losses.append(total / max(count, 1))
        print(f"epoch {epoch:02d}/{cfg.epochs} loss={losses[-1]:.6f}")
    return model, losses


def evaluate_node_scores(
    model: EdgeVectorGAT,
    data: np.ndarray,
    edge_attr: torch.Tensor,
    cfg: TopConfig,
    device: torch.device,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    times, score, node_errors, attention_sum, attention_count = [], [], [], None, 0
    with torch.no_grad():
        for start in range(0, len(data) - cfg.window, stride):
            x = torch.tensor(data[start : start + cfg.window].T[None, ...], dtype=torch.float32, device=device)
            y = data[start + cfg.window]
            pred, alpha = model(x, edge_attr)
            err = np.abs(pred.cpu().numpy()[0] - y)
            times.append(start + cfg.window)
            node_errors.append(err)
            score.append(float(err.mean()))
            alpha_np = alpha.cpu().numpy()[0]
            attention_sum = alpha_np if attention_sum is None else attention_sum + alpha_np
            attention_count += 1
    attention = attention_sum / max(attention_count, 1)
    return np.asarray(times, dtype=int), np.asarray(score), np.asarray(node_errors), attention


def root_cause_hit_at_k_joint(
    times: np.ndarray,
    root_scores: np.ndarray,
    intervals: list[tuple[int, int, set[int]]],
    k: int,
) -> float:
    hits = []
    for start, end, true_dims in intervals:
        mask = (times >= start) & (times <= end)
        if not mask.any() or not true_dims:
            continue
        ranking = np.argsort(root_scores[mask].mean(axis=0))[::-1][:k]
        hits.append(float(bool(set(ranking) & true_dims)))
    return float(np.mean(hits)) if hits else float("nan")


def save_top_plots(
    out_dir: Path,
    times: np.ndarray,
    labels: np.ndarray,
    node_score: np.ndarray,
    edge_score: np.ndarray,
    joint_score: np.ndarray,
    threshold: float,
    root_scores: np.ndarray,
    attention: np.ndarray,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    series = [
        ("Node prediction error score", node_score, "#2f6f9f"),
        ("Relation-degradation score", edge_score, "#9f6a2f"),
        ("Joint node-edge anomaly score", joint_score, "#7d3c98"),
    ]
    for ax, (title, score, color) in zip(axes, series):
        ax.plot(times, score, color=color, lw=1.0)
        ax.scatter(times[labels[times] == 1], score[labels[times] == 1], s=4, color="crimson", alpha=0.55)
        ax.set_title(title)
        ax.set_ylabel("Score")
    axes[-1].axhline(threshold, color="orange", ls="--", label="Joint threshold")
    axes[-1].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Relation-degradation-aware anomaly detection", y=0.995)
    fig.tight_layout()
    path = out_dir / "top_relation_joint_scores.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["joint_scores"] = str(path)

    anomaly_root = root_scores[labels[times] == 1].mean(axis=0)
    ranking = np.argsort(anomaly_root)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([f"D{i + 1}" for i in ranking[:15]], anomaly_root[ranking[:15]], color="#6c8ebf")
    ax.set_title("Joint root-cause candidate ranking")
    ax.set_ylabel("Node-edge root score")
    fig.tight_layout()
    path = out_dir / "top_joint_root_cause_ranking.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["joint_root_ranking"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(attention, cmap="viridis")
    axes[0].set_title("Average GAT attention")
    axes[0].set_xlabel("Destination")
    axes[0].set_ylabel("Source")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    relation_heat = np.corrcoef(root_scores.T)
    im1 = axes[1].imshow(relation_heat, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Root-score co-variation")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Dimension")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / "top_attention_relation_heatmaps.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["attention_relation_heatmaps"] = str(path)
    return paths


def metrics_for_score(labels: np.ndarray, score: np.ndarray, train_score: np.ndarray | None = None) -> dict[str, object]:
    if train_score is None:
        threshold, threshold_metrics = best_f1_threshold(labels, score)
        mode = "best_f1_on_test"
    else:
        threshold = threshold_from_train(train_score)
        threshold_metrics = point_metrics(labels, score > threshold)
        mode = "train_q995"
    curves = roc_pr_auc(labels, score)
    best_threshold, best_metrics = best_f1_threshold(labels, score)
    return {
        "threshold_mode": mode,
        "threshold": float(threshold),
        "metrics_at_threshold": threshold_metrics,
        "best_f1_threshold": float(best_threshold),
        "best_f1_metrics": best_metrics,
        "roc_auc": float(curves["roc_auc"]),
        "pr_auc": float(curves["pr_auc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--machine", default=TopConfig.machine)
    parser.add_argument("--epochs", type=int, default=TopConfig.epochs)
    parser.add_argument("--edge-weight", type=float, default=TopConfig.edge_weight)
    parser.add_argument("--eval-stride", type=int, default=TopConfig.eval_stride)
    parser.add_argument("--relation-window", type=int, default=TopConfig.relation_window)
    parser.add_argument("--important-edge-quantile", type=float, default=TopConfig.important_edge_quantile)
    args = parser.parse_args()

    cfg = TopConfig(
        machine=args.machine,
        epochs=args.epochs,
        edge_weight=args.edge_weight,
        eval_stride=args.eval_stride,
        relation_window=args.relation_window,
        important_edge_quantile=args.important_edge_quantile,
    )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.project / "data" / "ServerMachineDataset"
    suffix = f"w{cfg.edge_weight:.2f}_rw{cfg.relation_window}_q{cfg.important_edge_quantile:.2f}".replace(".", "p")
    out_dir = args.project / "outputs" / "smd_relation_degradation_gat" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, label = load_smd(data_root, cfg.machine)
    label = label.astype(int)
    train, test = standardize(train_raw, test_raw)
    normal_edge = build_edge_vectors(train, cfg.max_lag)
    edge_attr = torch.tensor(normal_edge, dtype=torch.float32, device=device)

    model, losses = train_model(cfg, train, edge_attr, device)
    train_times, train_node_score, train_node_errors, _ = evaluate_node_scores(
        model, train, edge_attr, cfg, device, stride=max(cfg.eval_stride * 2, 8)
    )
    times, node_score, node_errors, attention = evaluate_node_scores(
        model, test, edge_attr, cfg, device, stride=cfg.eval_stride
    )

    print("computing relation-degradation scores...")
    train_edge_score, train_edge_node = relation_degradation_scores(
        train, normal_edge, train_times, cfg.relation_window, cfg.max_lag, cfg.important_edge_quantile
    )
    edge_score, edge_node = relation_degradation_scores(
        test, normal_edge, times, cfg.relation_window, cfg.max_lag, cfg.important_edge_quantile
    )

    node_norm_train = robust_minmax(train_node_score, train_node_score)
    node_norm = robust_minmax(train_node_score, node_score)
    edge_norm_train = robust_minmax(train_edge_score, train_edge_score)
    edge_norm = robust_minmax(train_edge_score, edge_score)
    joint_train = (1.0 - cfg.edge_weight) * node_norm_train + cfg.edge_weight * edge_norm_train
    joint_score = (1.0 - cfg.edge_weight) * node_norm + cfg.edge_weight * edge_norm
    joint_threshold = threshold_from_train(joint_train)

    labels_aligned = label[times].astype(int)
    node_metrics = metrics_for_score(labels_aligned, node_norm, robust_minmax(train_node_score, train_node_score))
    edge_metrics = metrics_for_score(labels_aligned, edge_norm, robust_minmax(train_edge_score, train_edge_score))
    joint_metrics = metrics_for_score(labels_aligned, joint_score, joint_train)

    node_root = node_errors
    edge_root = edge_node / (np.quantile(train_edge_node, 0.995, axis=0, keepdims=True) + 1e-8)
    root_scores = (1.0 - cfg.edge_weight) * node_root + cfg.edge_weight * edge_root
    intervals = parse_interpretation(data_root / "interpretation_label" / f"{cfg.machine}.txt")
    hit_metrics = {
        "node_hit_at_1": root_cause_hit_at_k_joint(times, node_root, intervals, 1),
        "node_hit_at_3": root_cause_hit_at_k_joint(times, node_root, intervals, 3),
        "node_hit_at_5": root_cause_hit_at_k_joint(times, node_root, intervals, 5),
        "joint_hit_at_1": root_cause_hit_at_k_joint(times, root_scores, intervals, 1),
        "joint_hit_at_3": root_cause_hit_at_k_joint(times, root_scores, intervals, 3),
        "joint_hit_at_5": root_cause_hit_at_k_joint(times, root_scores, intervals, 5),
    }

    figures = save_top_plots(
        out_dir,
        times,
        label,
        node_norm,
        edge_norm,
        joint_score,
        joint_threshold,
        root_scores,
        attention,
    )

    summary = {
        "experiment": "relation_degradation_aware_edge_vector_gat",
        "machine": cfg.machine,
        "device": str(device),
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
        "window": cfg.window,
        "relation_window": cfg.relation_window,
        "edge_weight": cfg.edge_weight,
        "important_edge_quantile": cfg.important_edge_quantile,
        "epochs": cfg.epochs,
        "final_loss": losses[-1] if losses else None,
        "node_only": node_metrics,
        "edge_only": edge_metrics,
        "joint_node_edge": joint_metrics,
        "root_cause": hit_metrics,
        "figures": figures,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
