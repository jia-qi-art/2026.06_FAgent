from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    machine: str = "machine-1-1"
    window: int = 30
    max_lag: int = 5
    hidden: int = 32
    edge_hidden: int = 16
    batch_size: int = 128
    epochs: int = 8
    lr: float = 1e-3
    max_train_windows: int = 8000
    eval_stride: int = 4
    seed: int = 7


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int, max_windows: int | None = None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window = window
        n = len(data) - window
        indices = np.arange(n)
        if max_windows is not None and n > max_windows:
            rng = np.random.default_rng(7)
            indices = np.sort(rng.choice(indices, size=max_windows, replace=False))
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.indices[idx])
        x = self.data[start : start + self.window].T
        y = self.data[start + self.window]
        return x, y


class EdgeVectorGAT(nn.Module):
    def __init__(self, n_nodes: int, window: int, edge_dim: int, hidden: int, edge_hidden: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, hidden),
        )
        self.attn = nn.Linear(hidden * 3, 1)
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        mask = torch.eye(n_nodes, dtype=torch.bool)
        self.register_buffer("self_mask", mask)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, nodes, window], edge_attr: [nodes, nodes, edge_dim]
        h = self.node_encoder(x)
        bsz, n_nodes, hidden = h.shape
        edge_h = self.edge_encoder(edge_attr)

        src = h.unsqueeze(2).expand(bsz, n_nodes, n_nodes, hidden)
        dst = h.unsqueeze(1).expand(bsz, n_nodes, n_nodes, hidden)
        edge = edge_h.unsqueeze(0).expand(bsz, n_nodes, n_nodes, hidden)
        score = self.attn(torch.cat([src, dst, edge], dim=-1)).squeeze(-1)
        score = score.masked_fill(self.self_mask.unsqueeze(0), -1e9)
        alpha = torch.softmax(torch.nn.functional.leaky_relu(score, 0.2), dim=1)
        message = torch.einsum("bij,bih->bjh", alpha, h)
        pred = self.out(torch.cat([h, message], dim=-1)).squeeze(-1)
        return pred, alpha


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_smd(root: Path, machine: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = root / "train" / f"{machine}.txt"
    test_path = root / "test" / f"{machine}.txt"
    label_path = root / "test_label" / f"{machine}.txt"
    missing = [str(path) for path in [train_path, test_path, label_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SMD data files were not found. Missing:\n"
            + "\n".join(missing)
            + "\nRun with --project pointing to a project that contains data/ServerMachineDataset."
        )
    train = np.loadtxt(train_path, delimiter=",")
    test = np.loadtxt(test_path, delimiter=",")
    label = np.loadtxt(label_path, delimiter=",")
    return train, test, label


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (test - mean) / std


def lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    best_corr = 0.0
    best_lag = 0
    for lag in range(max_lag + 1):
        aa = a[:-lag] if lag else a
        bb = b[lag:] if lag else b
        if aa.std() < 1e-12 or bb.std() < 1e-12:
            corr = 0.0
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
            continue
        corr = np.corrcoef(aa, bb)[0, 1]
        corr = 0.0 if np.isnan(corr) else float(corr)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag


def build_edge_vectors(train: np.ndarray, max_lag: int) -> np.ndarray:
    n_nodes = train.shape[1]
    edge = np.zeros((n_nodes, n_nodes, 4), dtype=np.float32)
    sample = train[: min(len(train), 8000)]
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst:
                continue
            corr, lag = lagged_corr(sample[:, src], sample[:, dst], max_lag)
            edge[src, dst, 0] = corr
            edge[src, dst, 1] = abs(corr)
            edge[src, dst, 2] = lag / max_lag
            edge[src, dst, 3] = 1.0 if corr >= 0 else -1.0
    return edge


def evaluate_scores(
    model: EdgeVectorGAT,
    data: np.ndarray,
    edge_attr: torch.Tensor,
    window: int,
    device: torch.device,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    times = []
    node_errors = []
    with torch.no_grad():
        for start in range(0, len(data) - window, stride):
            x = torch.tensor(data[start : start + window].T[None, ...], dtype=torch.float32, device=device)
            y = data[start + window]
            pred, _ = model(x, edge_attr)
            err = np.abs(pred.cpu().numpy()[0] - y)
            times.append(start + window)
            node_errors.append(err)
    times_arr = np.array(times, dtype=int)
    node_err = np.array(node_errors)
    score = node_err.mean(axis=1)
    return times_arr, score, node_err


def point_metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def roc_pr_auc(labels: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray | float]:
    labels = labels.astype(int)
    order = np.argsort(score)[::-1]
    y = labels[order]
    s = score[order]
    distinct = np.r_[True, s[1:] != s[:-1]]
    tps = np.cumsum(y)[distinct]
    fps = np.cumsum(1 - y)[distinct]
    positives = max(int(labels.sum()), 1)
    negatives = max(int((1 - labels).sum()), 1)

    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    precision = np.r_[1.0, tps / np.maximum(tps + fps, 1)]
    recall = np.r_[0.0, tps / positives]
    roc_auc = float(np.trapezoid(tpr, fpr))
    pr_auc = float(np.trapezoid(precision, recall))
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall, "roc_auc": roc_auc, "pr_auc": pr_auc}


def best_f1_threshold(labels: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.quantile(score, np.linspace(0.50, 0.995, 120))
    best_threshold = float(candidates[0])
    best_metrics = point_metrics(labels, score > best_threshold)
    for threshold in candidates:
        metrics = point_metrics(labels, score > threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def parse_interpretation(path: Path) -> list[tuple[int, int, set[int]]]:
    intervals = []
    if not path.exists():
        return intervals
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        span, dims = line.split(":", 1)
        start, end = [int(v) for v in span.split("-")]
        dim_set = {int(v) - 1 for v in dims.split(",") if v.strip()}
        intervals.append((start, end, dim_set))
    return intervals


def root_cause_hit_at_k(
    times: np.ndarray,
    node_errors: np.ndarray,
    intervals: list[tuple[int, int, set[int]]],
    k: int,
) -> float:
    hits = []
    for start, end, true_dims in intervals:
        mask = (times >= start) & (times <= end)
        if not mask.any() or not true_dims:
            continue
        ranking = np.argsort(node_errors[mask].mean(axis=0))[::-1][:k]
        hits.append(float(bool(set(ranking) & true_dims)))
    return float(np.mean(hits)) if hits else math.nan


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 0.18, height: float = 0.13) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#2f3b52",
        facecolor="#f7f9fc",
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#2f3b52")
    ax.add_patch(arrow)


def save_manual_figures(out_dir: Path) -> dict[str, str]:
    paths = {}

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()
    labels = [
        "SMD raw\nmultivariate series",
        "Sliding windows\nand labels",
        "Lagged edge\nvectors",
        "Edge-Vector\nGAT",
        "Prediction\nerror score",
        "Fault detection\nand root cause",
    ]
    xs = np.linspace(0.04, 0.82, len(labels))
    for x, label in zip(xs, labels):
        draw_box(ax, (float(x), 0.54), label)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        draw_arrow(ax, (float(x1 + 0.18), 0.605), (float(x2), 0.605))
    ax.text(0.5, 0.24, "Fig. 2. End-to-end workflow used by the SMD Edge-Vector GAT experiment.", ha="center", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig02_workflow.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["fig02_workflow"] = str(path)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_axis_off()
    draw_box(ax, (0.08, 0.62), "Source metric\nx_i(t)", 0.2, 0.16)
    draw_box(ax, (0.08, 0.20), "Target metric\nx_j(t)", 0.2, 0.16)
    draw_box(ax, (0.42, 0.42), "Lag search\n0...L", 0.18, 0.16)
    draw_box(ax, (0.72, 0.42), "Edge vector\n[corr, |corr|,\nlag/L, sign]", 0.22, 0.18)
    draw_arrow(ax, (0.28, 0.70), (0.42, 0.52))
    draw_arrow(ax, (0.28, 0.28), (0.42, 0.46))
    draw_arrow(ax, (0.60, 0.50), (0.72, 0.50))
    ax.plot([0.08, 0.28], [0.58, 0.86], color="#5277a3", lw=1.8)
    ax.plot([0.08, 0.28], [0.15, 0.38], color="#b45f5f", lw=1.8)
    ax.text(0.5, 0.08, "Fig. 3. Construction of lagged causal edge attributes for each ordered metric pair.", ha="center", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig03_edge_vector_construction.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["fig03_edge_vector_construction"] = str(path)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_axis_off()
    draw_box(ax, (0.05, 0.58), "Windowed\nnode signals", 0.18, 0.15)
    draw_box(ax, (0.32, 0.72), "Node\nencoder", 0.16, 0.13)
    draw_box(ax, (0.32, 0.38), "Edge\nencoder", 0.16, 0.13)
    draw_box(ax, (0.58, 0.55), "Attention\nmessage passing", 0.2, 0.16)
    draw_box(ax, (0.84, 0.58), "Next-step\nprediction", 0.16, 0.15)
    draw_arrow(ax, (0.23, 0.66), (0.32, 0.78))
    draw_arrow(ax, (0.23, 0.64), (0.32, 0.45))
    draw_arrow(ax, (0.48, 0.78), (0.58, 0.66))
    draw_arrow(ax, (0.48, 0.45), (0.58, 0.60))
    draw_arrow(ax, (0.78, 0.63), (0.84, 0.66))
    ax.text(0.5, 0.18, "Fig. 4. Edge-Vector GAT model architecture for prediction-error anomaly detection.", ha="center", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig04_model_architecture.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["fig04_model_architecture"] = str(path)
    return paths


def interval_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    starts = np.where((labels[1:] == 1) & (labels[:-1] == 0))[0] + 1
    if labels[0] == 1:
        starts = np.r_[0, starts]
    ends = np.where((labels[1:] == 0) & (labels[:-1] == 1))[0]
    if labels[-1] == 1:
        ends = np.r_[ends, len(labels) - 1]
    for start, end in zip(starts, ends):
        spans.append((int(start), int(end)))
    return spans


def moving_average_scores(data: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = []
    node_errors = []
    for start in range(0, len(data) - window, stride):
        target_t = start + window
        pred = data[start:target_t].mean(axis=0)
        err = np.abs(data[target_t] - pred)
        times.append(target_t)
        node_errors.append(err)
    node_err = np.asarray(node_errors)
    return np.asarray(times, dtype=int), node_err.mean(axis=1), node_err


def z_deviation_scores(data: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(window, len(data), stride, dtype=int)
    node_err = np.abs(data[times])
    return times, node_err.mean(axis=1), node_err


def threshold_from_train(train_score: np.ndarray) -> float:
    return float(np.quantile(train_score, 0.995))


def evaluate_edge_variant(
    name: str,
    model: EdgeVectorGAT,
    train: np.ndarray,
    test: np.ndarray,
    edge: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> dict[str, float | str | np.ndarray]:
    variant = edge.copy()
    if name == "Corr only":
        variant[..., 2] = 0.0
        variant[..., 3] = 0.0
    elif name == "No lag":
        variant[..., 2] = 0.0
    elif name == "No sign":
        variant[..., 3] = 0.0
    elif name == "Shuffled edge":
        rng = np.random.default_rng(cfg.seed)
        flat = variant.reshape(-1, variant.shape[-1]).copy()
        rng.shuffle(flat, axis=0)
        variant = flat.reshape(variant.shape)
    edge_attr = torch.tensor(variant, dtype=torch.float32, device=device)
    _, train_score, _ = evaluate_scores(model, train, edge_attr, cfg.window, device, stride=max(cfg.eval_stride * 2, 8))
    times, score, _ = evaluate_scores(model, test, edge_attr, cfg.window, device, cfg.eval_stride)
    threshold = threshold_from_train(train_score)
    aligned = labels[times].astype(int)
    metrics = point_metrics(aligned, score > threshold)
    curves = roc_pr_auc(aligned, score)
    return {
        "name": name,
        "threshold": threshold,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "roc_auc": curves["roc_auc"],
        "pr_auc": curves["pr_auc"],
    }


def save_paper_figures(
    out_dir: Path,
    cfg: Config,
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    intervals: list[tuple[int, int, set[int]]],
    times: np.ndarray,
    score: np.ndarray,
    node_errors: np.ndarray,
    threshold: float,
    attention: np.ndarray,
    edge: np.ndarray,
    model: EdgeVectorGAT,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, object]]:
    paths = {}
    details: dict[str, object] = {}
    labels_aligned = labels[times].astype(int)
    pred = (score > threshold).astype(int)
    metrics = point_metrics(labels_aligned, pred)
    curves = roc_pr_auc(labels_aligned, score)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), height_ratios=[1.0, 1.0, 0.52], sharex=False)
    selected = np.linspace(0, test.shape[1] - 1, min(6, test.shape[1]), dtype=int)
    offset = 0.0
    for dim in selected:
        trace = test_raw[:, dim]
        trace = (trace - trace.mean()) / (trace.std() + 1e-6)
        axes[0].plot(trace + offset, lw=0.8, label=f"D{dim + 1}")
        offset += 4.0
    axes[0].set_title(f"Fig. 1. SMD {cfg.machine} data overview")
    axes[0].set_ylabel("Selected metrics")
    axes[0].legend(ncol=6, fontsize=8, loc="upper right")
    axes[1].plot(np.arange(len(labels)), labels, color="crimson", lw=1.0)
    axes[1].set_ylabel("Anomaly label")
    axes[1].set_ylim(-0.08, 1.18)
    axes[2].bar(["Train length", "Test length", "Metrics", "Anomaly points"], [len(train_raw), len(test_raw), test.shape[1], int(labels.sum())], color=["#5277a3", "#5277a3", "#6aa36f", "#c95f5f"])
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Dataset summary")
    for ax in axes[:2]:
        for start, end in interval_spans(labels.astype(int))[:8]:
            ax.axvspan(start, end, color="crimson", alpha=0.08)
    fig.tight_layout()
    path = out_dir / "fig01_smd_data_overview.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["fig01_smd_data_overview"] = str(path)

    manual_paths = save_manual_figures(out_dir)
    paths.update(manual_paths)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(times, score, lw=1.2, label="Edge-Vector GAT score", color="#2f6f9f")
    ax.axhline(threshold, color="#d5962c", ls="--", label="Train 99.5% threshold")
    anomalous = labels_aligned == 1
    if anomalous.any():
        ax.scatter(times[anomalous], score[anomalous], s=7, color="crimson", alpha=0.7, label="Labeled anomaly")
    for start, end in interval_spans(labels.astype(int))[:8]:
        ax.axvspan(start, end, color="crimson", alpha=0.07)
    ax.set_title("Fig. 5. Anomaly score produced by prediction error")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean node error")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / "fig05_anomaly_score.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["fig05_anomaly_score"] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(curves["fpr"], curves["tpr"], color="#2f6f9f", lw=2)
    axes[0].plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    axes[0].set_title(f"ROC AUC={curves['roc_auc']:.3f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].plot(curves["recall"], curves["precision"], color="#6aa36f", lw=2)
    axes[1].set_title(f"PR AUC={curves['pr_auc']:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_title(f"F1={metrics['f1']:.3f}")
    axes[2].set_xticks([0, 1], ["Normal", "Anomaly"])
    axes[2].set_yticks([0, 1], ["Normal", "Anomaly"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(int(cm[i, j])), ha="center", va="center", color="#1b2a41")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Fig. 6. Detection quality under labeled SMD test windows", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig06_detection_quality.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["fig06_detection_quality"] = str(path)

    base_train_t, base_train_score, _ = moving_average_scores(train, cfg.window, max(cfg.eval_stride * 2, 8))
    base_t, base_score, _ = moving_average_scores(test, cfg.window, cfg.eval_stride)
    z_train_t, z_train_score, _ = z_deviation_scores(train, cfg.window, max(cfg.eval_stride * 2, 8))
    z_t, z_score, _ = z_deviation_scores(test, cfg.window, cfg.eval_stride)
    method_rows = []
    for name, train_score_i, test_times_i, test_score_i in [
        ("Z-deviation", z_train_score, z_t, z_score),
        ("Moving average", base_train_score, base_t, base_score),
        ("Edge-Vector GAT", score, times, score),
    ]:
        aligned = labels[test_times_i].astype(int)
        if name == "Edge-Vector GAT":
            threshold_i = threshold
        else:
            threshold_i = threshold_from_train(train_score_i)
        method_metrics = point_metrics(aligned, test_score_i > threshold_i)
        method_curves = roc_pr_auc(aligned, test_score_i)
        method_rows.append(
            {
                "name": name,
                "f1": method_metrics["f1"],
                "precision": method_metrics["precision"],
                "recall": method_metrics["recall"],
                "roc_auc": method_curves["roc_auc"],
                "pr_auc": method_curves["pr_auc"],
            }
        )
    x = np.arange(len(method_rows))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, metric_name in enumerate(["f1", "precision", "recall", "roc_auc"]):
        ax.bar(x + (k - 1.5) * width, [row[metric_name] for row in method_rows], width, label=metric_name.upper())
    ax.set_xticks(x, [row["name"] for row in method_rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fig. 7. Baseline comparison on SMD anomaly detection")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    path = out_dir / "fig07_method_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["fig07_method_comparison"] = str(path)
    details["method_comparison"] = method_rows

    variant_rows = [
        evaluate_edge_variant("Full edge", model, train, test, edge, labels, cfg, device),
        evaluate_edge_variant("Corr only", model, train, test, edge, labels, cfg, device),
        evaluate_edge_variant("No lag", model, train, test, edge, labels, cfg, device),
        evaluate_edge_variant("No sign", model, train, test, edge, labels, cfg, device),
        evaluate_edge_variant("Shuffled edge", model, train, test, edge, labels, cfg, device),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = [str(row["name"]) for row in variant_rows]
    axes[0].bar(names, [float(row["f1"]) for row in variant_rows], color="#5277a3")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Edge-attribute ablation")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].plot(names, [float(row["roc_auc"]) for row in variant_rows], marker="o", label="ROC AUC")
    axes[1].plot(names, [float(row["pr_auc"]) for row in variant_rows], marker="s", label="PR AUC")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Sensitivity to edge information")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    fig.suptitle("Fig. 8. Ablation and sensitivity analysis", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig08_ablation_sensitivity.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["fig08_ablation_sensitivity"] = str(path)
    details["ablation"] = [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()} for row in variant_rows]

    if intervals:
        case_start, case_end, true_dims = intervals[0]
    else:
        spans = interval_spans(labels.astype(int))
        case_start, case_end = spans[0] if spans else (int(times[len(times) // 3]), int(times[len(times) // 3] + 240))
        true_dims = set()
    pad = 120
    mask = (times >= max(0, case_start - pad)) & (times <= min(len(labels) - 1, case_end + pad))
    case_errors = node_errors[mask].T
    case_times = times[mask]
    root_score = node_errors[(times >= case_start) & (times <= case_end)].mean(axis=0)
    ranking = np.argsort(root_score)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), width_ratios=[2.1, 1.0])
    im = axes[0].imshow(case_errors, aspect="auto", cmap="magma", extent=[case_times[0], case_times[-1], case_errors.shape[0], 1])
    axes[0].axvspan(case_start, case_end, color="cyan", alpha=0.15, label="Labeled interval")
    axes[0].set_title("Local node-error heatmap")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Metric dimension")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    top_k = ranking[:12]
    colors = ["crimson" if int(dim) in true_dims else "#6c8ebf" for dim in top_k]
    axes[1].barh([f"D{dim + 1}" for dim in top_k[::-1]], root_score[top_k[::-1]], color=colors[::-1])
    axes[1].set_title("Top root-cause candidates")
    axes[1].set_xlabel("Mean node error")
    fig.suptitle("Fig. 9. Typical anomaly case visualization", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig09_typical_case.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["fig09_typical_case"] = str(path)

    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(120):
        cols = rng.integers(0, node_errors.shape[1], size=node_errors.shape[1])
        boot.append(node_errors[:, cols].mean(axis=1))
    boot_arr = np.asarray(boot)
    lower = np.quantile(boot_arr, 0.05, axis=0)
    upper = np.quantile(boot_arr, 0.95, axis=0)
    source_scores = {
        "Top 5 metrics": float(np.mean(np.sort(node_errors[labels_aligned == 1].mean(axis=0))[-5:])),
        "Remaining metrics": float(np.mean(np.sort(node_errors[labels_aligned == 1].mean(axis=0))[:-5])),
        "Normal windows": float(np.mean(node_errors[labels_aligned == 0])),
        "Anomaly windows": float(np.mean(node_errors[labels_aligned == 1])),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].plot(times, score, color="#2f6f9f", lw=1.0, label="Mean score")
    axes[0].fill_between(times, lower, upper, color="#2f6f9f", alpha=0.22, label="90% bootstrap band")
    axes[0].axhline(threshold, color="#d5962c", ls="--", label="Threshold")
    axes[0].set_title("Score uncertainty from metric bootstrap")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right")
    axes[1].bar(source_scores.keys(), source_scores.values(), color=["#5277a3", "#8aa6c6", "#6aa36f", "#c95f5f"])
    axes[1].set_title("Error source summary")
    axes[1].set_ylabel("Mean node error")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Fig. 10. Uncertainty and error-source analysis", y=1.02)
    fig.tight_layout()
    path = out_dir / "fig10_uncertainty_error_sources.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["fig10_uncertainty_error_sources"] = str(path)

    details["detection_metrics"] = metrics
    details["roc_auc"] = curves["roc_auc"]
    details["pr_auc"] = curves["pr_auc"]
    return paths, details


def save_legacy_plots(
    out_dir: Path,
    times: np.ndarray,
    score: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    node_errors: np.ndarray,
    attention: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(times, score, lw=1.2, label="Anomaly score")
    ax.axhline(threshold, color="orange", ls="--", label="Train threshold")
    anomalous = times[labels[times] == 1]
    if len(anomalous):
        ax.scatter(anomalous, score[labels[times] == 1], s=4, color="crimson", label="True anomaly")
    ax.set_title("SMD machine-1-1 anomaly score")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean prediction error")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "smd_anomaly_score.png", dpi=170)
    plt.close(fig)

    root_score = node_errors[labels[times] == 1].mean(axis=0)
    ranking = np.argsort(root_score)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([f"D{i + 1}" for i in ranking[:15]], root_score[ranking[:15]], color="#6c8ebf")
    ax.set_title("Top anomalous dimensions on labeled anomaly windows")
    ax.set_ylabel("Mean node prediction error")
    fig.tight_layout()
    fig.savefig(out_dir / "smd_dimension_ranking.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(attention, cmap="viridis")
    ax.set_title("Average Edge-Vector GAT attention")
    ax.set_xlabel("Destination metric")
    ax.set_ylabel("Source metric")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "smd_attention_heatmap.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--machine", default=Config.machine)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--eval-stride", type=int, default=Config.eval_stride)
    parser.add_argument("--max-train-windows", type=int, default=Config.max_train_windows)
    args = parser.parse_args()

    cfg = Config(machine=args.machine, epochs=args.epochs, eval_stride=args.eval_stride, max_train_windows=args.max_train_windows)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.project / "data" / "ServerMachineDataset"
    out_dir = args.project / "outputs" / "smd_edge_gat"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, label = load_smd(data_root, cfg.machine)
    label = label.astype(int)
    train, test = standardize(train_raw, test_raw)
    edge = build_edge_vectors(train, cfg.max_lag)
    edge_attr = torch.tensor(edge, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        WindowDataset(train, cfg.window, cfg.max_train_windows),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = EdgeVectorGAT(train.shape[1], cfg.window, edge.shape[-1], cfg.hidden, cfg.edge_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    losses = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
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
        epoch_loss = total / max(count, 1)
        losses.append(epoch_loss)
        print(f"epoch {epoch:02d}/{cfg.epochs} loss={epoch_loss:.6f}")

    train_times, train_score, _ = evaluate_scores(
        model, train, edge_attr, cfg.window, device, stride=max(cfg.eval_stride * 2, 8)
    )
    threshold = threshold_from_train(train_score)
    times, score, node_errors = evaluate_scores(model, test, edge_attr, cfg.window, device, cfg.eval_stride)
    pred = (score > threshold).astype(int)
    labels_aligned = label[times].astype(int)
    metrics = point_metrics(labels_aligned, pred)

    with torch.no_grad():
        x = torch.tensor(test[: cfg.window].T[None, ...], dtype=torch.float32, device=device)
        _, alpha = model(x, edge_attr)
        attention = alpha.cpu().numpy()[0]

    intervals = parse_interpretation(data_root / "interpretation_label" / f"{cfg.machine}.txt")
    hit1 = root_cause_hit_at_k(times, node_errors, intervals, k=1)
    hit3 = root_cause_hit_at_k(times, node_errors, intervals, k=3)
    hit5 = root_cause_hit_at_k(times, node_errors, intervals, k=5)

    save_legacy_plots(out_dir, times, score, label, threshold, node_errors, attention)
    paper_figures, paper_details = save_paper_figures(
        out_dir,
        cfg,
        train_raw,
        test_raw,
        train,
        test,
        label,
        intervals,
        times,
        score,
        node_errors,
        threshold,
        attention,
        edge,
        model,
        device,
    )

    summary = {
        "machine": cfg.machine,
        "device": str(device),
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
        "window": cfg.window,
        "epochs": cfg.epochs,
        "eval_stride": cfg.eval_stride,
        "threshold_train_quantile": 0.995,
        "threshold": threshold,
        "metrics": metrics,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "final_loss": losses[-1] if losses else None,
        "legacy_figures": [
            "outputs/smd_edge_gat/smd_anomaly_score.png",
            "outputs/smd_edge_gat/smd_dimension_ranking.png",
            "outputs/smd_edge_gat/smd_attention_heatmap.png",
        ],
        "paper_figures": paper_figures,
        "paper_details": paper_details,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "figure_manifest.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in sorted(paper_figures.items())),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
