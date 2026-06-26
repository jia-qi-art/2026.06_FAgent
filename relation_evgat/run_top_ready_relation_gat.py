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
    EdgeVectorGAT,
    WindowDataset,
    best_f1_threshold,
    build_edge_vectors,
    point_metrics,
    roc_pr_auc,
    standardize,
    threshold_from_train,
)
from run_smd_relation_degradation_gat import relation_degradation_scores, robust_minmax


@dataclass
class Config:
    dataset: str = "WaDI_A2_ds10"
    window: int = 30
    max_lag: int = 5
    hidden: int = 24
    edge_hidden: int = 12
    batch_size: int = 64
    epochs: int = 6
    lr: float = 1e-3
    max_train_windows: int = 6000
    eval_stride: int = 8
    relation_window: int = 40
    edge_weight: float = 0.10
    important_edge_quantile: float = 0.90
    edge_mode: str = "full"
    use_relation_degradation: bool = True
    smooth_window: int = 5
    min_consecutive: int = 3
    val_fraction: float = 0.20
    seed: int = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_top_ready(project: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    root = project / "data" / "top_ready" / dataset
    if not root.exists():
        raise FileNotFoundError(f"Top-ready dataset not found: {root}")
    train = np.load(root / "train.npy").astype(np.float32)
    test = np.load(root / "test.npy").astype(np.float32)
    label = np.load(root / "test_label.npy").astype(np.int64)
    columns = json.loads((root / "columns.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return train, test, label, columns, summary


def apply_edge_mode(edge: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros((*edge.shape[:2], 1), dtype=np.float32)
    if mode == "corr":
        return edge[..., [0]].astype(np.float32)
    if mode == "corr_lag":
        return edge[..., [0, 2]].astype(np.float32)
    if mode == "full":
        return edge.astype(np.float32)
    raise ValueError(f"Unsupported edge mode: {mode}")


def train_model(
    train: np.ndarray,
    edge_attr: torch.Tensor,
    cfg: Config,
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
    losses: list[float] = []
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
        epoch_loss = total / max(count, 1)
        losses.append(epoch_loss)
        print(f"epoch {epoch:02d}/{cfg.epochs} loss={epoch_loss:.6f}")
    return model, losses


def evaluate_node_scores(
    model: EdgeVectorGAT,
    data: np.ndarray,
    edge_attr: torch.Tensor,
    cfg: Config,
    device: torch.device,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    times = []
    node_scores = []
    node_errors = []
    attention_sum = None
    attention_count = 0
    with torch.no_grad():
        for start in range(0, len(data) - cfg.window, stride):
            x = torch.tensor(data[start : start + cfg.window].T[None, ...], dtype=torch.float32, device=device)
            y = data[start + cfg.window]
            pred, alpha = model(x, edge_attr)
            err = np.abs(pred.cpu().numpy()[0] - y)
            times.append(start + cfg.window)
            node_errors.append(err)
            node_scores.append(float(err.mean()))
            alpha_np = alpha.cpu().numpy()[0]
            attention_sum = alpha_np if attention_sum is None else attention_sum + alpha_np
            attention_count += 1
    attention = attention_sum / max(attention_count, 1)
    return np.asarray(times, dtype=int), np.asarray(node_scores), np.asarray(node_errors), attention


def metrics_for_score(labels: np.ndarray, score: np.ndarray, train_score: np.ndarray | None) -> dict[str, object]:
    if train_score is None:
        threshold, threshold_metrics = best_f1_threshold(labels, score)
        mode = "best_f1_on_test"
    else:
        threshold = threshold_from_train(train_score)
        threshold_metrics = point_metrics(labels, score > threshold)
        mode = "train_q995"
    best_threshold, best_metrics = best_f1_threshold(labels, score)
    curves = roc_pr_auc(labels, score)
    return {
        "threshold_mode": mode,
        "threshold": float(threshold),
        "metrics_at_threshold": threshold_metrics,
        "best_f1_threshold": float(best_threshold),
        "best_f1_metrics": best_metrics,
        "roc_auc": float(curves["roc_auc"]),
        "pr_auc": float(curves["pr_auc"]),
    }


def smooth_score(score: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return score.copy()
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(score, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply_consecutive_filter(pred: np.ndarray, min_consecutive: int) -> np.ndarray:
    if min_consecutive <= 1:
        return pred.astype(int)
    pred = pred.astype(int)
    filtered = np.zeros_like(pred)
    start = None
    for idx, value in enumerate(pred):
        if value and start is None:
            start = idx
        if (not value or idx == len(pred) - 1) and start is not None:
            end = idx if not value else idx + 1
            if end - start >= min_consecutive:
                filtered[start:end] = 1
            start = None
    return filtered


def validation_threshold_metrics(
    labels: np.ndarray,
    score: np.ndarray,
    val_fraction: float,
    smooth_window: int,
    min_consecutive: int,
) -> dict[str, object]:
    split = max(1, min(len(score) - 1, int(len(score) * val_fraction)))
    val_labels = labels[:split]
    test_labels = labels[split:]
    smoothed = smooth_score(score, smooth_window)
    val_score = smoothed[:split]
    test_score = smoothed[split:]
    threshold, val_metrics = best_f1_threshold(val_labels, val_score)
    raw_pred = (test_score > threshold).astype(int)
    filtered_pred = apply_consecutive_filter(raw_pred, min_consecutive)
    curves = roc_pr_auc(test_labels, test_score)
    return {
        "validation_fraction": val_fraction,
        "validation_points": int(split),
        "test_points": int(len(test_score)),
        "smooth_window": int(smooth_window),
        "min_consecutive": int(min_consecutive),
        "threshold_from_validation": float(threshold),
        "validation_best_f1_metrics": val_metrics,
        "test_metrics_raw_threshold": point_metrics(test_labels, raw_pred),
        "test_metrics_after_consecutive_filter": point_metrics(test_labels, filtered_pred),
        "test_roc_auc": float(curves["roc_auc"]),
        "test_pr_auc": float(curves["pr_auc"]),
    }


def save_plots(
    out_dir: Path,
    cfg: Config,
    times: np.ndarray,
    label: np.ndarray,
    node_norm: np.ndarray,
    edge_norm: np.ndarray,
    joint_score: np.ndarray,
    joint_threshold: float,
    node_errors: np.ndarray,
    attention: np.ndarray,
    columns: list[str],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.4), sharex=True)
    scores = [
        ("Node prediction error", node_norm, "#2f6f9f"),
        ("Relation degradation", edge_norm, "#9f6a2f"),
        ("Joint score", joint_score, "#7d3c98"),
    ]
    for ax, (title, score, color) in zip(axes, scores):
        ax.plot(times, score, color=color, lw=1.0)
        anomaly_mask = label[times] == 1
        ax.scatter(times[anomaly_mask], score[anomaly_mask], s=5, color="crimson", alpha=0.55)
        ax.set_title(title)
        ax.set_ylabel("Score")
    axes[-1].axhline(joint_threshold, color="orange", ls="--", label="Train q99.5 threshold")
    axes[-1].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{cfg.dataset}: node-edge joint anomaly scoring", y=0.995)
    fig.tight_layout()
    path = out_dir / "top_joint_scores.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["joint_scores"] = str(path)

    anomaly_nodes = node_errors[label[times] == 1]
    if len(anomaly_nodes):
        dim_score = anomaly_nodes.mean(axis=0)
    else:
        dim_score = node_errors.mean(axis=0)
    ranking = np.argsort(dim_score)[::-1]
    top = ranking[: min(20, len(ranking))]
    labels = [columns[i] if i < len(columns) else f"D{i + 1}" for i in top]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(range(len(top)), dim_score[top], color="#6c8ebf")
    ax.set_xticks(range(len(top)), labels, rotation=60, ha="right")
    ax.set_title(f"{cfg.dataset}: top anomalous dimensions")
    ax.set_ylabel("Mean node prediction error")
    fig.tight_layout()
    path = out_dir / "top_dimension_ranking.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["dimension_ranking"] = str(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(attention, cmap="viridis")
    ax.set_title(f"{cfg.dataset}: average Edge-Vector GAT attention")
    ax.set_xlabel("Destination dimension")
    ax.set_ylabel("Source dimension")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / "top_attention_heatmap.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths["attention_heatmap"] = str(path)
    return paths


def save_optimized_threshold_plot(
    out_dir: Path,
    cfg: Config,
    times: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
    threshold: float,
    split: int,
) -> str:
    smoothed = smooth_score(score, cfg.smooth_window)
    pred = apply_consecutive_filter((smoothed > threshold).astype(int), cfg.min_consecutive)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(times, smoothed, color="#7d3c98", lw=1.0, label="Smoothed joint score")
    ax.axhline(threshold, color="orange", ls="--", label="Validation threshold")
    ax.axvline(times[split], color="#555555", ls=":", label="Validation/Test split")
    anomaly_mask = labels == 1
    ax.scatter(times[anomaly_mask], smoothed[anomaly_mask], s=5, color="crimson", alpha=0.5, label="True anomaly")
    pred_mask = pred == 1
    ax.scatter(times[pred_mask], smoothed[pred_mask], s=7, color="#2c7a3f", alpha=0.5, label="Predicted alarm")
    ax.set_title(f"{cfg.dataset}: validation-threshold post-processed alarms")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / "top_validation_threshold_alarms.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default=Config.dataset)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--max-train-windows", type=int, default=Config.max_train_windows)
    parser.add_argument("--eval-stride", type=int, default=Config.eval_stride)
    parser.add_argument("--edge-weight", type=float, default=Config.edge_weight)
    parser.add_argument("--relation-window", type=int, default=Config.relation_window)
    parser.add_argument("--edge-mode", choices=["none", "corr", "corr_lag", "full"], default=Config.edge_mode)
    parser.add_argument("--no-relation-degradation", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=Config.smooth_window)
    parser.add_argument("--min-consecutive", type=int, default=Config.min_consecutive)
    parser.add_argument("--val-fraction", type=float, default=Config.val_fraction)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    cfg = Config(
        dataset=args.dataset,
        epochs=args.epochs,
        max_train_windows=args.max_train_windows,
        eval_stride=args.eval_stride,
        edge_weight=args.edge_weight,
        relation_window=args.relation_window,
        edge_mode=args.edge_mode,
        use_relation_degradation=not args.no_relation_degradation,
        smooth_window=args.smooth_window,
        min_consecutive=args.min_consecutive,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation_tag = "joint" if cfg.use_relation_degradation else "node"
    output_name = f"{cfg.edge_mode}_{relation_tag}{args.output_tag}"
    out_dir = args.project / "outputs" / "top_ready_relation_gat" / cfg.dataset / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, label, columns, dataset_summary = load_top_ready(args.project, cfg.dataset)
    print(f"loaded {cfg.dataset}: train={train_raw.shape}, test={test_raw.shape}, anomalies={int(label.sum())}")
    train, test = standardize(train_raw, test_raw)

    print("building normal edge vectors...")
    normal_edge_full = build_edge_vectors(train, cfg.max_lag)
    normal_edge = apply_edge_mode(normal_edge_full, cfg.edge_mode)
    edge_attr = torch.tensor(normal_edge, dtype=torch.float32, device=device)

    model, losses = train_model(train, edge_attr, cfg, device)
    train_times, train_node_score, train_node_errors, _ = evaluate_node_scores(
        model, train, edge_attr, cfg, device, stride=max(cfg.eval_stride * 2, 16)
    )
    times, node_score, node_errors, attention = evaluate_node_scores(
        model, test, edge_attr, cfg, device, stride=cfg.eval_stride
    )

    print("computing relation degradation scores...")
    train_edge_score, _ = relation_degradation_scores(
        train, normal_edge_full, train_times, cfg.relation_window, cfg.max_lag, cfg.important_edge_quantile
    )
    edge_score, _ = relation_degradation_scores(
        test, normal_edge_full, times, cfg.relation_window, cfg.max_lag, cfg.important_edge_quantile
    )

    node_norm_train = robust_minmax(train_node_score, train_node_score)
    node_norm = robust_minmax(train_node_score, node_score)
    edge_norm_train = robust_minmax(train_edge_score, train_edge_score)
    edge_norm = robust_minmax(train_edge_score, edge_score)
    if cfg.use_relation_degradation:
        joint_train = (1.0 - cfg.edge_weight) * node_norm_train + cfg.edge_weight * edge_norm_train
        joint_score = (1.0 - cfg.edge_weight) * node_norm + cfg.edge_weight * edge_norm
    else:
        joint_train = node_norm_train
        joint_score = node_norm
    joint_threshold = threshold_from_train(joint_train)

    np.save(out_dir / "times.npy", times)
    np.save(out_dir / "node_score.npy", node_norm)
    np.save(out_dir / "edge_score.npy", edge_norm)
    np.save(out_dir / "joint_score.npy", joint_score)
    np.save(out_dir / "node_errors.npy", node_errors)

    labels_aligned = label[times].astype(int)
    node_metrics = metrics_for_score(labels_aligned, node_norm, node_norm_train)
    edge_metrics = metrics_for_score(labels_aligned, edge_norm, edge_norm_train)
    joint_metrics = metrics_for_score(labels_aligned, joint_score, joint_train)
    optimized_threshold = {
        "node_only": validation_threshold_metrics(
            labels_aligned, node_norm, cfg.val_fraction, cfg.smooth_window, cfg.min_consecutive
        ),
        "edge_only": validation_threshold_metrics(
            labels_aligned, edge_norm, cfg.val_fraction, cfg.smooth_window, cfg.min_consecutive
        ),
        "joint_node_edge": validation_threshold_metrics(
            labels_aligned, joint_score, cfg.val_fraction, cfg.smooth_window, cfg.min_consecutive
        ),
    }
    figures = save_plots(
        out_dir,
        cfg,
        times,
        label,
        node_norm,
        edge_norm,
        joint_score,
        joint_threshold,
        node_errors,
        attention,
        columns,
    )
    split = max(1, min(len(joint_score) - 1, int(len(joint_score) * cfg.val_fraction)))
    figures["validation_threshold_alarms"] = save_optimized_threshold_plot(
        out_dir,
        cfg,
        times,
        labels_aligned,
        joint_score,
        optimized_threshold["joint_node_edge"]["threshold_from_validation"],
        split,
    )

    summary = {
        "experiment": "top_ready_relation_degradation_edge_vector_gat",
        "dataset": cfg.dataset,
        "device": str(device),
        "dataset_summary": dataset_summary,
        "window": cfg.window,
        "max_lag": cfg.max_lag,
        "hidden": cfg.hidden,
        "edge_hidden": cfg.edge_hidden,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "output_name": output_name,
        "max_train_windows": cfg.max_train_windows,
        "eval_stride": cfg.eval_stride,
        "relation_window": cfg.relation_window,
        "edge_weight": cfg.edge_weight,
        "important_edge_quantile": cfg.important_edge_quantile,
        "edge_mode": cfg.edge_mode,
        "use_relation_degradation": cfg.use_relation_degradation,
        "smooth_window": cfg.smooth_window,
        "min_consecutive": cfg.min_consecutive,
        "val_fraction": cfg.val_fraction,
        "final_loss": losses[-1] if losses else None,
        "node_only": node_metrics,
        "edge_only": edge_metrics,
        "joint_node_edge": joint_metrics,
        "optimized_threshold": optimized_threshold,
        "figures": figures,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
