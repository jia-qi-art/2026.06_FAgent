from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from finetune_service import build_finetune_dataset  # noqa: E402
from lora_finetune import (  # noqa: E402
    apply_lora,
    load_model_and_tokenizer,
    save_lora,
)


SYSTEM_PROMPT = (
    "你是工业时序异常诊断助手。请严格依据给定传感器证据，"
    "给出根因、支撑证据和有优先级的排查步骤，不要编造输入中不存在的数据。"
)
SENSOR_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9_])([12](?:B)?_[A-Za-z0-9_]+)")
ROOT_CAUSE_PATTERN = re.compile(r"根因定位于\s*([^\s，。]+)\s*传感器")


@dataclass
class Generation:
    text: str
    latency_seconds: float
    generated_tokens: int


class AssistantOnlyDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]], tokenizer, max_length: int):
        self.rows: list[dict[str, torch.Tensor]] = []
        tokenizer.padding_side = "right"
        for sample in samples:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["input"]},
            ]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": sample["output"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            tokens = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[: min(prompt_len, max_length)] = -100
            labels[attention_mask == 0] = -100
            if not torch.any(labels != -100):
                raise ValueError("max_length leaves no assistant answer tokens")
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project's quick LoRA benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("quick_config.json"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> Generation:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[1] :]
    return Generation(
        text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
        latency_seconds=elapsed,
        generated_tokens=int(generated.numel()),
    )


def lcs_length(a: str, b: str) -> int:
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for index, char_b in enumerate(b, start=1):
            if char_a == char_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = re.sub(r"\s+", "", prediction)
    ref = re.sub(r"\s+", "", reference)
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def sensor_ids(text: str) -> set[str]:
    return {match.group(1).rstrip("_") for match in SENSOR_PATTERN.finditer(text)}


def score_answer(sample: dict[str, str], prediction: str) -> dict[str, float]:
    reference = sample["output"]
    root_match = ROOT_CAUSE_PATTERN.search(reference)
    expected_root = root_match.group(1) if root_match else ""
    expected_ids = sensor_ids(reference)
    predicted_ids = sensor_ids(prediction)
    allowed_ids = sensor_ids(sample["input"]) | expected_ids
    hallucinated_ids = predicted_ids - allowed_ids
    sections = [
        "根因" in prediction,
        "证据" in prediction,
        "建议" in prediction or "排查" in prediction,
    ]
    action_terms = ("检查", "复核", "确认", "排查")
    return {
        "root_cause_accuracy": float(bool(expected_root and expected_root in prediction)),
        "evidence_coverage": (
            len(expected_ids & predicted_ids) / len(expected_ids) if expected_ids else 0.0
        ),
        "format_compliance": sum(sections) / len(sections),
        "action_coverage": min(1.0, sum(term in prediction for term in action_terms) / 3),
        "sensor_hallucination_rate": (
            len(hallucinated_ids) / len(predicted_ids) if predicted_ids else 0.0
        ),
        "rouge_l_f1": rouge_l_f1(prediction, reference),
    }


def evaluate_model(
    model_name: str,
    label: str,
    model,
    tokenizer,
    samples: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for index, sample in enumerate(samples):
        generation = generate(model, tokenizer, sample["input"], max_new_tokens)
        scores = score_answer(sample, generation.text)
        rows.append(
            {
                "sample_index": index,
                "model": label,
                "model_id": model_name,
                "prompt": sample["input"],
                "reference": sample["output"],
                "prediction": generation.text,
                "latency_seconds": generation.latency_seconds,
                "generated_tokens": generation.generated_tokens,
                **scores,
            }
        )
        print(f"[{label}] evaluated {index + 1}/{len(samples)}", flush=True)

    metric_names = [
        "root_cause_accuracy",
        "evidence_coverage",
        "format_compliance",
        "action_coverage",
        "sensor_hallucination_rate",
        "rouge_l_f1",
    ]
    metrics = {
        name: sum(float(row[name]) for row in rows) / len(rows) for name in metric_names
    }
    total_time = sum(float(row["latency_seconds"]) for row in rows)
    total_tokens = sum(int(row["generated_tokens"]) for row in rows)
    metrics.update(
        {
            "model": label,
            "model_id": model_name,
            "test_samples": len(rows),
            "avg_latency_seconds": total_time / len(rows),
            "tokens_per_second": total_tokens / total_time if total_time else 0.0,
            "peak_vram_gb": (
                torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            ),
        }
    )
    return metrics, rows


def train_lora(
    model, tokenizer, samples: list[dict[str, str]], config: dict[str, Any]
) -> tuple[Any, list[float]]:
    if getattr(model, "is_loaded_in_4bit", False):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = apply_lora(
        model,
        {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    )
    model.config.use_cache = False
    dataset = AssistantOnlyDataset(samples, tokenizer, int(config["max_length"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
    )
    accumulation = int(config["gradient_accumulation_steps"])
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(config["epochs"])):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(**batch)
                loss = output.loss / accumulation
            scaler.scale(loss).backward()
            epoch_loss += float(loss.detach()) * accumulation
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        average = epoch_loss / len(loader)
        losses.append(average)
        print(f"[train] epoch={epoch + 1} loss={average:.4f}", flush=True)
    return model, losses


def release_model(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_results(
    output_dir: Path,
    config: dict[str, Any],
    split: dict[str, Any],
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    train_losses: list[float],
    adapter_path: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "adapter_path": adapter_path,
        "train_losses": train_losses,
        "config": config,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    columns = [
        "model",
        "model_id",
        "root_cause_accuracy",
        "evidence_coverage",
        "format_compliance",
        "action_coverage",
        "sensor_hallucination_rate",
        "rouge_l_f1",
        "avg_latency_seconds",
        "tokens_per_second",
        "peak_vram_gb",
        "test_samples",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in metrics)

    markdown = [
        "# Qwen LoRA 快速对比实验",
        "",
        "> 自动评测基于项目现有的模板生成数据，仅衡量结构化证据归纳和诊断报告生成；不等同于独立专家根因评审。",
        "",
        "| 模型 | 根因提及率 ↑ | 证据覆盖率 ↑ | 格式合规率 ↑ | 操作覆盖率 ↑ | 传感器幻觉率 ↓ | ROUGE-L ↑ | 平均延迟(s) ↓ | tokens/s ↑ | 峰值显存(GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        markdown.append(
            f"| {row['model']} | {row['root_cause_accuracy']:.1%} | "
            f"{row['evidence_coverage']:.1%} | {row['format_compliance']:.1%} | "
            f"{row['action_coverage']:.1%} | {row['sensor_hallucination_rate']:.1%} | "
            f"{row['rouge_l_f1']:.3f} | {row['avg_latency_seconds']:.2f} | "
            f"{row['tokens_per_second']:.1f} | {row['peak_vram_gb']:.2f} |"
        )
    markdown.extend(
        [
            "",
            "## 训练信息",
            "",
            f"- 固定划分：train={split['train_size']}，test={split['test_size']}。",
            f"- LoRA epoch loss：{', '.join(f'{loss:.4f}' for loss in train_losses)}。",
            f"- Adapter：`{adapter_path}`。",
            "- 所有正式推理使用 greedy decoding，未使用人工评分或 LLM-as-judge。",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Quick benchmark requires CUDA for the 1.5B QLoRA run")

    samples = build_finetune_dataset(str(config["dataset"]))
    train_size = int(config["train_size"])
    if len(samples) <= train_size:
        raise ValueError(f"Need more than {train_size} samples, got {len(samples)}")
    train_samples = samples[:train_size]
    test_samples = samples[train_size:]
    split = {
        "dataset": config["dataset"],
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "train_indices": list(range(train_size)),
        "test_indices": list(range(train_size, len(samples))),
        "seed": config["seed"],
    }
    run_id = datetime.now().strftime("quick_%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "finetune_benchmark" / run_id
    adapter_dir = PROJECT_ROOT / "outputs" / "lora_adapters" / run_id
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    target_id = str(config["finetuned_model"])
    target_model, target_tokenizer = load_model_and_tokenizer(
        target_id, use_4bit=bool(config["use_4bit"]), local_only=True
    )
    metrics, predictions = evaluate_model(
        target_id,
        "Qwen2.5-1.5B（未微调）",
        target_model,
        target_tokenizer,
        test_samples,
        int(config["max_new_tokens"]),
    )
    all_metrics.append(metrics)
    all_predictions.extend(predictions)

    target_model, train_losses = train_lora(
        target_model, target_tokenizer, train_samples, config
    )
    adapter_path = save_lora(target_model, adapter_dir)
    metrics, predictions = evaluate_model(
        target_id,
        "Qwen2.5-1.5B + LoRA（本项目）",
        target_model,
        target_tokenizer,
        test_samples,
        int(config["max_new_tokens"]),
    )
    all_metrics.append(metrics)
    all_predictions.extend(predictions)
    del target_model, target_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    weak_id = str(config["weak_baseline_model"])
    weak_model, weak_tokenizer = load_model_and_tokenizer(
        weak_id, use_4bit=bool(config["use_4bit"]), local_only=True
    )
    metrics, predictions = evaluate_model(
        weak_id,
        "Qwen2.5-0.5B（弱基线）",
        weak_model,
        weak_tokenizer,
        test_samples,
        int(config["max_new_tokens"]),
    )
    all_metrics.append(metrics)
    all_predictions.extend(predictions)
    del weak_model, weak_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    write_results(
        output_dir,
        config,
        split,
        all_metrics,
        all_predictions,
        train_losses,
        adapter_path,
    )
    print(f"RESULT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
