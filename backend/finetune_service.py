"""
微调服务：数据构建 + 训练任务管理 + 评估对比。

从 WaDI_A2_ds10 数据集的异常事件中构建 instruction-tuning 训练对，
将结构化传感器诊断数据（模态1）与自然语言诊断结论（模态2）进行跨模态对齐。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DATA_DIR = PROJECT_ROOT / "data" / "finetune"
FINETUNE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lora_adapters"

sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from lora_finetune import (
    DEFAULT_LORA_CONFIG,
    DEVICE_MAP,
    FinetuneJob,
    JOBS,
    _is_cuda_available,
    apply_lora,
    list_saved_adapters,
    load_model_and_tokenizer,
    run_inference,
    save_lora,
)
import data_service as svc


INSTRUCTION_TEMPLATE = """根据以下传感器异常检测数据，给出工业诊断结论和排查建议。

数据集: {dataset}
异常事件 #{event_id}
时间窗口: {time_window}
当前异常分数: {current_score:.2f}（阈值 {threshold:.2f}）
异常状态: {alert_status}

【根因候选 Top-5】
{candidates}

【关系退化边 Top-3】
{degraded_edges}

【运维知识库参考】
{knowledge_refs}

请分析以上数据，给出：
1. 最可能的根因定位
2. 支撑证据（节点误差 + 关系退化）
3. 优先级排查步骤"""

OUTPUT_TEMPLATE = """异常事件 #{event_id} 的根因定位于 {top_sensor} 传感器。

主要证据：
1) 节点预测误差显著升高（联合分 {top_score:.2f}），该传感器的读数偏离正常模式
2) 与相邻传感器的关系退化：{edge_evidence}，退化强度 {edge_strength:.2f}

建议排查步骤：
1. 优先检查 {top_sensor} 的原始读数、量程设定和采集回路状态
2. 复核 {edge_source} 与 {edge_target} 的联动关系和通信链路
3. 对照同一时间窗口的阀门、泵和控制指令记录
4. 将异常事件窗口数据导出提交给现场运维人员确认
5. 若传感器硬件正常，排查上游工艺参数波动（温度、压力、流量）"""


class FinetuneDataset(Dataset):
    """instruction-tuning 格式数据集。"""

    def __init__(self, samples: list[dict[str, str]], tokenizer, max_length: int = 1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        messages = [
            {"role": "system", "content": "你是工业时序异常诊断助手。请基于传感器检测数据，给出专业的根因分析和排查建议。"},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": tokens["input_ids"].squeeze(0).clone(),
        }


def build_finetune_dataset(dataset: str = "WaDI_A2_ds10") -> list[dict[str, str]]:
    """从 WaDI 数据集异常事件构建 instruction-tuning 训练样本。"""
    FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FINETUNE_DATA_DIR / f"{dataset}_train.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    samples = []
    try:
        ov = svc.overview(dataset)
        events = ov.get("events", [])
    except Exception:
        logger.warning("无法加载数据集 %s 的概览信息，使用空样本", dataset)
        return samples

    for event in events[:20]:  # 最多 20 个样本
        event_id = event.get("event_id", 1)
        try:
            rc = svc.root_cause(dataset, event_id)
            graph = svc.relation_graph(dataset, event_id)
        except Exception:
            continue

        candidates = rc.get("candidates", [])[:5]
        top = candidates[0] if candidates else {"name": "unknown", "score": 0.0}
        edges = graph.get("top_edges", [])[:3]
        top_edge = edges[0] if edges else {"source": "?", "target": "?", "degradation": 0.0}

        candidates_text = "\n".join(
            f"  #{c.get('rank', '?')} {c.get('name', '?')}: "
            f"联合分={c.get('score', 0):.3f}, "
            f"节点分={c.get('node_score', 0):.3f}, "
            f"边退化分={c.get('edge_score', 0):.3f}"
            for c in candidates
        )
        edges_text = "\n".join(
            f"  {e.get('source', '?')} → {e.get('target', '?')}: 退化强度={e.get('degradation', 0):.2f}"
            for e in edges
        )

        knowledge_refs = []
        try:
            from knowledge_service import search_knowledge
            query = f"{dataset} {top.get('name', '')} 异常 排查"
            result = search_knowledge(query, top_k=2)
            for hit in result.get("hits", []):
                knowledge_refs.append(f"  - {hit.get('title', '?')}: {hit.get('text', '')[:200]}")
        except Exception:
            pass
        knowledge_text = "\n".join(knowledge_refs) if knowledge_refs else "  （暂无匹配知识库条目）"

        inp = INSTRUCTION_TEMPLATE.format(
            dataset=dataset,
            event_id=event_id,
            time_window=f"{event.get('start', '?')}~{event.get('end', '?')}",
            current_score=ov.get("current_score", 0),
            threshold=ov.get("threshold", 0.5),
            alert_status="报警" if ov.get("alert") else "正常",
            candidates=candidates_text,
            degraded_edges=edges_text,
            knowledge_refs=knowledge_text,
        )
        out = OUTPUT_TEMPLATE.format(
            event_id=event_id,
            top_sensor=top.get("name", "unknown"),
            top_score=top.get("score", 0),
            edge_evidence=f"{top_edge.get('source', '?')} → {top_edge.get('target', '?')}",
            edge_strength=top_edge.get("degradation", 0),
            edge_source=top_edge.get("source", "?"),
            edge_target=top_edge.get("target", "?"),
        )

        samples.append({"input": inp, "output": out})

    cache_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("构建微调数据集 %s: %d 个样本", dataset, len(samples))
    return samples


def _run_finetune(job_id: str):
    """后台执行 LoRA 微调。"""
    job = JOBS.get(job_id)
    if not job:
        return

    try:
        # 阶段 1: 准备数据
        job.stage = "preparing_data"
        job.status = "running"
        job.updated_at = time.time()
        samples = build_finetune_dataset(job.dataset)
        if not samples:
            job.status = "failed"
            job.error = "无法构建微调数据集：无可用异常事件"
            return
        split = max(1, int(len(samples) * 0.8))
        train_samples = samples[:split]
        val_samples = samples[split:]

        # 阶段 2: 加载模型
        job.stage = "loading_model"
        job.updated_at = time.time()
        model, tokenizer = load_model_and_tokenizer(
            job.model_name,
            use_4bit=job.config.get("use_4bit", True),
        )

        # 阶段 3: 注入 LoRA
        job.stage = "applying_lora"
        job.updated_at = time.time()
        lora_cfg = {
            "r": job.config.get("lora_r", 8),
            "lora_alpha": job.config.get("lora_alpha", 16),
            "lora_dropout": job.config.get("lora_dropout", 0.1),
            "target_modules": job.config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        }
        model = apply_lora(model, lora_cfg)

        # 阶段 4: 训练
        job.stage = "training"
        job.total_epochs = job.config.get("epochs", 5)
        job.updated_at = time.time()

        train_dataset = FinetuneDataset(train_samples, tokenizer)
        eval_dataset = FinetuneDataset(val_samples, tokenizer) if val_samples else None

        adapter_dir = FINETUNE_OUTPUT_DIR / job.job_id
        use_cuda = _is_cuda_available()

        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False) if eval_dataset else None

        lr = job.config.get("learning_rate", 2e-4)
        accumulation_steps = 4 if use_cuda else 2
        total_steps = max(1, (len(train_loader) * job.total_epochs) // accumulation_steps)
        warmup_steps = max(1, total_steps // 10)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        scaler = torch.amp.GradScaler("cuda") if use_cuda else None
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            model = model.to(device)

        model.train()
        job.total_steps = total_steps
        job.current_step = 0
        job.current_epoch = 0

        for epoch in range(job.total_epochs):
            job.current_epoch = epoch + 1
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                if use_cuda:
                    with torch.amp.autocast("cuda"):
                        outputs = model(**batch)
                        loss = outputs.loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss / accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * accumulation_steps

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    if use_cuda:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    job.current_step += 1
                    avg_loss = epoch_loss / (step + 1)
                    job.train_loss.append(float(avg_loss))
                    job.updated_at = time.time()

            # end of epoch eval
            if eval_loader:
                model.eval()
                eval_total = 0.0
                with torch.no_grad():
                    for batch in eval_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        if use_cuda:
                            with torch.amp.autocast("cuda"):
                                out = model(**batch)
                        else:
                            out = model(**batch)
                        eval_total += out.loss.item()
                job.eval_loss.append(float(eval_total / len(eval_loader)))
                model.train()

        # 阶段 5: 保存
        job.stage = "saving"
        job.updated_at = time.time()
        adapter_path = save_lora(model, adapter_dir)
        job.adapter_path = adapter_path

        # 阶段 6: 评估
        job.stage = "evaluating"
        job.updated_at = time.time()

        if train_samples:
            test_prompt = train_samples[0]["input"]
            base_answer = ""
            if "base_model" not in job.metrics:
                try:
                    base_answer = run_inference(model, tokenizer, test_prompt)
                except Exception:
                    pass

        job.metrics = {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "final_train_loss": job.train_loss[-1] if job.train_loss else None,
            "final_eval_loss": job.eval_loss[-1] if job.eval_loss else None,
            "adapter_path": adapter_path,
            "test_inference": base_answer[:300] if base_answer else "",
            "trainable_params_info": "LoRA 注入于 q_proj/k_proj/v_proj/o_proj 交叉注意力投影层",
        }

        job.stage = "completed"
        job.status = "completed"
        job.updated_at = time.time()

    except Exception as exc:
        logger.exception("微调任务 %s 失败", job_id)
        job.status = "failed"
        job.stage = "failed"
        job.error = str(exc)
        job.updated_at = time.time()


def start_finetune(config: dict[str, Any]) -> FinetuneJob:
    """启动 LoRA 微调任务。"""
    job_id = uuid.uuid4().hex
    job = FinetuneJob(
        job_id=job_id,
        model_name=config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        dataset=config.get("dataset", "WaDI_A2_ds10"),
        config=config,
    )
    JOBS[job_id] = job

    thread = threading.Thread(target=_run_finetune, args=(job_id,), daemon=True)
    thread.start()
    return job


def get_job(job_id: str) -> dict[str, Any]:
    """获取微调任务状态。"""
    job = JOBS.get(job_id)
    if not job:
        raise KeyError(f"Finetune job not found: {job_id}")
    return {
        "job_id": job.job_id,
        "model_name": job.model_name,
        "dataset": job.dataset,
        "status": job.status,
        "stage": job.stage,
        "config": job.config,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "train_loss": job.train_loss,
        "eval_loss": job.eval_loss,
        "metrics": job.metrics,
        "error": job.error,
        "adapter_path": job.adapter_path,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def get_default_status() -> dict[str, Any]:
    """获取微调系统状态和默认配置。"""
    from lora_finetune import (
        AVAILABLE_MODELS, AVAILABLE_TARGET_MODULES, DEFAULT_LORA_CONFIG, get_model_download_info,
    )

    info = get_model_download_info()
    has_cuda = _is_cuda_available()

    return {
        "device": DEVICE_MAP,
        "cuda_available": has_cuda,
        "recommended_model": "Qwen/Qwen2.5-1.5B-Instruct" if has_cuda else "Qwen/Qwen2.5-0.5B-Instruct",
        "available_models": AVAILABLE_MODELS,
        "available_target_modules": AVAILABLE_TARGET_MODULES,
        "default_config": DEFAULT_LORA_CONFIG,
        "saved_adapters": list_saved_adapters(),
        "dataset_options": [{"id": "WaDI_A2_ds10", "name": "WaDI_A2_ds10 (WADI 水处理)"}],
        "loading_time_estimate": (
            "0.5B 模型加载约 5-10 秒；1.5B 模型加载约 5-10 分钟（CPU 模式不建议使用 1.5B）"
            if not has_cuda
            else "1.5B 模型加载约 10-30 秒（GPU 加速）"
        ),
        "model_cache": info,
    }


def test_inference(job_id: str, dataset: str, event_id: int, question: str) -> dict[str, Any]:
    """用微调后的模型做推理测试，并与 baseline 对比。"""
    job_record = get_job(job_id)
    adapter_path = job_record.get("adapter_path", "")
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    model_name = job_record.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")

    # 构建输入 prompt（复用 finetune 时的数据处理逻辑）
    try:
        ov = svc.overview(dataset)
        rc = svc.root_cause(dataset, event_id)
        graph = svc.relation_graph(dataset, event_id)
    except Exception as e:
        raise FileNotFoundError(f"数据集加载失败: {e}")

    candidates = rc.get("candidates", [])[:5]
    top = candidates[0] if candidates else {"name": "unknown", "score": 0.0}
    edges = graph.get("top_edges", [])[:3]
    top_edge = edges[0] if edges else {"source": "?", "target": "?", "degradation": 0.0}

    prompt = (
        f"数据集: {dataset}, 异常事件 #{event_id}\n"
        f"异常分数: {ov.get('current_score', 0):.2f} (阈值 {ov.get('threshold', 0.5):.2f})\n"
        f"根因候选首位: {top['name']} (联合分 {top.get('score', 0):.3f})\n"
        f"最严重关系退化边: {top_edge['source']} → {top_edge['target']} (退化强度 {top_edge.get('degradation', 0):.2f})\n"
        f"\n用户问题: {question}"
    )

    # 加载微调后模型
    base_model, tokenizer = load_model_and_tokenizer(model_name, use_4bit=True)
    from peft import PeftModel

    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)

    # 微调后推理
    finetuned_answer = run_inference(finetuned_model, tokenizer, prompt)

    # Baseline 推理（无 LoRA）
    baseline_answer = run_inference(base_model, tokenizer, prompt)

    return {
        "dataset": dataset,
        "event_id": event_id,
        "question": question,
        "prompt": prompt[:500],
        "finetuned_answer": finetuned_answer,
        "baseline_answer": baseline_answer,
        "model": model_name,
        "adapter_path": adapter_path,
    }
