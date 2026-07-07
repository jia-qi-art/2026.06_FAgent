"""
Qwen 开源大模型 LoRA 微调模块。

在 Qwen 的交叉注意力投影层 (q_proj/k_proj/v_proj/o_proj) 注入低秩适配器，
加强传感器异常数据（结构化诊断信息）与自然语言诊断结论之间的跨模态对齐。

模型下载来源：
- 优先从 HuggingFace Hub 下载（自动使用 http_proxy/https_proxy 代理）
- 可通过 HF_ENDPOINT 环境变量设置镜像站（如 https://hf-mirror.com）
- 如有本地模型，设置 HF_HOME 或 MODEL_PATH 指向本地目录
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B (CPU推荐)", "size": "0.5B", "cpu_ok": True},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen2.5-1.5B (GPU推荐)", "size": "1.5B", "cpu_ok": False},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B (GPU高精度)", "size": "7B", "cpu_ok": False},
]

AVAILABLE_TARGET_MODULES = [
    {"id": "q_proj", "label": "Q 投影", "desc": "Query — 查询向量投影"},
    {"id": "k_proj", "label": "K 投影", "desc": "Key — 键向量投影"},
    {"id": "v_proj", "label": "V 投影", "desc": "Value — 值向量投影"},
    {"id": "o_proj", "label": "O 投影", "desc": "Output — 注意力输出投影"},
]

DEFAULT_LORA_CONFIG: dict[str, Any] = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
}


@dataclass
class FinetuneJob:
    job_id: str
    model_name: str
    dataset: str
    status: str = "queued"
    stage: str = "preparing"
    config: dict[str, Any] = field(default_factory=dict)
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    train_loss: list[float] = field(default_factory=list)
    eval_loss: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    adapter_path: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


JOBS: dict[str, FinetuneJob] = {}

DEVICE_MAP = "auto" if torch.cuda.is_available() else "cpu"

REQUIRED_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

MODEL_DOWNLOAD_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "*.py",
)


def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


def _fix_proxy_env():
    """修复代理环境变量。

    清除 ALL_PROXY/all_proxy（通常为 socks:// 格式，urllib3 不支持），
    保留 http_proxy/https_proxy（标准 HTTP 代理，用于访问 HuggingFace）。
    """
    removed = {}
    for key in ("ALL_PROXY", "all_proxy"):
        if key in os.environ:
            removed[key] = os.environ.pop(key)
    if removed:
        logger.info("已临时清除 socks 代理变量: %s，保留 http_proxy/https_proxy", list(removed.keys()))
    return removed


def _restore_proxy_env(removed: dict[str, str]):
    """恢复被清除的代理变量。"""
    for key, value in removed.items():
        os.environ[key] = value


def _cached_file_path(model_name: str, filename: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    try:
        path = try_to_load_from_cache(
            repo_id=model_name,
            filename=filename,
        )
    except Exception:
        return None

    if not path:
        return None

    file_path = Path(path)
    if file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0:
        return file_path
    return None


def _cached_weights_complete(model_name: str) -> tuple[bool, str]:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        cached = _cached_file_path(model_name, filename)
        if cached:
            return True, str(cached)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = _cached_file_path(model_name, index_name)
        if not index_path:
            continue
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(index_data.get("weight_map", {}).values())
        except Exception:
            return False, str(index_path)

        if shard_names and all(_cached_file_path(model_name, shard) for shard in shard_names):
            return True, str(index_path)
        return False, str(index_path)

    return False, ""


def _get_model_cache_status(model_name: str) -> dict[str, Any]:
    """检查模型本地缓存是否完整。"""
    config_path = _cached_file_path(model_name, "config.json")
    has_weights, weight_path = _cached_weights_complete(model_name)

    return {
        "model_name": model_name,
        "complete": bool(config_path and has_weights),
        "has_config": config_path is not None,
        "has_weights": has_weights,
        "config_path": str(config_path) if config_path else "",
        "weight_path": weight_path,
    }


def _check_model_cached(model_name: str) -> bool:
    """检查模型是否已在本地完整缓存。"""
    return bool(_get_model_cache_status(model_name)["complete"])


def _download_model_snapshot(model_name: str, local_only: bool) -> str | None:
    """下载或定位完整模型快照。"""
    if local_only:
        return None

    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    errors = []
    endpoints = [endpoint]
    if endpoint.rstrip("/") != "https://huggingface.co":
        endpoints.append("https://huggingface.co")

    for current_endpoint in endpoints:
        try:
            return snapshot_download(
                repo_id=model_name,
                allow_patterns=list(MODEL_DOWNLOAD_PATTERNS),
                local_files_only=False,
                endpoint=current_endpoint,
            )
        except Exception as exc:
            logger.warning("模型 %s 从 %s 自动下载失败: %s", model_name, current_endpoint, exc)
            errors.append(f"{current_endpoint}: {exc}")

    error_text = "\n".join(errors)
    raise RuntimeError(
        f"模型 {model_name} 权重未完整下载，且自动下载失败。\n"
        f"已尝试 endpoint: {', '.join(endpoints)}。\n"
        f"最后错误：\n{error_text}\n"
        f"你也可以切换到已完整缓存的 Qwen/Qwen2.5-0.5B-Instruct。"
    )


def _check_download_metadata(model_name: str, endpoint: str) -> None:
    """只检查远端元数据，不下载大文件。"""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_name,
            allow_patterns=["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"],
            local_files_only=False,
            endpoint=endpoint,
            dry_run=True,
        )
    except Exception as exc:
        raise RuntimeError(f"{endpoint} 元数据检查失败: {exc}") from exc


def _resolve_model_source(model_name: str, local_only: bool) -> str:
    """返回可传给 from_pretrained 的模型来源。"""
    cache_status = _get_model_cache_status(model_name)
    if cache_status["complete"]:
        logger.info("模型 %s 已在本地完整缓存，直接加载。", model_name)
        return model_name

    if cache_status["has_config"] and not cache_status["has_weights"]:
        logger.warning(
            "模型 %s 本地缓存不完整：已有 config/tokenizer，但缺少 model.safetensors 或 pytorch_model.bin。",
            model_name,
        )
    else:
        model_size = "约 1GB (0.5B) / 3GB (1.5B) / 15GB (7B)"
        logger.info(
            "模型 %s 未完整缓存，将从 HuggingFace Hub 下载 (%s)。首次下载需要几分钟，请耐心等待。",
            model_name,
            model_size,
        )

    if local_only:
        raise RuntimeError(
            f"模型 {model_name} 未在本地完整缓存中找到，且 local_only=True。\n"
            f"已找到 config={cache_status['has_config']}，weights={cache_status['has_weights']}。\n"
            f"请先下载完整模型权重，缓存路径通常为: "
            f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
        )

    snapshot_path = _download_model_snapshot(model_name, local_only=False)
    refreshed_status = _get_model_cache_status(model_name)
    if not refreshed_status["complete"]:
        raise RuntimeError(
            f"模型 {model_name} 下载后仍不完整：缺少 model.safetensors 或 pytorch_model.bin。\n"
            f"请清理该模型的不完整缓存后重试，或手动下载完整权重。"
        )
    return snapshot_path or model_name


def _from_pretrained_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """transformers 5 使用 dtype；旧版本仍使用 torch_dtype。"""
    try:
        import transformers

        major = int(transformers.__version__.split(".", 1)[0])
    except Exception:
        major = 4
    if major >= 5:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    local_only: bool = False,
):
    """加载 Qwen 模型和分词器。

    Args:
        model_name: HuggingFace 模型 ID (如 Qwen/Qwen2.5-1.5B-Instruct)
        use_4bit: 是否使用 4-bit QLoRA 量化
        local_only: 仅使用本地缓存，不尝试下载

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 修复 socks 代理问题
    proxy_backup = _fix_proxy_env()

    try:
        # 设置 HF 镜像（国内用户可通过环境变量 HF_ENDPOINT 加速）
        hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if hf_endpoint:
            logger.info("使用 HF 镜像: %s", hf_endpoint)

        model_source = _resolve_model_source(model_name, local_only=local_only)

        logger.info("正在加载 tokenizer: %s", model_source)
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=local_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("正在加载模型: %s (use_4bit=%s, device=%s)", model_source, use_4bit, DEVICE_MAP)

        if use_4bit and _is_cuda_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                quantization_config=bnb_config,
                device_map=DEVICE_MAP,
                trust_remote_code=True,
                local_files_only=local_only,
                **_from_pretrained_dtype_kwargs(torch.float16),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                device_map=DEVICE_MAP if _is_cuda_available() else None,
                trust_remote_code=True,
                local_files_only=local_only,
                **_from_pretrained_dtype_kwargs(torch.float16 if _is_cuda_available() else torch.float32),
            )

        logger.info("模型加载成功: %s", model_name)
        return model, tokenizer

    finally:
        _restore_proxy_env(proxy_backup)


def apply_lora(model, config: dict[str, Any]):
    """在 Qwen 交叉注意力投影层注入 LoRA 适配器。"""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=config.get("r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA 注入完成: 可训练参数=%s (总参数=%s, 训练比例=%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total if total else 0,
    )
    return model


def save_lora(model, output_dir: Path) -> str:
    """保存 LoRA adapter 权重。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logger.info("LoRA adapter 已保存到: %s", output_dir)
    return str(output_dir)


def load_lora(model, adapter_path: str):
    """加载已保存的 LoRA adapter。"""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """用微调后模型做推理。"""
    proxy_backup = _fix_proxy_env()
    try:
        messages = [
            {"role": "system", "content": "你是工业时序异常诊断助手，基于传感器检测数据给出专业诊断和建议。"},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    finally:
        _restore_proxy_env(proxy_backup)


def list_saved_adapters() -> list[dict[str, Any]]:
    """列出已保存的 LoRA adapter。"""
    adapters_dir = PROJECT_ROOT / "outputs" / "lora_adapters"
    if not adapters_dir.exists():
        return []
    result = []
    for d in sorted(adapters_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "adapter_config.json").exists():
            cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
            result.append({
                "id": d.name,
                "path": str(d),
                "base_model": cfg.get("base_model_name_or_path", ""),
                "r": cfg.get("r", "?"),
                "lora_alpha": cfg.get("lora_alpha", "?"),
                "target_modules": cfg.get("target_modules", []),
                "updated_at": d.stat().st_mtime,
            })
    return result


def get_model_download_info() -> dict[str, Any]:
    """获取模型下载状态和建议。"""
    from huggingface_hub import scan_cache_dir

    cached_models = []
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            for rev in repo.revisions:
                cached_models.append({
                    "repo_id": repo.repo_id,
                    "size_on_disk": rev.size_on_disk_str,
                    "last_modified": rev.last_modified,
                    "complete": _check_model_cached(repo.repo_id),
                })
                break
    except Exception:
        pass

    qwen_cache_status = {
        model["id"]: _get_model_cache_status(model["id"])
        for model in AVAILABLE_MODELS
    }

    return {
        "device": DEVICE_MAP,
        "cuda_available": _is_cuda_available(),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "hf_home": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub")),
        "cached_models": cached_models,
        "qwen_cache_status": qwen_cache_status,
        "has_qwen_1_5b": qwen_cache_status["Qwen/Qwen2.5-1.5B-Instruct"]["complete"],
        "has_qwen_7b": qwen_cache_status["Qwen/Qwen2.5-7B-Instruct"]["complete"],
        "download_tip": (
            "如果下载速度慢,设置环境变量加速：\n"
            "  $env:HF_ENDPOINT='https://hf-mirror.com'  # PowerShell\n"
            "  hf download Qwen/Qwen2.5-1.5B-Instruct\n"
            "模型存放路径: " + str(Path.home() / ".cache" / "huggingface" / "hub")
        ),
    }
