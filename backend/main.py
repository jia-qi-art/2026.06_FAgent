from __future__ import annotations

import os
from typing import Literal
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

DEPS = Path(__file__).resolve().parent / ".deps"
if os.environ.get("FAGENT_USE_LOCAL_DEPS") == "1" and DEPS.exists():
    sys.path.insert(0, str(DEPS))

# ErnieBot 认证配置：优先读环境变量，否则使用默认值
os.environ.setdefault("ERNIE_API_KEY", "dPiR9tEhhwi9ioLWmFrj1ZFB")
os.environ.setdefault("ERNIE_SECRET_KEY", "nLYGWiCG7nwt76rq2Jplz7wigQYscYGO")

import json
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import load_env_file

load_env_file()

import data_service as svc
import finetune_service as ftsvc
import diagnosis_tasks


app = FastAPI(title="Relation-EVGAT Industrial Diagnosis Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    epochs: int = Field(default=1, ge=1, le=12)
    max_train_windows: int = Field(default=1000, ge=100, le=20000)
    eval_stride: int = Field(default=8, ge=1, le=64)
    edge_mode: Literal["none", "corr", "corr_lag", "full"] = "full"
    use_relation_degradation: bool = True


class AgentRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    question: str
    event_id: int | None = None
    image_base64: str | None = None


class DocxRequest(BaseModel):
    file_base64: str
    filename: str = ""


class OCRRequest(BaseModel):
    image_base64: str


class CrossModalRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    doc_text: str
    doc_info: dict | None = None
    event_id: int | None = None


@app.get("/api/health")
def health():
    return svc.health()


@app.get("/api/datasets")
def datasets():
    return {"datasets": svc.available_datasets()}


@app.post("/api/jobs/train")
def train(req: TrainRequest):
    try:
        job = svc.create_train_job(req.dataset, req.model_dump())
        return {"job_id": job.job_id, "status": job.status, "dataset": job.dataset}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    try:
        return svc.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


@app.get("/api/overview")
def overview(dataset: str = "WaDI_A2_ds10"):
    try:
        return svc.overview(dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/timeseries")
def timeseries(dataset: str = "WaDI_A2_ds10", start: int | None = None, end: int | None = None):
    try:
        return svc.timeseries(dataset, start, end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/relation-graph")
def relation_graph(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.relation_graph(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/root-cause")
def root_cause(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.root_cause(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/agent/ask")
def agent(req: AgentRequest):
    try:
        ocr_text: str | None = None
        if req.image_base64:
            ocr_result = svc.ocr_extract_image(req.image_base64)
            if ocr_result["success"]:
                ocr_text = ocr_result["text"]
        return svc.agent_answer(req.dataset, req.question, req.event_id, ocr_text=ocr_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/ocr/extract")
def ocr_extract(req: OCRRequest):
    """工厂文档图片 OCR 文字提取（DB + SVTR_LCNet）。"""
    result = svc.ocr_extract_image(req.image_base64)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "OCR failed"))
    return result


@app.post("/api/ocr/extract-info")
def ocr_extract_info(req: OCRRequest):
    """工厂文档图片 → OCR 提取文字 → ErnieBot 抽取关键信息。"""
    ocr_result = svc.ocr_extract_image(req.image_base64)
    if not ocr_result["success"]:
        raise HTTPException(status_code=400, detail=ocr_result.get("error", "OCR failed"))
    info_result = svc.extract_industrial_info(ocr_result["text"])
    return {
        "ocr_text": ocr_result["text"],
        "ocr_items_count": len(ocr_result.get("items", [])),
        "industrial_info": info_result.get("info", {}),
        "info_raw": info_result.get("raw", ""),
    }


@app.post("/api/document/extract")
def document_extract(req: DocxRequest):
    """DOCX 文档文字提取（python-docx）。"""
    result = svc.extract_docx_text(req.file_base64)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "extraction failed"))
    return result


@app.post("/api/document/extract-info")
def document_extract_info(req: DocxRequest):
    """
    文档文字提取（支持 DOCX / PDF）→ ErnieBot 抽取工业关键信息。
    同时自动保存到历史上传记录。
    """
    b64 = req.file_base64
    is_pdf = "application/pdf" in b64.split(",", 1)[0] if "," in b64 else b64.startswith("JVBER")
    if is_pdf:
        doc_result = svc.extract_pdf_text(b64)
    else:
        doc_result = svc.extract_docx_text(b64)
    if not doc_result["success"]:
        raise HTTPException(status_code=400, detail=doc_result.get("error", "extraction failed"))

    info_result = svc.extract_industrial_info(doc_result["text"])
    doc_id = svc.save_document_to_history(
        req.filename or "document",
        doc_result["text"],
        info_result.get("info", {}),
    )
    return {
        "doc_id": doc_id,
        "doc_text": doc_result["text"],
        "paragraphs_count": doc_result.get("paragraphs_count", 0),
        "industrial_info": info_result.get("info", {}),
        "info_raw": info_result.get("raw", ""),
    }


@app.get("/api/document/history")
def document_history():
    """返回历史上传文档列表。"""
    return {"documents": svc.document_history()}


@app.get("/api/document/history/{doc_id}")
def document_history_item(doc_id: str):
    """获取某个历史文档的完整内容。"""
    record = svc.get_document_from_history(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return record


@app.delete("/api/document/history/{doc_id}")
def document_history_delete(doc_id: str):
    """删除某个历史文档。"""
    deleted = svc.delete_document_from_history(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {"deleted": True, "doc_id": doc_id}


class UpdateAnalysisRequest(BaseModel):
    analysis: str


@app.put("/api/document/history/{doc_id}/analysis")
def document_history_update_analysis(doc_id: str, req: UpdateAnalysisRequest):
    """将跨模态分析结果保存到文档历史记录。"""
    updated = svc.update_document_analysis(doc_id, req.analysis)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {"updated": True, "doc_id": doc_id}


@app.post("/api/agent/cross-modal")
def cross_modal(req: CrossModalRequest):
    """
    跨模态关联分析：
    将质检文档信息与传感器异常检测数据关联，辅助缺陷溯源。
    """
    try:
        doc_info = req.doc_info or {}
        result = svc.cross_modal_analyze(req.dataset, req.doc_text, doc_info, req.event_id)
        if not result["success"]:
            status_code = 429 if "配额" in (result.get("error") or "") else 500
            raise HTTPException(status_code=status_code, detail=result.get("error", "analysis failed"))
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/report")
def report(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.report(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------- 诊断任务 ----------

class DiagnosisRequest(BaseModel):
    dataset: str = "WaDI_A2_ds10"
    event_id: int | None = None
    question: str = "为什么报警？"
    use_llm: bool = True


@app.post("/api/diagnosis/tasks")
def create_diagnosis(req: DiagnosisRequest):
    try:
        return diagnosis_tasks.create_task(req.dataset, req.event_id, req.question, req.use_llm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/diagnosis/tasks/{task_id}")
def get_diagnosis(task_id: str):
    try:
        return diagnosis_tasks.task_summary(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc


@app.get("/api/diagnosis/tasks/{task_id}/thinking/stream")
def diagnosis_thinking_stream(task_id: str):
    try:
        events = diagnosis_tasks.stream_events(task_id, "thinking")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc
    return StreamingResponse(
        diagnosis_tasks.format_sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/diagnosis/tasks/{task_id}/report/stream")
def diagnosis_report_stream(task_id: str):
    try:
        events = diagnosis_tasks.stream_events(task_id, "report")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc
    return StreamingResponse(
        diagnosis_tasks.format_sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/diagnosis/history")
def diagnosis_history():
    """返回诊断任务历史记录列表。"""
    return {"tasks": diagnosis_tasks.list_history()}


# ---------- 知识库 ----------

class KnowledgeUploadRequest(BaseModel):
    filename: str
    content: str


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/api/knowledge/documents")
def knowledge_documents():
    return svc.knowledge_documents()


@app.post("/api/knowledge/upload")
def knowledge_upload(req: KnowledgeUploadRequest):
    return svc.knowledge_upload(req.filename, req.content)


@app.post("/api/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest):
    return svc.knowledge_search(req.query, req.top_k)


# ---------- LoRA 微调 ----------

class FinetuneStartRequest(BaseModel):
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "WaDI_A2_ds10"
    lora_r: int = Field(default=8, ge=1, le=64)
    lora_alpha: float = Field(default=16.0, ge=1.0, le=64.0)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    target_modules: list[str] = Field(default=["q_proj", "k_proj", "v_proj", "o_proj"])
    epochs: int = Field(default=5, ge=1, le=20)
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2)
    use_4bit: bool = True


class FinetuneTestRequest(BaseModel):
    job_id: str
    dataset: str = "WaDI_A2_ds10"
    event_id: int = 1
    question: str = "为什么报警？请给出根因和排查步骤。"


@app.get("/api/finetune/status")
def finetune_status():
    """获取微调系统默认配置、可用模型列表和已保存的 adapter。"""
    return ftsvc.get_default_status()


@app.post("/api/finetune/start")
def finetune_start(req: FinetuneStartRequest):
    """启动 LoRA 微调任务（后台线程异步执行）。"""
    try:
        job = ftsvc.start_finetune(req.model_dump())
        return {
            "job_id": job.job_id,
            "status": job.status,
            "model_name": job.model_name,
            "dataset": job.dataset,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/finetune/jobs/{job_id}")
def finetune_job(job_id: str):
    """轮询微调任务进度（当前 epoch、loss 曲线等）。"""
    try:
        return ftsvc.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Finetune job not found: {job_id}") from exc


@app.get("/api/finetune/jobs/{job_id}/metrics")
def finetune_metrics(job_id: str):
    """获取微调完成后的评估指标。"""
    try:
        data = ftsvc.get_job(job_id)
        if data["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        return data.get("metrics", {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Finetune job not found: {job_id}") from exc


@app.post("/api/finetune/test")
def finetune_test(req: FinetuneTestRequest):
    """用微调后模型做推理测试，与 baseline 对比。"""
    try:
        return ftsvc.test_inference(req.job_id, req.dataset, req.event_id, req.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/finetune/models")
def finetune_models():
    """列出已保存的 LoRA adapter。"""
    from lora_finetune import list_saved_adapters
    return {"adapters": list_saved_adapters()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

