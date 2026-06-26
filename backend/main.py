from __future__ import annotations

from typing import Literal
from pathlib import Path
import sys

DEPS = Path(__file__).resolve().parent / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import data_service as svc


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
        return svc.agent_answer(req.dataset, req.question, req.event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/report")
def report(dataset: str = "WaDI_A2_ds10", event_id: int | None = Query(default=None)):
    try:
        return svc.report(dataset, event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

