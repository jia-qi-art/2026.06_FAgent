from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent.tools import DiagnosisTools
from backend.config.settings import OUTPUT_ROOT
from backend.model import ModelFactory


DEFAULT_MODELS = [
    "rule-agent",
    "qwen-flash",
    "qwen3.5-flash",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]
DEFAULT_DATASETS = ["WaDI_A2_ds10", "SMD_machine_1_1"]
QUESTIONS = [
    ("diagnosis", "为什么报警？请说明最可能的根因和关键证据。"),
    ("operations", "请给出按优先级排列的现场排查步骤和运维建议。"),
]
REQUIRED_FIELDS = [
    "conclusion",
    "root_causes",
    "degraded_edges",
    "recommendations",
    "citations",
    "report_text",
]
SENSOR_RE = re.compile(r"\b(?:\d+[A-Z]?_[A-Z0-9_]+|TOTAL_[A-Z0-9_]+)\b")


@dataclass
class BenchmarkCase:
    case_id: str
    dataset: str
    event_id: int
    question_type: str
    question: str
    evidence: dict[str, Any]


def build_evidence(dataset: str, event_id: int, question: str) -> dict[str, Any]:
    tools = DiagnosisTools()
    event = tools.get_event_summary(dataset, event_id)
    root = tools.rank_root_causes(dataset, event_id)
    graph = tools.inspect_edge_degradation(dataset, event_id)
    report = tools.generate_report(dataset, event_id)
    candidates = root.get("candidates", [])[:5]
    top = candidates[0] if candidates else {"name": "unknown", "score": 0.0}
    edges = graph.get("top_edges", [])[:5]
    top_edge = edges[0] if edges else {"source": top["name"], "target": "unknown", "degradation": 0.0}
    query = f"{dataset} {top.get('name')} {top_edge.get('source')} {top_edge.get('target')} 异常 排查"
    knowledge = tools.retrieve_maintenance_knowledge(query)
    return {
        "question": question,
        "dataset": dataset,
        "event_id": event_id,
        "event_summary": {
            "score": event.get("current_score"),
            "threshold": event.get("threshold"),
            "time_window": report.get("time_window"),
        },
        "root_cause_candidates": candidates,
        "degraded_edges": edges,
        "knowledge_hits": [
            {
                "title": hit.get("title"),
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("score"),
                "text": hit.get("text", "")[:600],
            }
            for hit in knowledge.get("hits", [])[:3]
        ],
        "draft_report_sections": report.get("sections", []),
    }


def build_cases(datasets: list[str], events_per_dataset: int) -> list[BenchmarkCase]:
    from backend import data_service as svc

    cases: list[BenchmarkCase] = []
    for dataset in datasets:
        events = svc.overview(dataset).get("events", [])[:events_per_dataset]
        for event in events:
            event_id = int(event.get("event_id", 1))
            for question_type, question in QUESTIONS:
                cases.append(
                    BenchmarkCase(
                        case_id=f"{dataset}__e{event_id}__{question_type}",
                        dataset=dataset,
                        event_id=event_id,
                        question_type=question_type,
                        question=question,
                        evidence=build_evidence(dataset, event_id, question),
                    )
                )
    return cases


def prompt_for(case: BenchmarkCase) -> list[dict[str, str]]:
    schema = {
        field: [] if field in {"root_causes", "degraded_edges", "recommendations", "citations"} else "string"
        for field in REQUIRED_FIELDS
    }
    system = (
        "你是工业异常诊断模型。只能使用给定证据，关系退化只能作为排查线索，不能宣称严格因果。"
        "输出一个 JSON 对象，不要输出 Markdown 代码块。root_causes 包含 sensor、rank、reason；"
        "degraded_edges 包含 source、target、degradation；citations 只能引用给定知识条目的 title/chunk_id；"
        "report_text 是面向运维人员的完整中文报告。"
    )
    user = (
        "请回答问题并严格遵守此字段结构：\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n证据：\n"
        + json.dumps(case.evidence, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rule_output(case: BenchmarkCase) -> dict[str, Any]:
    evidence = case.evidence
    candidates = evidence["root_cause_candidates"]
    edges = evidence["degraded_edges"]
    top = candidates[0] if candidates else {"name": "unknown", "rank": 1}
    top_edge = edges[0] if edges else {"source": top["name"], "target": "unknown", "degradation": 0.0}
    recommendations = [
        f"核对 {top.get('name')} 的原始读数、量程和采集状态。",
        f"复核 {top_edge.get('source')} 到 {top_edge.get('target')} 的联动关系。",
        "对照异常窗口检查阀门、泵、控制指令和现场工况记录。",
    ]
    conclusion = f"事件 #{case.event_id} 的首要根因候选为 {top.get('name')}。"
    return {
        "conclusion": conclusion,
        "root_causes": [
            {"sensor": item.get("name"), "rank": item.get("rank"), "reason": "Relation-EVGAT 根因候选"}
            for item in candidates[:3]
        ],
        "degraded_edges": edges[:3],
        "recommendations": recommendations,
        "citations": [
            {"title": hit.get("title"), "chunk_id": hit.get("chunk_id")}
            for hit in evidence.get("knowledge_hits", [])
        ],
        "report_text": conclusion + " " + " ".join(recommendations),
    }


def parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def score_output(case: BenchmarkCase, parsed: dict[str, Any] | None, raw_text: str) -> dict[str, Any]:
    parsed = parsed or {}
    evidence = case.evidence
    allowed_sensors = {
        item.get("name") for item in evidence["root_cause_candidates"] if item.get("name")
    } | {
        value
        for edge in evidence["degraded_edges"]
        for value in (edge.get("source"), edge.get("target"))
        if value
    }
    allowed_citations = {
        (hit.get("title"), hit.get("chunk_id")) for hit in evidence.get("knowledge_hits", [])
    }
    report_text = str(parsed.get("report_text") or raw_text)
    predicted_sensors = set(SENSOR_RE.findall(json.dumps(parsed, ensure_ascii=False)))
    root_rows = parsed.get("root_causes") if isinstance(parsed.get("root_causes"), list) else []
    predicted_roots = {
        str(row.get("sensor")) for row in root_rows if isinstance(row, dict) and row.get("sensor")
    }
    candidate_names = [
        str(item.get("name")) for item in evidence["root_cause_candidates"] if item.get("name")
    ]
    citation_rows = parsed.get("citations") if isinstance(parsed.get("citations"), list) else []
    predicted_citations = {
        (row.get("title"), row.get("chunk_id")) for row in citation_rows if isinstance(row, dict)
    }
    required_present = sum(field in parsed and parsed[field] not in (None, "") for field in REQUIRED_FIELDS)
    recommendations = parsed.get("recommendations") if isinstance(parsed.get("recommendations"), list) else []
    return {
        "json_valid": bool(parsed),
        "schema_completeness": required_present / len(REQUIRED_FIELDS),
        "top1_root_mentioned": bool(candidate_names and candidate_names[0] in report_text),
        "candidate_coverage": len(predicted_roots & set(candidate_names)) / max(1, min(3, len(candidate_names))),
        "sensor_hallucination_count": len(predicted_sensors - allowed_sensors),
        "citation_precision": (
            len(predicted_citations & allowed_citations) / len(predicted_citations)
            if predicted_citations
            else (1.0 if not allowed_citations else 0.0)
        ),
        "recommendation_count": len(recommendations),
        "report_length": len(report_text),
    }


def run_model(case: BenchmarkCase, model: str, execute: bool) -> dict[str, Any]:
    started = time.time()
    if model == "rule-agent":
        parsed = rule_output(case)
        raw_text = json.dumps(parsed, ensure_ascii=False)
        metadata: dict[str, Any] = {"usage": {}}
        status, error = "ok", None
    elif not execute:
        return {
            "case_id": case.case_id,
            "model": model,
            "status": "skipped",
            "reason": "use --execute",
        }
    else:
        factory = ModelFactory()
        factory.config["llm_model"] = model
        try:
            response = factory.chat_with_metadata(prompt_for(case), timeout=120)
            raw_text = response["content"]
            parsed = parse_json_response(raw_text)
            metadata = response
            status, error = "ok", None
        except Exception as exc:
            raw_text, parsed, metadata = "", None, {}
            status, error = "failed", str(exc)
    return {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "event_id": case.event_id,
        "question_type": case.question_type,
        "model": model,
        "status": status,
        "error": error,
        "latency_ms": int((time.time() - started) * 1000),
        "usage": metadata.get("usage") or {},
        "raw_text": raw_text,
        "parsed": parsed,
        "scores": score_output(case, parsed, raw_text),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        completed = [row for row in model_rows if row.get("status") == "ok"]
        score_keys = [
            "json_valid",
            "schema_completeness",
            "top1_root_mentioned",
            "candidate_coverage",
            "sensor_hallucination_count",
            "citation_precision",
            "recommendation_count",
            "report_length",
        ]
        summary: dict[str, Any] = {
            "model": model,
            "total": len(model_rows),
            "completed": len(completed),
            "success_rate": len(completed) / max(1, len(model_rows)),
            "latency_ms_mean": (
                statistics.fmean(row["latency_ms"] for row in completed) if completed else None
            ),
        }
        for key in score_keys:
            values = [float(row["scores"][key]) for row in completed]
            summary[f"{key}_mean"] = statistics.fmean(values) if values else None
        summary["input_tokens"] = sum(
            int(row.get("usage", {}).get("prompt_tokens", 0)) for row in completed
        )
        summary["output_tokens"] = sum(
            int(row.get("usage", {}).get("completion_tokens", 0)) for row in completed
        )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare diagnosis generation models on grounded industrial cases."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--events-per-dataset", type=int, default=2)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call paid DashScope models. Omit for rule-only dry run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "model_benchmark",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.datasets, args.events_per_dataset)
    (args.output_dir / "cases.json").write_text(
        json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [run_model(case, model, args.execute) for case in cases for model in args.models]
    summaries = aggregate(rows)
    (args.output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "summary.csv", summaries)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "models": args.models,
                "execute": args.execute,
                "summary": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
