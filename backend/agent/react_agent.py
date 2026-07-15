from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from backend.agent.tools import DiagnosisTools
from backend.model import ModelFactory


class RuleDiagnosisAgent:
    """LangChain ReAct-style orchestration with optional DashScope LLM generation."""

    def __init__(self) -> None:
        self.tools = DiagnosisTools()
        self.model_factory = ModelFactory()
        self.model_status = self.model_factory.status().__dict__

    def execute(
        self,
        dataset: str,
        event_id: int | None,
        question: str,
        emit: Callable[[str, str, dict[str, Any] | None], None] | None = None,
        use_llm: bool | None = None,
    ) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []

        def call(name: str, fn, *args):
            started = time.time()
            if emit:
                emit("thinking", f"调用工具 {name}，收集诊断证据。", {"tool": name, "status": "running"})
            result = fn(*args)
            record = {"name": name, "status": "ok", "duration_ms": int((time.time() - started) * 1000)}
            tool_calls.append(record)
            if emit:
                emit("tool", f"{name} 完成。", record)
            return result

        event_summary = call("get_event_summary", self.tools.get_event_summary, dataset, event_id)
        root = call("rank_root_causes", self.tools.rank_root_causes, dataset, event_id)
        graph = call("inspect_edge_degradation", self.tools.inspect_edge_degradation, dataset, event_id)
        window = call("inspect_sensor_window", self.tools.inspect_sensor_window, dataset, event_id)
        top = root["candidates"][0] if root.get("candidates") else {"name": "unknown", "score": 0.0}
        edge = graph["top_edges"][0] if graph.get("top_edges") else {"source": top["name"], "target": "adjacent", "degradation": 0.0}
        query = f"{dataset} {top['name']} {edge['source']} {edge['target']} 异常 报警 排查 Relation-EVGAT"
        knowledge = call("retrieve_maintenance_knowledge", self.tools.retrieve_maintenance_knowledge, query)
        report = call("generate_report", self.tools.generate_report, dataset, event_id)
        qwen_evidence = call("inspect_qwen_evidence", self.tools.inspect_qwen_evidence)

        evidence_pack = {
            "question": question,
            "dataset": dataset,
            "event_summary": {
                "event_id": event_summary["event"].get("event_id", event_id or 1),
                "time_window": report.get("time_window"),
                "current_score": event_summary.get("current_score"),
                "threshold": event_summary.get("threshold"),
                "metrics": event_summary.get("metrics"),
            },
            "top_root_cause": top,
            "top_degraded_edge": edge,
            "top_candidates": root.get("candidates", [])[:5],
            "edge_vector_compare": graph.get("edge_vector_compare", []),
            "knowledge_hits": [
                {"title": hit.get("title"), "text": hit.get("text", "")[:500], "score": hit.get("score")}
                for hit in knowledge.get("hits", [])[:3]
            ],
            "sensor_window": {"start": window.get("start"), "end": window.get("end"), "sensors": window.get("sensors", [])},
            "draft_report_sections": report.get("sections", []),
            "qwen_evidence": qwen_evidence,
        }

        answer = self._rule_answer(question, event_summary, top, edge, knowledge, report)
        llm_error = None
        llm_metadata: dict[str, Any] = {}
        should_use_llm = self.model_status.get("configured") if use_llm is None else bool(use_llm)
        if should_use_llm and self.model_status.get("configured"):
            try:
                if emit:
                    emit("thinking", "调用 DashScope 大模型，根据工具证据生成最终中文诊断。", {"tool": "dashscope_chat", "status": "running"})
                llm_result = self._llm_answer(evidence_pack)
                answer = llm_result["content"]
                llm_metadata = {key: value for key, value in llm_result.items() if key != "content"}
                tool_calls.append({"name": "dashscope_chat", "status": "ok", "duration_ms": llm_metadata.get("duration_ms", 0), "model": llm_metadata.get("model")})
                if emit:
                    emit("tool", "DashScope 大模型回答生成完成。", {"name": "dashscope_chat", "status": "ok"})
            except Exception as exc:
                llm_error = str(exc)
                tool_calls.append({"name": "dashscope_chat", "status": "failed", "error": llm_error})
                if emit:
                    emit("tool", f"DashScope 调用失败，已回退规则回答：{llm_error}", {"name": "dashscope_chat", "status": "failed"})
        elif should_use_llm:
            llm_error = "LLM_API_KEY 未配置，已使用规则回答。"

        if emit:
            emit("report", answer, {"status": "completed", "llm_error": llm_error})

        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "report": report,
            "knowledge_hits": knowledge["hits"],
            "llm_error": llm_error,
            "used_llm": bool(llm_metadata),
            "fallback_used": not bool(llm_metadata),
            "provider": llm_metadata.get("provider", self.model_status.get("provider")),
            "model_name": llm_metadata.get("model", self.model_status.get("model")),
            "request_id": llm_metadata.get("request_id"),
            "duration_ms": llm_metadata.get("duration_ms"),
            "fusion": {
                "summary": answer,
                "confidence": "medium" if llm_metadata else "low",
                "sources": {
                    "relation_evgat": {"status": "available", "top_root_cause": top, "top_degraded_edge": edge},
                    "lora_qwen": qwen_evidence,
                    "knowledge": {"status": knowledge.get("status", {}), "hits": evidence_pack["knowledge_hits"]},
                },
                "citations": [
                    {"source": hit.get("title"), "chunk_id": hit.get("chunk_id"), "score": hit.get("score")}
                    for hit in knowledge.get("hits", [])[:3]
                ],
                "degraded": ["lora_qwen"] if qwen_evidence.get("status") != "available" else [],
            },
            "evidence": {
                "event_summary": event_summary,
                "root_cause": root,
                "relation_graph": graph,
                "window": {"start": window.get("start"), "end": window.get("end"), "sensors": window.get("sensors", [])},
            },
            "model": self.model_status,
        }

    def _rule_answer(self, question: str, event_summary: dict[str, Any], top: dict[str, Any], edge: dict[str, Any], knowledge: dict[str, Any], report: dict[str, Any]) -> str:
        advice = [
            f"优先核对 {top['name']} 的原始读数、量程和采集状态。",
            f"复核 {edge['source']} -> {edge['target']} 的联动关系，当前退化强度约 {edge['degradation']:.2f}。",
            "对照同一时间窗口的阀门、泵、控制指令和现场工况记录。",
        ]
        if knowledge.get("hits"):
            advice.append(f"知识库命中《{knowledge['hits'][0]['title']}》，建议结合该资料复核处置流程。")
        else:
            advice.append("知识库暂未命中高相关资料，可上传 SOP 或设备说明书增强诊断依据。")
        answer = (
            f"事件 #{event_summary['event'].get('event_id', 1)} 的联合异常分数为 "
            f"{event_summary['current_score']:.2f}，阈值为 {event_summary['threshold']:.2f}。"
            f"根因候选中 {top['name']} 排名第一，主要证据来自节点误差和关系退化边 "
            f"{edge['source']} -> {edge['target']}。"
        )
        if "步骤" in question or "排查" in question or "checklist" in question.lower():
            answer = "建议排查步骤：" + " ".join(f"{idx + 1}. {item}" for idx, item in enumerate(advice))
        elif "报告" in question or "report" in question.lower():
            answer = "已生成诊断报告：" + " ".join(section["body"] for section in report["sections"])
        return answer

    def _llm_answer(self, evidence_pack: dict[str, Any]) -> dict[str, Any]:
        system = (
            "你是工业时序异常诊断 Agent。必须基于工具返回的 Relation-EVGAT 异常分数、根因候选、"
            "关系退化边、知识库资料和报告结构回答。关系退化证据只能表述为诊断证据或排查线索，"
            "不要宣称严格因果链。回答要中文、具体、适合答辩演示，包含结论、证据、排查步骤。"
        )
        user = "请根据以下工具证据回答用户问题。\n" + json.dumps(evidence_pack, ensure_ascii=False, indent=2)
        return self.model_factory.chat_with_metadata([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
