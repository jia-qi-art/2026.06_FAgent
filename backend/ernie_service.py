"""
ErnieBot（百度文心大模型）工业诊断对话模块。
参考论文第四部分：
- 基于工业语料预训练，提取文档文本中的关键信息（工艺参数、缺陷描述等）
- 实现语义理解、智能问答、跨模态关联
- 将 OCR 提取的文档信息与传感器异常数据关联，辅助缺陷溯源与成因分析

注：已从 ErnieBot API 迁移至 DashScope (ModelFactory)，保留接口兼容性。
"""
from __future__ import annotations
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_INDUSTRIAL_SYSTEM_PROMPT = """你是一个工业时序异常诊断助手，负责根据 Relation-EVGAT 模型的检测结果为用户解答问题。

你需要结合以下数据来回答问题：
- 异常事件概览（异常分数、阈值、告警状态）
- 根因候选排序（Top-K 传感器/变量，含节点误差分和关系退化分）
- 传感器关系图（拓扑结构和退化边）

回答要求：
1. 用简洁专业的中文回答，避免冗长。
2. 涉及数据时引用具体数值。
3. 如果用户问排查步骤，给出有序、可操作的建议。
4. 如果数据不足以回答，如实说明并建议补充哪些信息。
5. 判断用户问题是否与工业时序异常诊断相关。相关则直接回答不加后缀；不相关则自然回答后另起一段：本系统主要服务于工业时序异常诊断，可以向我提出相关时序异常问题。"""

_INFO_EXTRACTION_PROMPT = """你是一个工业文档信息抽取助手。请从以下工业文档文本中，提取关键的工艺参数和缺陷描述信息。

请以 JSON 格式返回，包含以下字段（如未提取到则填空字符串）：
{
  "process_params": {"参数名": "值", ...},
  "defect_type": "缺陷类型",
  "defect_location": "缺陷位置",
  "defect_severity": "严重程度",
  "material": "材料/部件名称",
  "batch_number": "批次号",
  "inspection_result": "检测结论",
  "summary": "文档内容摘要（一句话）"
}

只返回 JSON，不要有其他文字。"""

_CROSS_MODAL_PROMPT = """你是一个工业缺陷溯源分析助手。请结合以下两类信息进行跨模态关联分析：

1. **文档 提取的质检文档信息**：包含工艺参数、缺陷描述、检测结论等
2. **传感器时序异常检测数据**：包含异常传感器排名、关系退化信息、异常时间窗口

请分析：
- 质检文档中记录的缺陷/异常，与传感器检测到的异常变量之间是否存在关联
- 工艺参数的偏差是否可能是导致传感器异常的原因
- 给出综合的缺陷溯源结论和工艺改进建议

用简洁专业的中文回答。"""


def _build_context_prompt(question: str, context: dict[str, Any]) -> str:
    """将诊断上下文构建为 prompt。"""
    parts: list[str] = []

    overview = context.get("overview", {})
    if overview:
        parts.append("【异常概览】")
        parts.append(f"数据集: {overview.get('dataset', 'N/A')}")
        parts.append(f"当前异常分数: {overview.get('current_score', 'N/A')}")
        parts.append(f"告警阈值: {overview.get('threshold', 'N/A')}")
        parts.append(f"是否告警: {'是' if overview.get('alert') else '否'}")
        events = overview.get("events", [])
        if events:
            parts.append(f"历史异常事件数: {len(events)}")

    root_cause = context.get("root_cause", {})
    if root_cause:
        parts.append("")
        parts.append("【根因候选】")
        event = root_cause.get("event", {})
        if event:
            parts.append(
                f"当前事件ID: {event.get('event_id', 'N/A')}, "
                f"时间窗: {event.get('start', '?')}~{event.get('end', '?')}"
            )
        candidates = root_cause.get("candidates", [])
        for c in candidates[:5]:
            parts.append(
                f"  #{c.get('rank')} {c.get('name')}: "
                f"联合分={c.get('score', 0):.3f}, "
                f"节点分={c.get('node_score', 0):.3f}, "
                f"边退化分={c.get('edge_score', 0):.3f}"
            )

        evidence = root_cause.get("evidence", [])
        if evidence:
            parts.append("诊断证据:")
            for ev in evidence:
                parts.append(f"  [{ev.get('severity', '?')}] {ev.get('label')}: {ev.get('value')}")

    relation_graph = context.get("relation_graph", {})
    if relation_graph:
        parts.append("")
        parts.append("【关系退化信息】")
        top_edges = relation_graph.get("top_edges", [])
        for e in top_edges[:3]:
            parts.append(
                f"  {e.get('source')} → {e.get('target')}: "
                f"退化强度={e.get('degradation', 0):.2f}"
            )

    report = context.get("report", {})
    if report:
        parts.append("")
        parts.append("【诊断报告摘要】")
        for section in report.get("sections", []):
            parts.append(f"  {section.get('title')}: {section.get('body')}")

    parts.append("")
    parts.append(f"用户问题: {question}")
    parts.append("")
    parts.append("注意：判断问题是否与工业诊断相关。相关则直接回答；不相关则先自然回答，再换行写：本系统主要服务于工业时序异常诊断，可以向我提出相关时序异常问题。")
    return "\n".join(parts)


class ErnieService:
    """对话服务，包含信息抽取、智能问答、跨模态关联。已迁移至 DashScope。"""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        access_token: str | None = None,
        model: str = "qwen-plus",
    ):
        self._model = model

    def chat(
            self,
            question: str,
            context: dict[str, Any] | None = None,
            temperature: float = 0.3,
        ) -> dict[str, Any]:
        """智能问答：结合诊断上下文生成自然语言回答。"""
        from backend.model.factory import ModelFactory

        user_prompt = _build_context_prompt(question, context or {})
        try:
            llm = ModelFactory()
            answer = llm.chat([
                {"role": "system", "content": _INDUSTRIAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ], temperature=temperature, timeout=60)
            return {
                "success": True,
                "answer": answer,
                "model": llm.status().model,
                "error": None,
            }
        except Exception as exc:
            logger.exception("LLM 调用失败")
            return {"success": False, "answer": "", "model": "dashscope", "error": str(exc)}

    def extract_industrial_info(self, ocr_text: str) -> dict[str, Any]:
        """从文档文本中抽取工业生产关键信息。"""
        from backend.model.factory import ModelFactory

        user_prompt = f"以下是工业文档识别结果，请提取关键信息：\n\n{ocr_text[:8000]}"
        try:
            llm = ModelFactory()
            raw = llm.chat([
                {"role": "system", "content": _INFO_EXTRACTION_PROMPT},
                {"role": "user", "content": user_prompt},
            ], temperature=0.1, timeout=60)
            info = self._parse_json_response(raw)
            return {"success": True, "info": info, "raw": raw, "error": None}
        except Exception as exc:
            logger.exception("信息抽取失败")
            return {"success": False, "info": {}, "error": str(exc)}

    def cross_modal_analyze(
        self,
        ocr_text: str,
        ocr_info: dict[str, Any],
        diagnosis_context: dict[str, Any],
    ) -> dict[str, Any]:
        """跨模态关联分析：将质检文档与传感器异常数据关联。"""
        from backend.model.factory import ModelFactory

        diagnosis_text = _build_context_prompt("跨模态关联分析", diagnosis_context)
        user_prompt = (
            f"【质检文档内容】\n{ocr_text}\n\n"
            f"【抽取的关键信息】\n{json.dumps(ocr_info, ensure_ascii=False, indent=2)}\n\n"
            f"【传感器异常检测数据】\n{diagnosis_text}"
        )

        try:
            llm = ModelFactory()
            analysis = llm.chat([
                {"role": "system", "content": _CROSS_MODAL_PROMPT},
                {"role": "user", "content": user_prompt},
            ], temperature=0.3, timeout=60)
            return {
                "success": True,
                "analysis": analysis,
                "model": llm.status().model,
                "error": None,
            }
        except Exception as exc:
            logger.exception("跨模态分析失败")
            return {"success": False, "analysis": "", "error": str(exc)}

    def generate_report(self, dataset: str, context: dict[str, Any]) -> dict[str, Any]:
        """生成诊断报告。"""
        prompt = (
            f"请为数据集 {dataset} 的异常事件生成一份简洁的诊断报告，"
            f"包含：异常概况、根因分析、关系退化、运维建议四个部分。"
        )
        return self.chat(prompt, context)

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """从 LLM 回复中解析 JSON。"""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:]) if len(lines) > 1 else raw
            if raw.endswith("```"):
                raw = raw[:-3]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            result: dict[str, Any] = {}
            patterns = {
                "defect_type": r"缺陷类型[：:]\s*(\S+)",
                "defect_location": r"缺陷位置[：:]\s*(\S+)",
                "defect_severity": r"严重程度[：:]\s*(\S+)",
                "material": r"材料[：:]\s*(\S+)",
                "batch_number": r"批次号[：:]\s*(\S+)",
                "inspection_result": r"检测结论[：:]\s*(.+?)(?:\n|$)",
                "summary": r"摘要[：:]\s*(.+?)(?:\n|$)",
            }
            for key, pat in patterns.items():
                m = re.search(pat, raw)
                if m:
                    result[key] = m.group(1).strip()
            param_pattern = re.compile(r"(\S+)[=＝](\S+)")
            params: dict[str, str] = {}
            for m in param_pattern.finditer(raw):
                params[m.group(1)] = m.group(2)
            if params:
                result["process_params"] = params
            return result if result else {"raw": raw}


_ERNIE_INSTANCE: ErnieService | None = None


def get_ernie() -> ErnieService:
    """懒加载全局实例。"""
    global _ERNIE_INSTANCE
    if _ERNIE_INSTANCE is None:
        logger.info("对话服务初始化中...")
        _ERNIE_INSTANCE = ErnieService()
        logger.info("对话服务初始化完成")
    return _ERNIE_INSTANCE
