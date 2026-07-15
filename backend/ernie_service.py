"""
ErnieBot（百度文心大模型）工业诊断对话模块。
参考论文第四部分：
- 基于工业语料预训练，提取文档文本中的关键信息（工艺参数、缺陷描述等）
- 实现语义理解、智能问答、跨模态关联
- 将 OCR 提取的文档信息与传感器异常数据关联，辅助缺陷溯源与成因分析
"""
from __future__ import annotations
import threading
import time
import json
import logging
import os
from typing import Any

try:
    from erniebot.errors import RequestLimitError, RateLimitError
except ImportError:
    RequestLimitError = Exception
    RateLimitError = Exception

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
    """ErnieBot 对话服务，包含信息抽取、智能问答、跨模态关联。"""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        access_token: str | None = None,
        #model: str = "ernie-4.5-VL-8k-preview"
        #model: str = "ernie-speed-8k"
        model: str = "ernie-3.5",
    ):
        try:
            import erniebot
        except ImportError:
            raise ImportError("erniebot 未安装，请执行: pip install erniebot")

        api_key = api_key or os.environ.get("ERNIE_API_KEY", "")
        secret_key = secret_key or os.environ.get("ERNIE_SECRET_KEY", "")
        access_token = access_token or os.environ.get("ERNIE_ACCESS_TOKEN", "")

        if access_token:
            erniebot.api_type = "aistudio"
            erniebot.access_token = access_token
        elif api_key and secret_key:
            erniebot.api_type = "qianfan"
            erniebot.ak = api_key
            erniebot.sk = secret_key
        else:
            logger.warning(
                "ErnieBot 未配置认证信息，请设置环境变量 ERNIE_API_KEY/ERNIE_SECRET_KEY "
                "或 ERNIE_ACCESS_TOKEN"
            )

        self._model = model
        # QPS限流控制
        self._last_call_time = 0.0
        self._qps_lock = threading.Lock()
        self._min_interval = 1.0  # 保守1秒1次，彻底防超限

    def chat(
            self,
            question: str,
            context: dict[str, Any] | None = None,
            temperature: float = 0.3,
        ) -> dict[str, Any]:
        # QPS限流等待
        with self._qps_lock:
            now = time.time()
            wait_sec = self._last_call_time + self._min_interval - now
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_call_time = time.time()
        """
        智能问答：结合诊断上下文生成自然语言回答。
        """
        import erniebot

        user_prompt = _build_context_prompt(question, context or {})

        try:
            response = erniebot.ChatCompletion.create(
                model=self._model,
                messages=[{"role": "user", "content": user_prompt}],
                system=_INDUSTRIAL_SYSTEM_PROMPT,
                temperature=temperature,
                top_p=0.7,
            )
            return {
                "success": True,
                "answer": response.get_result(),
                "model": self._model,
                "error": None,
            }
        except (RequestLimitError, RateLimitError):
            logger.warning("ErnieBot QPS超限/配额用尽")
            return {"success": False, "answer": "", "model": self._model,
                    "error": "接口调用频次已达上限，请稍后再试或更换密钥"}
        except Exception as exc:
            logger.exception("ErnieBot 调用失败")
            return {"success": False, "answer": "", "model": self._model, "error": str(exc)}

    def extract_industrial_info(self, ocr_text: str) -> dict[str, Any]:
        """
        从文档文本中抽取工业生产关键信息。
        """
        with self._qps_lock:
            now = time.time()
            wait_sec = self._last_call_time + self._min_interval - now
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_call_time = time.time()

        import erniebot

        user_prompt = f"以下是工业文档识别结果，请提取关键信息：\n\n{ocr_text}"

        try:
            response = erniebot.ChatCompletion.create(
                model=self._model,
                messages=[{"role": "user", "content": user_prompt}],
                system=_INFO_EXTRACTION_PROMPT,
                temperature=0.1,
                top_p=0.5,
            )
            raw = response.get_result()
            info = self._parse_json_response(raw)
            return {"success": True, "info": info, "raw": raw, "error": None}
        except (RequestLimitError, RateLimitError):
            logger.warning("信息抽取触发QPS限制")
            return {"success": False, "info": {}, "error": "接口调用频次已达上限，请稍后再试"}
        except Exception as exc:
            logger.exception("信息抽取失败")
            return {"success": False, "info": {}, "error": str(exc)}

    def cross_modal_analyze(
        self,
        ocr_text: str,
        ocr_info: dict[str, Any],
        diagnosis_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        跨模态关联分析：将质检文档与传感器异常数据关联。
        """
        with self._qps_lock:
            now = time.time()
            wait_sec = self._last_call_time + self._min_interval - now
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_call_time = time.time()

        import erniebot

        diagnosis_text = _build_context_prompt("跨模态关联分析", diagnosis_context)
        user_prompt = (
            f"【质检文档内容】\n{ocr_text}\n\n"
            f"【抽取的关键信息】\n{json.dumps(ocr_info, ensure_ascii=False, indent=2)}\n\n"
            f"【传感器异常检测数据】\n{diagnosis_text}"
        )

        try:
            response = erniebot.ChatCompletion.create(
                model=self._model,
                messages=[{"role": "user", "content": user_prompt}],
                system=_CROSS_MODAL_PROMPT,
                temperature=0.3,
                top_p=0.7,
            )
            return {
                "success": True,
                "analysis": response.get_result(),
                "model": self._model,
                "error": None,
            }
        except (RequestLimitError, RateLimitError):
            logger.warning("跨模态分析触发QPS限制")
            return {"success": False, "analysis": "", "error": "接口调用频次已达上限，请稍后再试"}
        except Exception as exc:
            logger.exception("跨模态分析失败")
            return {"success": False, "analysis": "", "error": str(exc)}

    def generate_report(self, dataset: str, context: dict[str, Any]) -> dict[str, Any]:
        """生成诊断报告。"""
        with self._qps_lock:
            now = time.time()
            wait_sec = self._last_call_time + self._min_interval - now
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_call_time = time.time()
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
    """懒加载全局 ErnieBot 实例。"""
    global _ERNIE_INSTANCE
    if _ERNIE_INSTANCE is None:
        logger.info("ErnieBot 初始化中...")
        _ERNIE_INSTANCE = ErnieService()
        logger.info("ErnieBot 初始化完成")
    return _ERNIE_INSTANCE