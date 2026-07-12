from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MOONSHOT_MODEL = "kimi-k2.6"


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def call_llm(prompt: str) -> str:
    """
    Unified LLM entry point for table SQL generation.

    Kimi / Moonshot is OpenAI-compatible, so we keep this provider-neutral enough
    for future replacements while using Moonshot by default.
    """
    _load_project_env()
    api_key = os.getenv("TABLE_AGENT_KIMI_API_KEY")
    if not api_key:
        raise RuntimeError("TABLE_AGENT_KIMI_API_KEY is not configured")

    base_url = os.getenv("TABLE_AGENT_KIMI_BASE_URL") or DEFAULT_MOONSHOT_BASE_URL
    model = os.getenv("TABLE_AGENT_KIMI_MODEL") or DEFAULT_MOONSHOT_MODEL

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你只输出一个合法 JSON 对象，不输出 Markdown，不输出解释文字。",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2400,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Kimi returned empty content")
    return content


def build_sql_prompt(question: str, schema: list[dict[str, Any]]) -> str:
    return f"""
你是一个工业表格数据分析 SQL 生成器。

你只能输出 JSON，不要输出 Markdown，不要解释。

当前数据库：DuckDB
当前表名：table_data

表结构：
{json.dumps(schema, ensure_ascii=False, indent=2)}

用户问题：
{question}

要求：
1. 只生成 SELECT 查询。
2. 禁止 INSERT、UPDATE、DELETE、DROP、ALTER、CREATE。
3. 查询必须基于已有业务字段，或使用下面列出的内部辅助字段。
4. 原始字段名必须用英文双引号包裹；SQL 别名可以使用中文。
5. 默认 LIMIT 100。
6. 主查询 sql 用来回答用户问题；visualizations 用来画图，两者可以是不同 SQL。
7. 具体展示什么由你根据表结构和用户问题决定，但 visualizations 必须至少包含 1 个 line 和 1 个 pie。
8. 每个 visualization 都必须提供自己的 SELECT SQL，后端会逐条执行；不要让前端二次计算。
9. 每个 visualization 的 SQL 结果必须直接包含 x_field 和 y_field 指向的两列，y_field 必须是数值列；禁止把 SQL 列别名直接写成 x_field 或 y_field 这种占位词。
10. line 图：优先选择日期、时间、序号、连续数值等有顺序的字段做 x_field；如果没有明显时间字段，可以使用 "_excel_row_number" 作为原始行顺序。
11. DuckDB 字符串时间字段必须先转换再格式化，例如 strftime('%Y-%m-%d', CAST("报警时间" AS TIMESTAMP))；不要写 strftime("报警时间", '%Y-%m-%d')。
12. pie 图：优先选择低基数类别字段做 x_field；如果没有明显类别字段，可以用 CASE WHEN 对一个数值字段分箱后统计。
13. 排行或对比可以额外返回 bar；明细可以额外返回 table。
14. 每条 SQL 都必须包含 LIMIT，最多 LIMIT 100。
15. title/chart_title 用中文短标题，描述图表展示什么。
16. 如果无法生成安全 SELECT，返回 error 字段。

可用内部辅助字段：
- "_excel_row_number"：原始 Excel/CSV 行号，适合没有时间字段时画曲线。
- "_source_row_id"：从 0 开始的内部行号。

输出 JSON：
{{
  "intent": "ranking/trend/filter/detail/compare",
  "sql": "SELECT ... FROM table_data ...",
  "chart_type": "bar/line/pie/table",
  "x_field": "...",
  "y_field": "...",
  "chart_title": "...",
  "reason": "...",
  "visualizations": [
    {{
      "chart_type": "line",
      "title": "中文短标题",
      "sql": "SELECT ... FROM table_data ... LIMIT 100",
      "x_field": "SQL 结果中的横轴字段名",
      "y_field": "SQL 结果中的数值字段名",
      "reason": "为什么这张曲线适合当前表"
    }},
    {{
      "chart_type": "pie",
      "title": "中文短标题",
      "sql": "SELECT ... FROM table_data ... LIMIT 100",
      "x_field": "SQL 结果中的分类字段名",
      "y_field": "SQL 结果中的数值字段名",
      "reason": "为什么这张饼图适合当前表"
    }}
  ]
}}
"""


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data
