from __future__ import annotations

import json
import math
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from llm_service import build_sql_prompt, call_llm, parse_llm_json
except ModuleNotFoundError:
    from .llm_service import build_sql_prompt, call_llm, parse_llm_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = PROJECT_ROOT / "storage" / "tables"
UPLOAD_ROOT = TABLE_ROOT / "uploads"
DB_ROOT = TABLE_ROOT / "duckdb"
META_ROOT = TABLE_ROOT / "meta"
QUERY_ROOT = TABLE_ROOT / "queries"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

for path in [UPLOAD_ROOT, DB_ROOT, META_ROOT, QUERY_ROOT]:
    path.mkdir(parents=True, exist_ok=True)


FORBIDDEN_SQL = [
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "truncate",
    "create",
    "attach",
    "detach",
    "copy",
    "install",
    "load",
    "pragma",
]

FORBIDDEN_SQL_PATTERNS = [
    r"--",
    r"/\*",
    r"\bread_csv\b",
    r"\bread_json\b",
    r"\bread_parquet\b",
    r"\bhttpfs\b",
]

ALLOWED_CHART_TYPES = {"bar", "line", "pie", "table"}
REQUIRED_VISUAL_CHART_TYPES = {"line", "pie"}


def _ensure_storage_dirs() -> None:
    for path in [UPLOAD_ROOT, DB_ROOT, META_ROOT, QUERY_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def _safe_error_message(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"sk-[a-zA-Z0-9]+", "sk-***", text)
    return text[:240]


def _safe_filename(filename: str) -> str:
    name = Path(filename).name or "uploaded_table"
    return re.sub(r"[^a-zA-Z0-9_\-.一-龥]", "_", name)


def _validate_file_id(file_id: str) -> str:
    if not re.fullmatch(r"tbl_[a-f0-9]{12}", file_id):
        raise FileNotFoundError(f"表格不存在：{file_id}")
    return file_id


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _records_from_df(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    data = df.head(limit) if limit else df
    data = data.astype(object).where(pd.notnull(data), None)
    return [{key: _json_safe(value) for key, value in row.items()} for row in data.to_dict(orient="records")]


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "string"


def _clean_columns(columns: list[Any]) -> list[str]:
    result: list[str] = []
    used: set[str] = set()

    for index, col in enumerate(columns):
        raw = str(col).strip()
        name = raw if raw else f"column_{index + 1}"
        name = name.replace("\n", "_").replace("\r", "_")
        base = name
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result.append(name)

    return result


def _user_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not col.startswith("_source_") and col != "_excel_row_number"]


def _schema_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []

    for col in _user_columns(df):
        series = df[col]
        samples = [_json_safe(value) for value in series.dropna().head(5).tolist()]
        schema.append(
            {
                "name": col,
                "type": _infer_kind(series),
                "missing_count": int(series.isna().sum()),
                "missing_ratio": round(float(series.isna().mean()), 4),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": samples,
            }
        )

    return schema


def _read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path)


def _read_table(file_path: Path, sheet_name: str | None = None) -> tuple[pd.DataFrame, str]:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return _read_csv_with_fallback(file_path), "csv"

    if suffix in [".xlsx", ".xls"]:
        excel = pd.ExcelFile(file_path)
        selected_sheet = sheet_name or excel.sheet_names[0]
        if selected_sheet not in excel.sheet_names:
            raise ValueError(f"Excel sheet 不存在：{selected_sheet}")
        return pd.read_excel(file_path, sheet_name=selected_sheet), selected_sheet

    raise ValueError("只支持 CSV、XLSX、XLS 文件")


def upload_table(filename: str, content: bytes, sheet_name: str | None = None) -> dict[str, Any]:
    _ensure_storage_dirs()

    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("上传文件不能超过 50MB")

    file_id = f"tbl_{uuid.uuid4().hex[:12]}"
    safe_name = _safe_filename(filename)
    file_path = UPLOAD_ROOT / f"{file_id}_{safe_name}"
    file_path.write_bytes(content)

    df, active_sheet = _read_table(file_path, sheet_name)
    if df.empty:
        raise ValueError("表格没有可分析的数据行")

    df.columns = _clean_columns(list(df.columns))
    df["_source_row_id"] = range(len(df))
    df["_excel_row_number"] = df.index + 2
    df["_source_sheet"] = active_sheet
    df["_source_file"] = safe_name

    db_path = DB_ROOT / f"{file_id}.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("uploaded_df", df)
        conn.execute("CREATE OR REPLACE TABLE table_data AS SELECT * FROM uploaded_df")
    finally:
        conn.close()

    meta = {
        "file_id": file_id,
        "filename": safe_name,
        "active_sheet": active_sheet,
        "db_path": str(db_path),
        "table_name": "table_data",
        "rows": int(len(df)),
        "columns": len(_user_columns(df)),
        "schema": _schema_from_df(df),
        "preview": _records_from_df(df, 20),
    }

    (META_ROOT / f"{file_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def get_table_meta(file_id: str) -> dict[str, Any]:
    _ensure_storage_dirs()

    checked_id = _validate_file_id(file_id)
    path = META_ROOT / f"{checked_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"表格不存在：{file_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sql(sql: str) -> str:
    checked_sql = sql.strip()
    if not checked_sql:
        raise ValueError("SQL 不能为空")

    lower = checked_sql.lower()

    if not lower.startswith("select"):
        raise ValueError("只允许执行 SELECT 查询")
    if ";" in lower[:-1]:
        raise ValueError("禁止多语句 SQL")
    if not re.search(r'\bfrom\s+"?table_data"?\b', lower):
        raise ValueError("SQL 必须查询 table_data 表")

    for word in FORBIDDEN_SQL:
        if re.search(rf"\b{word}\b", lower):
            raise ValueError(f"禁止使用危险 SQL 关键词：{word}")
    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, lower):
            raise ValueError("SQL 包含不允许的外部读取或注释语法")

    if not re.search(r"\blimit\b", lower):
        checked_sql = checked_sql.rstrip(";") + " LIMIT 100"

    return checked_sql.rstrip(";")


def run_sql(file_id: str, sql: str) -> pd.DataFrame:
    meta = get_table_meta(file_id)
    checked_sql = validate_sql(sql)
    db_path = Path(meta["db_path"])
    if not db_path.exists():
        raise FileNotFoundError("表格数据库不存在，请重新上传文件")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return conn.execute(checked_sql).fetchdf()
    finally:
        conn.close()


def fallback_generate_sql(question: str, schema: list[dict[str, Any]]) -> dict[str, Any]:
    columns = [item["name"] for item in schema]
    question_lower = question.lower()

    device_col = next((col for col in columns if "设备" in col or "device" in col.lower()), None)
    time_col = next((col for col in columns if "时间" in col or "日期" in col or "date" in col.lower() or "time" in col.lower()), None)
    alarm_col = next((col for col in columns if "报警" in col or "异常" in col or "alarm" in col.lower()), None)

    if ("最多" in question or "排行" in question or "top" in question_lower) and device_col:
        return {
            "intent": "ranking",
            "sql": f'SELECT "{device_col}" AS 设备, COUNT(*) AS 数量 FROM table_data GROUP BY "{device_col}" ORDER BY 数量 DESC LIMIT 20',
            "chart_type": "bar",
            "x_field": "设备",
            "y_field": "数量",
            "reason": "问题属于排行统计，因此按设备字段分组计数。",
        }

    if ("趋势" in question or "每天" in question or "日期" in question) and time_col:
        return {
            "intent": "trend",
            "sql": f'SELECT CAST("{time_col}" AS DATE) AS 日期, COUNT(*) AS 数量 FROM table_data GROUP BY 日期 ORDER BY 日期 LIMIT 100',
            "chart_type": "line",
            "x_field": "日期",
            "y_field": "数量",
            "reason": "问题属于时间趋势分析，因此按日期聚合计数。",
        }

    if ("报警" in question or "异常" in question) and alarm_col:
        return {
            "intent": "ranking",
            "sql": f'SELECT "{alarm_col}" AS 报警类型, COUNT(*) AS 数量 FROM table_data GROUP BY "{alarm_col}" ORDER BY 数量 DESC LIMIT 20',
            "chart_type": "bar",
            "x_field": "报警类型",
            "y_field": "数量",
            "reason": "问题关注报警或异常类型，因此按报警字段分组计数。",
        }

    selected = columns[: min(8, len(columns))]
    sql_cols = ", ".join([f'"{col}"' for col in selected])
    return {
        "intent": "detail",
        "sql": f"SELECT {sql_cols}, _excel_row_number FROM table_data LIMIT 50",
        "chart_type": "table",
        "x_field": "",
        "y_field": "",
        "reason": "无法稳定判断统计意图，因此返回明细数据。",
    }


def explain_result(question: str, sql: str, result_rows: list[dict[str, Any]], source: dict[str, Any]) -> str:
    if not result_rows:
        return "本次查询没有返回匹配记录，说明当前表格中没有满足条件的数据，或问题中的字段条件与表结构不完全匹配。"

    keys = list(result_rows[0].keys())
    first_row_summary = ""
    if len(keys) >= 2:
        first = result_rows[0]
        first_row_summary = f"首行结果显示：{keys[0]} 为 {first.get(keys[0])}，{keys[1]} 为 {first.get(keys[1])}。"

    return (
        f"本次问题是“{question}”。系统执行 SQL 后返回 {len(result_rows)} 条结果。"
        f"结果字段包括：{', '.join(keys)}。"
        f"{first_row_summary}"
        f"数据来源为文件《{source.get('filename')}》的 {source.get('active_sheet')}，"
        "可继续结合原始行号和业务背景判断异常原因。"
    )


def _build_plan(question: str, schema: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        prompt = build_sql_prompt(question, schema)
        llm_text = call_llm(prompt)
        plan = parse_llm_json(llm_text)
        if plan.get("error"):
            raise ValueError(str(plan["error"]))
        plan["generated_by"] = "kimi"
        plan["fallback_reason"] = ""
        return plan
    except Exception as exc:
        raise ValueError(f"Kimi 生成 SQL 失败：{_safe_error_message(exc)}") from exc


def _normalize_chart_fields(plan: dict[str, Any], result_rows: list[dict[str, Any]]) -> tuple[str, str]:
    x_field = str(plan.get("x_field") or "")
    y_field = str(plan.get("y_field") or "")
    if not result_rows:
        return x_field, y_field

    keys = list(result_rows[0].keys())
    if x_field not in keys:
        x_field = keys[0] if keys else ""
    if y_field not in keys:
        numeric_key = next(
            (
                key
                for key in keys
                if key != x_field and isinstance(result_rows[0].get(key), (int, float))
            ),
            "",
        )
        y_field = numeric_key or (keys[1] if len(keys) > 1 else "")
    return x_field, y_field


def _normalize_chart_type(plan: dict[str, Any]) -> str:
    chart_type = str(plan.get("chart_type") or "table").lower().strip()
    return chart_type if chart_type in ALLOWED_CHART_TYPES else "table"


def _has_numeric_values(rows: list[dict[str, Any]], field: str) -> bool:
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            continue
    return False


def _repair_duckdb_datetime_sql(sql: str) -> str:
    return re.sub(
        r"strftime\(\s*(\"[^\"]+\")\s*,\s*('[^']+')\s*\)",
        r"strftime(\2, CAST(\1 AS TIMESTAMP))",
        sql,
        flags=re.IGNORECASE,
    )


def _execute_visualization(file_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    chart_type = _normalize_chart_type(raw)
    sql_text = str(raw.get("sql") or "").strip()
    if chart_type == "table" or not sql_text:
        raise ValueError("Kimi 图表计划缺少可执行 SQL")

    checked_sql = validate_sql(sql_text)
    try:
        result_df = run_sql(file_id, checked_sql)
    except duckdb.BinderException:
        repaired_sql = validate_sql(_repair_duckdb_datetime_sql(checked_sql))
        if repaired_sql == checked_sql:
            raise
        checked_sql = repaired_sql
        result_df = run_sql(file_id, checked_sql)
    result_rows = _records_from_df(result_df)
    x_field, y_field = _normalize_chart_fields(raw, result_rows)

    if chart_type in REQUIRED_VISUAL_CHART_TYPES:
        if not result_rows:
            raise ValueError(f"Kimi 返回的 {chart_type} 图没有数据")
        if not x_field or not y_field:
            raise ValueError(f"Kimi 返回的 {chart_type} 图缺少 x_field 或 y_field")
        if not _has_numeric_values(result_rows, y_field):
            raise ValueError(f"Kimi 返回的 {chart_type} 图 y_field 不是数值列")

    return {
        "chart_type": chart_type,
        "title": raw.get("title") or raw.get("chart_title") or f"{chart_type} 图表",
        "sql": checked_sql,
        "x_field": x_field,
        "y_field": y_field,
        "reason": raw.get("reason", ""),
        "data": result_rows,
    }


def _execute_visualizations(file_id: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = plan.get("visualizations")
    if not isinstance(raw_items, list):
        raise ValueError("Kimi 未返回 visualizations 图表计划")

    visualizations: list[dict[str, Any]] = []
    required_errors: list[str] = []

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        chart_type = _normalize_chart_type(raw)
        try:
            spec = _execute_visualization(file_id, raw)
        except Exception as exc:
            if chart_type in REQUIRED_VISUAL_CHART_TYPES:
                required_errors.append(f"{chart_type}: {_safe_error_message(exc)}")
            continue

        visualizations.append(spec)

    available_required = {
        spec["chart_type"]
        for spec in visualizations
        if spec["chart_type"] in REQUIRED_VISUAL_CHART_TYPES
    }
    missing = sorted(REQUIRED_VISUAL_CHART_TYPES - available_required)
    if missing:
        detail = f"，失败原因：{'；'.join(required_errors)}" if required_errors else ""
        raise ValueError(f"Kimi 可视化计划缺少必需图表：{', '.join(missing)}{detail}")

    return visualizations


def query_table(file_id: str, question: str) -> dict[str, Any]:
    _ensure_storage_dirs()

    clean_question = question.strip()
    if not clean_question:
        raise ValueError("问题不能为空")

    meta = get_table_meta(file_id)
    plan = _build_plan(clean_question, meta["schema"])
    sql = validate_sql(str(plan["sql"]))
    result_df = run_sql(file_id, sql)
    result_rows = _records_from_df(result_df)
    x_field, y_field = _normalize_chart_fields(plan, result_rows)
    chart_type = _normalize_chart_type(plan)
    main_chart_spec = {
        "chart_type": chart_type,
        "x_field": x_field,
        "y_field": y_field,
        "title": plan.get("chart_title") or clean_question,
        "data": result_rows,
    }
    visualizations = _execute_visualizations(file_id, plan)

    query_id = f"q_{uuid.uuid4().hex[:12]}"
    source_location = {
        "file_id": file_id,
        "filename": meta["filename"],
        "sheet": meta["active_sheet"],
        "matched_row_count": len(result_rows),
    }

    payload = {
        "query_id": query_id,
        "file_id": file_id,
        "question": clean_question,
        "intent": plan.get("intent", "detail"),
        "generated_by": plan.get("generated_by", "kimi"),
        "fallback_reason": plan.get("fallback_reason", ""),
        "sql": sql,
        "result": result_rows,
        "source_location": source_location,
        "chart_spec": visualizations[0] if visualizations else main_chart_spec,
        "visualizations": visualizations,
        "explanation": explain_result(clean_question, sql, result_rows[:20], meta),
        "llm_reason": plan.get("reason", "Kimi 已生成 SQL 和多图表计划。"),
    }

    (QUERY_ROOT / f"{query_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
