"""
批量补全 industrial_info：用 DashScope (qwen-plus) 替代 ErnieBot 抽取工业信息。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.model.factory import ModelFactory

INFO_EXTRACTION_PROMPT = """你是一个工业文档信息抽取助手。请从以下工业文档文本中，提取关键的工艺参数和缺陷描述信息。

请严格按照 JSON 格式返回，包含以下字段（如未提取到则填空字符串 ""）：
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

只返回 JSON，不要任何其他文字。"""

DOC_HISTORY = Path(__file__).resolve().parent.parent / "outputs" / "document_history.json"


def extract_info(llm: ModelFactory, doc_text: str) -> dict:
    # 文本太长则截断（qwen-plus 上下文足够，但控制开销）
    text = doc_text[:8000] if len(doc_text) > 8000 else doc_text
    try:
        resp = llm.chat([
            {"role": "system", "content": INFO_EXTRACTION_PROMPT},
            {"role": "user", "content": f"以下是工业文档文本，请提取关键信息：\n\n{text}"},
        ], temperature=0.1, timeout=60)
        resp = resp.strip()
        if resp.startswith("```"):
            resp = resp.split("\n", 1)[-1]
            if resp.endswith("```"):
                resp = resp[:-3]
        return json.loads(resp)
    except json.JSONDecodeError:
        return {"process_params": {}, "defect_type": "", "defect_location": "", "defect_severity": "",
                "material": "", "batch_number": "", "inspection_result": "", "summary": f"(解析失败){resp[:200]}"}
    except Exception as e:
        return {"process_params": {}, "defect_type": "", "defect_location": "", "defect_severity": "",
                "material": "", "batch_number": "", "inspection_result": "", "summary": f"(异常){e}"}


def main():
    print("初始化 ModelFactory (qwen-plus)...")
    llm = ModelFactory()
    status = llm.status()
    print(f"  状态: {status}")

    docs = json.loads(DOC_HISTORY.read_text(encoding="utf-8"))
    print(f"共 {len(docs)} 条文档")

    updated = 0
    for i, doc in enumerate(docs):
        info = doc.get("industrial_info", {})
        # 跳过已有有效信息的（原始3条有内容）
        if info and info.get("summary", "").strip():
            print(f"[{i+1}/{len(docs)}] {doc['filename'][:40]}... 已有信息，跳过")
            continue

        doc_text = doc.get("doc_text", "")
        if not doc_text.strip():
            print(f"[{i+1}/{len(docs)}] {doc['filename'][:40]}... 无文本，跳过")
            continue

        print(f"[{i+1}/{len(docs)}] {doc['filename'][:40]}... 抽取中...", end=" ", flush=True)
        try:
            new_info = extract_info(llm, doc_text)
            doc["industrial_info"] = new_info
            print(f"OK: {new_info.get('summary', '')[:60]}")
            updated += 1
        except Exception as e:
            print(f"失败: {e}")

    DOC_HISTORY.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！更新 {updated} 条，共 {len(docs)} 条")
    print(f"文件: {DOC_HISTORY}")


if __name__ == "__main__":
    main()
