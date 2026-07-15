"""Batch-extract local Word/PDF documents and add them to document history.

Industrial information extraction continues to use the project's Ernie path.
Successfully extracted text is also indexed through the unified knowledge store.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import data_service as svc
from backend.rag import get_knowledge_store


SUPPORTED_SUFFIXES = {".docx", ".pdf"}


def extract_text(path: Path) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if path.suffix.lower() == ".docx":
        return svc.extract_docx_text(encoded)
    if path.suffix.lower() == ".pdf":
        return svc.extract_pdf_text(encoded)
    return {"success": False, "text": "", "error": f"不支持的格式：{path.suffix}"}


def process_directory(data_dir: Path) -> dict:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"数据目录不存在：{data_dir}")
    files = sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    store = get_knowledge_store()
    summary = {"total": len(files), "success": 0, "failed": 0, "items": []}
    for path in files:
        result = extract_text(path)
        text = str(result.get("text") or "").strip()
        if not result.get("success") or not text:
            summary["failed"] += 1
            summary["items"].append({"file": str(path), "success": False, "error": result.get("error")})
            continue

        info_result = svc.extract_industrial_info(text)
        industrial_info = info_result.get("info", {}) if info_result.get("success") else {}
        doc_id = svc.save_document_to_history(path.name, text, industrial_info)
        knowledge_result = store.ingest_text(path.name, text)
        summary["success"] += 1
        summary["items"].append({
            "file": str(path),
            "success": True,
            "doc_id": doc_id,
            "knowledge_status": knowledge_result.get("status"),
            "extraction_error": info_result.get("error"),
        })
    summary["knowledge_store"] = store.status
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="批量导入 Word/PDF 工业文档")
    parser.add_argument("data_dir", type=Path, help="待导入文档目录")
    args = parser.parse_args()
    print(json.dumps(process_directory(args.data_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
