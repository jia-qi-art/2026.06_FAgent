"""Import saved document history into the project's configured knowledge store.

The knowledge store uses Chroma plus the configured embedding service when
available and records an explicit keyword-fallback reason otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from backend.config.settings import PROJECT_ROOT
from backend.rag import get_knowledge_store


DEFAULT_HISTORY = PROJECT_ROOT / "outputs" / "document_history.json"


def build_document_text(document: dict[str, Any]) -> str:
    parts = []
    if document.get("filename"):
        parts.append(f"文件名：{document['filename']}")
    if document.get("doc_text"):
        parts.append(str(document["doc_text"]))
    if isinstance(document.get("industrial_info"), dict):
        parts.append(json.dumps(document["industrial_info"], ensure_ascii=False))
    if document.get("cross_modal_analysis"):
        parts.append(str(document["cross_modal_analysis"]))
    return "\n\n".join(part for part in parts if part.strip())


def import_history(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        raise FileNotFoundError(f"文档历史文件不存在：{history_path}")
    documents = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError("document_history.json 顶层必须为数组")

    store = get_knowledge_store()
    indexed = duplicate = skipped = 0
    for document in documents:
        text = build_document_text(document)
        if not text.strip():
            skipped += 1
            continue
        filename = str(document.get("filename") or f"{document.get('doc_id', 'document')}.txt")
        result = store.ingest_text(filename, text)
        if result.get("status") == "indexed":
            indexed += 1
        else:
            duplicate += 1
    return {
        "total": len(documents),
        "indexed": indexed,
        "duplicates": duplicate,
        "skipped": skipped,
        "store": store.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="将文档历史导入统一知识库")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    print(json.dumps(import_history(args.history.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
