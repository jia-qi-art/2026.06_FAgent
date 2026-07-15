"""Fill missing industrial information in document history through Ernie."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import data_service as svc


DEFAULT_HISTORY = PROJECT_ROOT / "outputs" / "document_history.json"


def update_history(path: Path) -> dict:
    documents = json.loads(path.read_text(encoding="utf-8"))
    updated = skipped = failed = 0
    for document in documents:
        existing = document.get("industrial_info")
        if isinstance(existing, dict) and str(existing.get("summary") or "").strip():
            skipped += 1
            continue
        text = str(document.get("doc_text") or "").strip()
        if not text:
            skipped += 1
            continue
        result = svc.extract_industrial_info(text)
        if result.get("success"):
            document["industrial_info"] = result.get("info", {})
            updated += 1
        else:
            failed += 1
    path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"total": len(documents), "updated": updated, "skipped": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Ernie 补全文档工业信息")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    print(json.dumps(update_history(args.history.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
