"""
批量导入脚本：读取 Data 目录下的 docx/pdf → 提取文字 → ErnieBot 抽取工业信息
→ 存入 document_history.json → 向量化导入 ChromaDB
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import data_service as svc

DATA_DIR = Path("D:/yjq/Python/Data")
SKIP_FILES = {
    # 扫描版 PDF，需 OCR 处理
    "GBT+3098.2-2025+紧固件机械性能_第2部分：螺母.pdf",
    "GB∕T-3098.1-2010-紧固件机械性能-螺栓、螺钉和螺柱.pdf",
    "传感器故障的检测与诊断.pdf",
    # .doc 旧格式，需手动转换
    "工业领域工程建设行业标准制定实施细则暂行.doc",
    "水处理标准规范一览表范文.doc",
    "水处理设备常见故障及处理.doc",
    "水处理设计规范.doc",
}


def extract_text(filepath: Path) -> dict:
    b64 = base64.b64encode(filepath.read_bytes()).decode()
    ext = filepath.suffix.lower()

    if ext in (".docx", ".doc"):
        result = svc.extract_docx_text(b64)
    elif ext == ".pdf":
        result = svc.extract_pdf_text(b64)
    else:
        return {"success": False, "text": "", "error": f"不支持格式: {ext}"}

    return result


def process_all():
    files = sorted(
        [f for f in DATA_DIR.iterdir() if f.suffix.lower() in (".docx", ".doc", ".pdf") and f.name not in SKIP_FILES]
    )

    print(f"共 {len(files)} 个文件待处理\n")

    success = 0
    skip_empty = 0
    fail = 0

    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fpath.name}")
        print(f"  大小: {fpath.stat().st_size / 1024:.0f} KB")

        # 1. 提取文字
        result = extract_text(fpath)
        if not result.get("success") or not result.get("text", "").strip():
            print(f"  ??? 提取失败或内容为空: {result.get('error', 'unknown')}")
            skip_empty += 1
            continue

        doc_text = result["text"]
        print(f"  提取文字: {len(doc_text)} 字符")

        # 2. ErnieBot 抽取工业信息
        print(f"  ErnieBot 抽取中...")
        info_result = svc.extract_industrial_info(doc_text)
        industrial_info = info_result.get("info", {}) if info_result.get("success") else {}

        # 3. 存入 document_history
        doc_id = svc.save_document_to_history(fpath.name, doc_text, industrial_info)
        print(f"  已保存: doc_id={doc_id}")
        print(f"  抽取信息: {json.dumps(industrial_info, ensure_ascii=False)[:200]}")
        print()
        success += 1

    print(f"===== 完成 =====")
    print(f"成功导入: {success}")
    print(f"内容为空/失败: {skip_empty}")
    print(f"总计: {len(files)}")

    # 4. 统计当前 document_history
    from data_service import document_history
    docs = document_history()
    print(f"\ndocument_history.json 现有 {len(docs)} 条记录")


if __name__ == "__main__":
    process_all()
