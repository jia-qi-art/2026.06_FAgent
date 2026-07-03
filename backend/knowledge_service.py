"""
知识库管理模块 — 前端 /api/knowledge/* 的对应后端。
目前使用内存存储作为轻量实现，后续可切换 ChromaDB。
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDoc:
    doc_id: str
    filename: str
    content: str
    chunk_count: int = 0


_DOCS: dict[str, KnowledgeDoc] = {}


def list_documents() -> dict[str, Any]:
    docs = list(_DOCS.values())
    return {
        "status": {
            "backend": "in_memory",
            "total_docs": len(docs),
            "total_chunks": sum(d.chunk_count for d in docs),
            "vector_db": "placeholder",
        },
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "chunk_count": d.chunk_count,
            }
            for d in docs
        ],
    }


def upload_document(filename: str, content: str) -> dict[str, Any]:
    doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
    # 简单按段落切片
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    chunk_count = max(1, len(paragraphs))
    _DOCS[doc_id] = KnowledgeDoc(
        doc_id=doc_id,
        filename=filename,
        content=content,
        chunk_count=chunk_count,
    )
    logger.info("知识文档已索引: %s (%d 个切片)", filename, chunk_count)
    return {"status": "ok", "filename": filename, "doc_id": doc_id}


def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """基于关键词的简单检索（ChromaDB 占位实现）。"""
    hits: list[dict[str, Any]] = []
    query_lower = query.lower()
    for doc in _DOCS.values():
        # 简单关键词匹配
        paragraphs = [p.strip() for p in doc.content.split("\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            if any(word.lower() in para.lower() for word in query_lower.split()):
                hits.append(
                    {
                        "chunk_id": f"{doc.doc_id}_{i}",
                        "title": f"{doc.filename} 第{i+1}段",
                        "score": 0.85 - len(hits) * 0.05,
                        "text": para[:500],
                    }
                )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return {"hits": hits[:top_k], "backend": "keyword"}
