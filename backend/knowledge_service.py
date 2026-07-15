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
    from backend.rag import get_knowledge_store

    store = get_knowledge_store()
    return {"status": store.status, "documents": store.list_documents()}

def upload_document(filename: str, content: str) -> dict[str, Any]:
    from backend.rag import get_knowledge_store

    return get_knowledge_store().ingest_text(filename, content)

def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    from backend.rag import get_knowledge_store

    store = get_knowledge_store()
    return {"hits": store.search(query, top_k), "backend": store.status["mode"], "status": store.status}
