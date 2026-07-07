"""
将 document_history.json 中的数据导入向量数据库。

由于 Windows 上 chromadb 的 hnswlib native 扩展存在兼容性问题（segfault），
这里使用 numpy + 文件持久化实现轻量向量存储，存于 outputs/chroma_db/ 目录。
接口与 ChromaDB 保持一致，后续环境就绪后替换即可。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_HISTORY_PATH = PROJECT_ROOT / "outputs" / "document_history.json"
PERSIST_DIR = PROJECT_ROOT / "outputs" / "chroma_db"
COLLECTION_NAME = "document_history"
EMBEDDING_DIM = 1024  # dashscope text-embedding-v4 输出维度

# 占位嵌入开关：后续替换为 dashscope API 时关掉即可
USE_PLACEHOLDER_EMBEDDING = True


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """占位嵌入：SHA256 hash → 确定性向量 → L2归一化。"""
    arr = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        h = hashlib.sha256(f"{text}|{i}".encode("utf-8")).digest()
        # 用前4字节做 float，范围控制在 [-1, 1]
        val = int.from_bytes(h[:4], "big", signed=True) / (2**31 - 1)
        arr[i] = max(-1.0, min(1.0, val))
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def _real_embedding(text: str) -> np.ndarray:
    """TODO: 对接 dashscope text-embedding-v4"""
    # import dashscope
    # resp = dashscope.TextEmbedding.call(
    #     model="text-embedding-v4", input=text,
    #     api_key=os.getenv("LLM_API_KEY")
    # )
    # return np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
    raise NotImplementedError("替换为 dashscope API 调用")


def embed_text(text: str) -> np.ndarray:
    if USE_PLACEHOLDER_EMBEDDING:
        return _hash_embedding(text)
    return _real_embedding(text)


class LightweightVectorStore:
    """轻量向量存储（numpy + JSON 持久化），接口兼容 ChromaDB collection。"""

    def __init__(self, persist_dir: Path, collection_name: str):
        self._dir = persist_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._name = collection_name
        self._data_path = self._dir / f"{collection_name}.npz"
        self._meta_path = self._dir / f"{collection_name}_meta.json"
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._load()

    def _load(self):
        if self._data_path.exists() and self._meta_path.exists():
            data = np.load(self._data_path)
            self._embeddings = data["embeddings"]
            with open(self._meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._ids = meta["ids"]
            self._documents = meta["documents"]
            self._metadatas = meta["metadatas"]

    def _save(self):
        np.savez_compressed(self._data_path, embeddings=self._embeddings)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self._ids, "documents": self._documents, "metadatas": self._metadatas}, f, ensure_ascii=False, indent=2)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str] | None = None, metadatas: list[dict] | None = None):
        new_emb = np.array(embeddings, dtype=np.float32)
        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = new_emb
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        self._ids.extend(ids)
        self._documents.extend(documents or [""] * len(ids))
        self._metadatas.extend(metadatas or [{}] * len(ids))
        self._save()

    def count(self) -> int:
        return len(self._ids)

    def peek(self, limit: int = 1) -> dict[str, Any]:
        n = min(limit, self.count())
        if n == 0:
            return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        return {
            "ids": self._ids[:n],
            "embeddings": self._embeddings[:n].tolist() if self._embeddings is not None else [],
            "documents": self._documents[:n],
            "metadatas": self._metadatas[:n],
        }

    def query(self, query_embeddings: list[list[float]], n_results: int = 4) -> dict[str, Any]:
        if self._embeddings is None or len(self._embeddings) == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        q = np.array(query_embeddings, dtype=np.float32)
        # cosine similarity
        q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
        db_norm = self._embeddings / np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        sim = q_norm @ db_norm.T  # (n_queries, n_docs)
        top_k = min(n_results, self.count())
        indices = np.argsort(-sim, axis=1)[:, :top_k]
        distances = np.take_along_axis(sim, indices, axis=1)
        all_ids = []
        all_dists = []
        all_docs = []
        all_metas = []
        for i in range(len(q)):
            all_ids.append([self._ids[j] for j in indices[i]])
            all_dists.append(distances[i].tolist())
            all_docs.append([self._documents[j] for j in indices[i]])
            all_metas.append([self._metadatas[j] for j in indices[i]])
        return {"ids": all_ids, "distances": all_dists, "documents": all_docs, "metadatas": all_metas}


def load_documents(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"文档历史文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_document_text(doc: dict) -> str:
    parts = []
    if doc.get("filename"):
        parts.append(f"文件名: {doc['filename']}")
    if doc.get("doc_text"):
        parts.append(doc["doc_text"])
    if doc.get("industrial_info"):
        info = doc["industrial_info"]
        if isinstance(info, dict):
            parts.append(json.dumps(info, ensure_ascii=False))
    if doc.get("cross_modal_analysis"):
        parts.append(doc["cross_modal_analysis"])
    return "\n\n".join(parts)


def main():
    print(f"读取文档历史: {DOC_HISTORY_PATH}")
    docs = load_documents(DOC_HISTORY_PATH)
    print(f"共 {len(docs)} 条文档")

    print(f"初始化向量存储 (持久化目录: {PERSIST_DIR})")
    store = LightweightVectorStore(PERSIST_DIR, COLLECTION_NAME)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for doc in docs:
        doc_id = doc.get("doc_id", hashlib.md5(doc.get("doc_text", "").encode()).hexdigest()[:12])
        text = build_document_text(doc)
        embedding = embed_text(text).tolist()

        ids.append(doc_id)
        embeddings.append(embedding)
        documents.append(text)
        metadatas.append({
            "filename": doc.get("filename", ""),
            "created_at": str(doc.get("created_at", "")),
            "material": doc.get("industrial_info", {}).get("material", ""),
            "batch_number": doc.get("industrial_info", {}).get("batch_number", ""),
            "inspection_result": doc.get("industrial_info", {}).get("inspection_result", ""),
            "defect_type": doc.get("industrial_info", {}).get("defect_type", ""),
        })

    print(f"正在导入 {len(ids)} 条文档...")
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"导入完成! 集合 '{COLLECTION_NAME}' 共 {store.count()} 条记录")
    print(f"持久化目录: {PERSIST_DIR}")

    # 验证：peek + query
    result = store.peek(limit=1)
    if result["ids"]:
        print(f"\nPeek 验证成功，样例 id: {result['ids'][0]}")
        print(f"元数据: {json.dumps(result['metadatas'][0], ensure_ascii=False)}")

    # 测试查询
    test_q = embed_text("热处理 工艺参数 冷却速率")
    q_result = store.query(query_embeddings=[test_q.tolist()], n_results=3)
    print(f"\n查询 '热处理 工艺参数 冷却速率' Top-3:")
    for i, (doc_id, dist, meta) in enumerate(zip(q_result["ids"][0], q_result["distances"][0], q_result["metadatas"][0])):
        print(f"  {i+1}. id={doc_id} file={meta['filename']} score={dist:.4f}")


if __name__ == "__main__":
    main()
