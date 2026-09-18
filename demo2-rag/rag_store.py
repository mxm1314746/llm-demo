"""
向量存储层 — 双 collection 设计
  - knowledge: 用户上传的文档片段（原始 RAG 能力）
  - skills:    L3 记忆层提取出的 skill（审批通过后写入，供检索增强）

封装 Chroma，隔离底层细节，供 memory 管线与 agent 工具复用。
"""
import os
import shutil
from typing import Any

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

KNOWLEDGE_COLLECTION = "knowledge"
SKILLS_COLLECTION = "skills"

_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def get_embeddings():
    """嵌入模型（bge-small-zh，约 30MB）"""
    return FastEmbedEmbeddings(model_name=_EMBEDDING_MODEL)


def _collection(name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  knowledge collection（文档 RAG）                            ║
# ╚══════════════════════════════════════════════════════════════╝

def add_documents(chunks: list) -> int:
    """把分割好的 Document 写入 knowledge collection"""
    if not chunks:
        return 0
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=KNOWLEDGE_COLLECTION,
    )
    return len(chunks)


def search_knowledge(query: str, top_k: int = 4) -> list[dict]:
    """检索文档片段，返回 [{content, source, score}]"""
    vs = _collection(KNOWLEDGE_COLLECTION)
    if vs._collection.count() == 0:  # noqa: SLF001
        return []
    hits = vs.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in hits:
        out.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "未知来源"),
            "score": round(float(score), 4),
        })
    return out


def knowledge_count() -> int:
    try:
        return _collection(KNOWLEDGE_COLLECTION)._collection.count()  # noqa: SLF001
    except Exception:
        return 0


def clear_knowledge() -> None:
    """清空 knowledge collection（保留 skills）"""
    try:
        client = _collection(KNOWLEDGE_COLLECTION)
        client.delete_collection()
    except Exception:
        pass


# ╔══════════════════════════════════════════════════════════════╗
# ║  skills collection（L3 记忆检索增强）                        ║
# ╚══════════════════════════════════════════════════════════════╝

def add_skill(skill_id: str, content: str, metadata: dict) -> None:
    """把一条已审批的 skill 写入 skills collection（幂等：先删同 id）"""
    vs = _collection(SKILLS_COLLECTION)
    try:
        vs.delete(ids=[skill_id])
    except Exception:
        pass
    meta = {"skill_id": skill_id, **metadata}
    vs.add_texts(texts=[content], metadatas=[meta], ids=[skill_id])


def remove_skill(skill_id: str) -> None:
    try:
        _collection(SKILLS_COLLECTION).delete(ids=[skill_id])
    except Exception:
        pass


def search_skills(query: str, top_k: int = 3, scope: str = "private",
                  owner: str = "default") -> list[dict]:
    """按 scope/owner 过滤检索 skill，返回 [{content, tag, score, skill_id}]"""
    vs = _collection(SKILLS_COLLECTION)
    if vs._collection.count() == 0:  # noqa: SLF001
        return []
    where = {"$and": [{"scope": scope}, {"owner": owner}]}
    try:
        hits = vs.similarity_search_with_score(query, k=top_k, filter=where)
    except Exception:
        hits = vs.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in hits:
        out.append({
            "content": doc.page_content,
            "tag": doc.metadata.get("tag", ""),
            "skill_id": doc.metadata.get("skill_id", ""),
            "score": round(float(score), 4),
        })
    return out


def skills_count() -> int:
    try:
        return _collection(SKILLS_COLLECTION)._collection.count()  # noqa: SLF001
    except Exception:
        return 0


# ╔══════════════════════════════════════════════════════════════╗
# ║  文档文件管理（沿用原 demo2 行为）                           ║
# ╚══════════════════════════════════════════════════════════════╝

def list_document_files() -> list[str]:
    files = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in sorted(os.listdir(KNOWLEDGE_DIR)):
            fp = os.path.join(KNOWLEDGE_DIR, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                files.append(f"{f} ({size_str})")
    return files


def clear_all_files() -> None:
    for f in os.listdir(KNOWLEDGE_DIR):
        fp = os.path.join(KNOWLEDGE_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
