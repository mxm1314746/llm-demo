"""
记忆持久化 — JSON 文件存储（原子写）
  memory_data/
    l0/{session_id}.json   原始会话消息
    l1_episodes.json       Episode 列表
    l2_structures.json     Structure 列表
    l3_skills.json         Skill 列表
"""
from __future__ import annotations

import json
import os
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "memory_data")
L0_DIR = os.path.join(MEMORY_DIR, "l0")
L1_FILE = os.path.join(MEMORY_DIR, "l1_episodes.json")
L2_FILE = os.path.join(MEMORY_DIR, "l2_structures.json")
L3_FILE = os.path.join(MEMORY_DIR, "l3_skills.json")

os.makedirs(L0_DIR, exist_ok=True)


def _atomic_write(path: str, data) -> None:
    """先写临时文件再替换，避免并发/崩溃损坏"""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


# ╔══════════════════════════════════════════════════════════════╗
# ║  L0 原始会话                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def _l0_path(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return os.path.join(L0_DIR, f"{safe}.json")


def load_session(session_id: str) -> dict:
    return _read_json(_l0_path(session_id), {"session_id": session_id, "messages": [], "last_ts": 0})


def save_session(session_id: str, messages: list[dict], last_ts: float) -> None:
    _atomic_write(_l0_path(session_id), {
        "session_id": session_id, "messages": messages, "last_ts": last_ts,
    })


def save_session_state(session_id: str, sess: dict) -> None:
    """保存完整会话状态（含 processed / scope / owner）"""
    sess = dict(sess)
    sess.setdefault("session_id", session_id)
    _atomic_write(_l0_path(session_id), sess)


def list_sessions() -> list[str]:
    if not os.path.exists(L0_DIR):
        return []
    return [f[:-5] for f in os.listdir(L0_DIR) if f.endswith(".json")]


# ╔══════════════════════════════════════════════════════════════╗
# ║  L1 / L2 / L3                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def load_episodes() -> list[dict]:
    return _read_json(L1_FILE, [])


def save_episodes(rows: list[dict]) -> None:
    _atomic_write(L1_FILE, rows)


def load_structures() -> list[dict]:
    return _read_json(L2_FILE, [])


def save_structures(rows: list[dict]) -> None:
    _atomic_write(L2_FILE, rows)


def load_skills() -> list[dict]:
    return _read_json(L3_FILE, [])


def save_skills(rows: list[dict]) -> None:
    _atomic_write(L3_FILE, rows)


def wipe_all() -> None:
    """清空所有记忆层（调试用）"""
    for f in (L1_FILE, L2_FILE, L3_FILE):
        if os.path.exists(f):
            os.remove(f)
    for s in list_sessions():
        p = _l0_path(s)
        if os.path.exists(p):
            os.remove(p)
