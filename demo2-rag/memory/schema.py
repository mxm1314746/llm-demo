"""
分层记忆数据模型
  L0 Message  — 原始对话消息
  L1 Episode  — 每 N 轮（或超时）聚合的一段对话，带 summary + tags
  L2 Structure— 相似 tag 的 episode 合并体，每种 tag 保留一个，记 merge_count
  L3 Skill    — structure 合并达阈值(防抖)后提取的技能，审批通过写入向量库

scope: private（个人）/ team（团队共享）——核心闭环默认 private，team 留接口。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Scope(str, Enum):
    PRIVATE = "private"
    TEAM = "team"


class SkillStatus(str, Enum):
    PENDING = "pending"      # 达到防抖阈值，等待人工审批
    APPROVED = "approved"    # 已审批，已写入 skills 向量库
    REJECTED = "rejected"    # 已拒绝


def _now() -> float:
    return time.time()


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass
class Message:
    """L0：单条原始消息"""
    role: str                 # user / assistant
    content: str
    ts: float = field(default_factory=_now)


@dataclass
class Episode:
    """L1：一段聚合对话"""
    id: str = field(default_factory=lambda: _uid("ep"))
    session_id: str = ""
    scope: str = Scope.PRIVATE.value
    owner: str = "default"
    start_ts: float = field(default_factory=_now)
    end_ts: float = field(default_factory=_now)
    rounds: int = 0
    messages: list[dict] = field(default_factory=list)   # 原始 L0 快照
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    promoted_to: str = ""     # 合并进的 structure id


@dataclass
class Structure:
    """L2：相似 tag 的 episode 合并体，每种 tag 保留一个"""
    id: str = field(default_factory=lambda: _uid("st"))
    scope: str = Scope.PRIVATE.value
    owner: str = "default"
    tag: str = ""                                   # 主标签
    tags: list[str] = field(default_factory=list)   # 归并后的标签集合
    summary: str = ""                               # 滚动合并的摘要
    episode_ids: list[str] = field(default_factory=list)
    merge_count: int = 0                            # 合并次数（防抖计数）
    updated_ts: float = field(default_factory=_now)
    skill_id: str = ""                              # 已提取的 skill（若有）


@dataclass
class Skill:
    """L3：从 structure 提取的技能/长期知识"""
    id: str = field(default_factory=lambda: _uid("sk"))
    structure_id: str = ""
    scope: str = Scope.PRIVATE.value
    owner: str = "default"
    tag: str = ""
    content: str = ""                               # 蒸馏后的可复用知识
    status: str = SkillStatus.PENDING.value
    source_episodes: list[str] = field(default_factory=list)
    created_ts: float = field(default_factory=_now)
    reviewed_ts: float | None = None


# ── 序列化辅助 ──

def to_dict(obj: Any) -> dict:
    return asdict(obj)


def episode_from_dict(d: dict) -> Episode:
    return Episode(**{k: v for k, v in d.items() if k in Episode.__dataclass_fields__})


def structure_from_dict(d: dict) -> Structure:
    return Structure(**{k: v for k, v in d.items() if k in Structure.__dataclass_fields__})


def skill_from_dict(d: dict) -> Skill:
    return Skill(**{k: v for k, v in d.items() if k in Skill.__dataclass_fields__})
