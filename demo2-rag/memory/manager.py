"""
MemoryManager — 分层记忆门面
统一封装 L0 记录、L1 聚合、L2 归并、L3 提取与审批、recall 检索增强。
状态以 JSON 文件为准，每次变更「读→改→写」，避免多副本不一致。
"""
from __future__ import annotations

import time
from typing import Any

from . import store
from .schema import (
    Episode, Structure, Skill, SkillStatus, Scope,
    to_dict, episode_from_dict, structure_from_dict, skill_from_dict,
)
from .pipeline import (
    MemoryConfig, tag_and_summarize, merge_summary, extract_skill_content,
    find_similar_structure, pick_primary_tag, make_skill, tag_similarity,
)
import rag_store


class MemoryManager:
    def __init__(self, client, model: str, config: MemoryConfig | None = None):
        self.client = client
        self.model = model
        self.cfg = config or MemoryConfig()

    # ══════════════════════════════════════════════════════════
    #  L0：记录原始消息
    # ══════════════════════════════════════════════════════════

    def record_message(self, session_id: str, role: str, content: str,
                       scope: str = Scope.PRIVATE.value, owner: str = "default") -> dict:
        """写入一条 L0 消息；返回当前会话状态（含未聚合轮数）"""
        sess = store.load_session(session_id)
        sess.setdefault("processed", 0)
        sess["scope"] = scope
        sess["owner"] = owner
        sess["messages"].append({"role": role, "content": content, "ts": time.time()})
        sess["last_ts"] = time.time()
        store.save_session_state(session_id, sess)

        unprocessed = sess["messages"][sess["processed"]:]
        rounds = sum(1 for m in unprocessed if m["role"] == "user")
        return {"session_id": session_id, "pending_rounds": rounds,
                "total_messages": len(sess["messages"])}

    def pending_rounds(self, session_id: str) -> int:
        sess = store.load_session(session_id)
        unprocessed = sess["messages"][sess.get("processed", 0):]
        return sum(1 for m in unprocessed if m["role"] == "user")

    # ══════════════════════════════════════════════════════════
    #  L0→L1：关闭并生成 Episode
    # ══════════════════════════════════════════════════════════

    def close_episode(self, session_id: str, force: bool = False) -> Episode | None:
        """把当前会话未聚合的消息生成一个 Episode（不足 N 轮时 force/超时才聚合）"""
        sess = store.load_session(session_id)
        processed = sess.get("processed", 0)
        unprocessed = sess["messages"][processed:]
        rounds = sum(1 for m in unprocessed if m["role"] == "user")
        if rounds == 0:
            return None
        if not force and rounds < self.cfg.episode_rounds:
            return None

        summary, tags = tag_and_summarize(self.client, self.model, unprocessed)
        ep = Episode(
            session_id=session_id,
            scope=sess.get("scope", Scope.PRIVATE.value),
            owner=sess.get("owner", "default"),
            start_ts=unprocessed[0].get("ts", time.time()),
            end_ts=unprocessed[-1].get("ts", time.time()),
            rounds=rounds,
            messages=unprocessed,
            summary=summary,
            tags=tags,
        )
        # 持久化 episode
        episodes = [episode_from_dict(d) for d in store.load_episodes()]
        episodes.append(ep)
        store.save_episodes([to_dict(e) for e in episodes])
        # 标记已处理
        sess["processed"] = len(sess["messages"])
        store.save_session_state(session_id, sess)

        # L1→L2→L3
        self._consolidate(ep)
        return ep

    def store_note(self, content: str, tags: list[str],
                   scope: str = Scope.PRIVATE.value, owner: str = "default") -> Episode:
        """主动记忆：把一条事实/偏好立即落为 L1 Episode 并参与归并（不等 N 轮）"""
        tags = [str(t).strip() for t in (tags or []) if str(t).strip()][:4] or ["用户偏好"]
        ep = Episode(
            session_id="note", scope=scope, owner=owner,
            rounds=1,
            messages=[{"role": "user", "content": content, "ts": time.time()}],
            summary=content[:80], tags=tags,
        )
        episodes = [episode_from_dict(d) for d in store.load_episodes()]
        episodes.append(ep)
        store.save_episodes([to_dict(e) for e in episodes])
        self._consolidate(ep)
        return ep

    def check_timeout(self) -> list[str]:
        """扫描所有会话，空闲超时的未聚合片段强制成 Episode。返回被聚合的 session_id"""
        flushed = []
        now = time.time()
        for sid in store.list_sessions():
            sess = store.load_session(sid)
            processed = sess.get("processed", 0)
            if processed >= len(sess["messages"]):
                continue
            if now - sess.get("last_ts", now) >= self.cfg.session_timeout:
                ep = self.close_episode(sid, force=True)
                if ep:
                    flushed.append(sid)
        return flushed

    # ══════════════════════════════════════════════════════════
    #  L1→L2→L3：归并 + 防抖 + 提取
    # ══════════════════════════════════════════════════════════

    def _consolidate(self, ep: Episode) -> None:
        structures = [structure_from_dict(d) for d in store.load_structures()]

        target = find_similar_structure(structures, ep.tags, self.cfg.tag_similarity)
        if target is None:
            # 新建 structure（每种 tag 保留一个）
            target = Structure(
                scope=ep.scope, owner=ep.owner,
                tag=pick_primary_tag(ep.tags), tags=list(ep.tags),
                summary=ep.summary, episode_ids=[ep.id], merge_count=1,
            )
            structures.append(target)
        else:
            target.summary = merge_summary(self.client, self.model, target.summary, ep.summary, target.tag)
            target.episode_ids.append(ep.id)
            for t in ep.tags:
                if t not in target.tags:
                    target.tags.append(t)
            target.merge_count += 1
            target.updated_ts = time.time()

        ep.promoted_to = target.id

        # 防抖：合并次数达阈值且尚未提取 skill → 提取
        if target.merge_count >= self.cfg.debounce_threshold and not target.skill_id:
            content = extract_skill_content(self.client, self.model, target)
            skill = make_skill(target, content, self.cfg.require_approval)
            skills = [skill_from_dict(d) for d in store.load_skills()]
            skills.append(skill)
            store.save_skills([to_dict(s) for s in skills])
            target.skill_id = skill.id
            # 无需审批时直接写入向量库
            if skill.status == SkillStatus.APPROVED.value:
                self._index_skill(skill)

        # 回写
        store.save_structures([to_dict(s) for s in structures])
        episodes = [episode_from_dict(d) for d in store.load_episodes()]
        store.save_episodes([to_dict(e) for e in episodes])

    # ══════════════════════════════════════════════════════════
    #  L3：审批 + 向量库索引
    # ══════════════════════════════════════════════════════════

    def _index_skill(self, skill: Skill) -> None:
        rag_store.add_skill(
            skill_id=skill.id,
            content=skill.content,
            metadata={"tag": skill.tag, "scope": skill.scope,
                      "owner": skill.owner, "level": 3},
        )

    def pending_skills(self) -> list[Skill]:
        return [skill_from_dict(d) for d in store.load_skills()
                if d.get("status") == SkillStatus.PENDING.value]

    def approve_skill(self, skill_id: str) -> Skill | None:
        skills = [skill_from_dict(d) for d in store.load_skills()]
        for s in skills:
            if s.id == skill_id:
                s.status = SkillStatus.APPROVED.value
                s.reviewed_ts = time.time()
                self._index_skill(s)
                store.save_skills([to_dict(x) for x in skills])
                return s
        return None

    def reject_skill(self, skill_id: str) -> Skill | None:
        skills = [skill_from_dict(d) for d in store.load_skills()]
        for s in skills:
            if s.id == skill_id:
                s.status = SkillStatus.REJECTED.value
                s.reviewed_ts = time.time()
                store.save_skills([to_dict(x) for x in skills])
                return s
        return None

    # ══════════════════════════════════════════════════════════
    #  recall：检索增强（L3 向量 + L2 标签）
    # ══════════════════════════════════════════════════════════

    def recall(self, query: str, scope: str = Scope.PRIVATE.value,
               owner: str = "default", top_k: int = 3) -> dict:
        """返回长期记忆：已审批 skill（向量检索）+ 相关 structure（标签匹配）"""
        skills = rag_store.search_skills(query, top_k=top_k, scope=scope, owner=owner)

        # L2：用查询词与 structure tag/summary 做轻量匹配
        structures = [structure_from_dict(d) for d in store.load_structures()
                      if d.get("scope") == scope and d.get("owner") == owner]
        q_tokens = set(query)
        scored = []
        for st in structures:
            hay = st.tag + " " + " ".join(st.tags) + " " + st.summary
            overlap = len(q_tokens & set(hay)) / max(len(q_tokens), 1)
            if overlap > 0.15:
                scored.append((overlap, st))
        scored.sort(key=lambda x: x[0], reverse=True)
        rel_structures = [st for _, st in scored[:top_k]]

        return {"skills": skills, "structures": [to_dict(s) for s in rel_structures]}

    def format_recall(self, recalled: dict) -> str:
        """把 recall 结果拼成给 LLM 的上下文文本"""
        parts = []
        skills = recalled.get("skills", [])
        if skills:
            parts.append("【长期技能记忆(L3)】")
            for s in skills:
                parts.append(f"- [{s.get('tag','')}] {s.get('content','')}")
        structures = recalled.get("structures", [])
        if structures:
            parts.append("【相关结构化记忆(L2)】")
            for st in structures:
                parts.append(f"- [{st.get('tag','')}] {st.get('summary','')}")
        return "\n".join(parts) if parts else "（暂无相关长期记忆）"

    # ══════════════════════════════════════════════════════════
    #  统计 / 列表（供 UI）
    # ══════════════════════════════════════════════════════════

    def stats(self) -> dict:
        episodes = store.load_episodes()
        structures = store.load_structures()
        skills = store.load_skills()
        return {
            "sessions": len(store.list_sessions()),
            "L0_messages": sum(len(store.load_session(s)["messages"]) for s in store.list_sessions()),
            "L1_episodes": len(episodes),
            "L2_structures": len(structures),
            "L3_skills": len(skills),
            "L3_approved": sum(1 for s in skills if s.get("status") == SkillStatus.APPROVED.value),
            "L3_pending": sum(1 for s in skills if s.get("status") == SkillStatus.PENDING.value),
            "skills_indexed": rag_store.skills_count(),
        }

    def list_episodes(self) -> list[Episode]:
        return [episode_from_dict(d) for d in store.load_episodes()]

    def list_structures(self) -> list[Structure]:
        return [structure_from_dict(d) for d in store.load_structures()]

    def list_skills(self) -> list[Skill]:
        return [skill_from_dict(d) for d in store.load_skills()]


# ── 单例 ──
_MANAGER: MemoryManager | None = None


def get_memory_manager(client=None, model: str = "", config: MemoryConfig | None = None) -> MemoryManager:
    global _MANAGER
    if _MANAGER is None:
        if client is None:
            raise RuntimeError("MemoryManager 尚未初始化")
        _MANAGER = MemoryManager(client, model, config)
    return _MANAGER


def init_memory_manager(client, model: str, config: MemoryConfig | None = None) -> MemoryManager:
    global _MANAGER
    _MANAGER = MemoryManager(client, model, config)
    return _MANAGER
