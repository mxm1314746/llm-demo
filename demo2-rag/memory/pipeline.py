"""
记忆分层管线 — LLM 驱动的提炼与合并
  L0→L1: tag_and_summarize  对一段对话打标签 + 摘要，生成 Episode
  L1→L2: consolidate        相似 tag 的 episode 合并进 Structure（每种 tag 一个），merge_count++
  L2→L3: extract_skill      Structure 合并次数达防抖阈值 → 提取 Skill（进入待审批队列）

所有 LLM 调用都有兜底：失败时退化为规则提取，保证管线不中断。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schema import Episode, Structure, Skill, SkillStatus


@dataclass
class MemoryConfig:
    episode_rounds: int = 3          # 每 N 轮对话聚合为一个 Episode
    session_timeout: int = 300       # 会话空闲 N 秒后，不足 N 轮也合并为 Episode
    debounce_threshold: int = 3      # 防抖：Structure 合并达 N 次才提取 Skill
    tag_similarity: float = 0.5      # tag 相似度阈值（Jaccard）
    require_approval: bool = True    # L3 生成 skill 前是否需人工审批


# ╔══════════════════════════════════════════════════════════════╗
# ║  LLM 调用（带兜底）                                          ║
# ╚══════════════════════════════════════════════════════════════╝

def _chat_json(client, model: str, system: str, user: str, default: dict) -> dict:
    """调用 LLM 并解析 JSON，失败返回 default"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return default


def _fallback_tags(messages: list[dict]) -> list[str]:
    """无 LLM 时的规则打标签：取用户消息里的高频中文词/关键词"""
    text = " ".join(m["content"] for m in messages if m.get("role") == "user")
    keywords = ["退票", "改签", "值机", "选座", "行李", "航班", "订票", "订单",
                "机场", "政策", "延误", "座位", "发票", "改期", "退款", "宠物",
                "儿童", "身份证", "里程", "会员", "中转", "签证", "天气"]
    hits = [k for k in keywords if k in text]
    return hits[:3] or ["通用咨询"]


def tag_and_summarize(client, model: str, messages: list[dict]) -> tuple[str, list[str]]:
    """L0→L1：对一段对话生成摘要 + 标签"""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    system = (
        "你是对话记忆提炼器。阅读一段人机对话，输出 JSON："
        '{"summary": "一句话中文摘要(≤50字)", "tags": ["2-4个中文主题标签"]}。'
        "标签要概括用户意图领域，如'退票规则'、'值机流程'、'行李政策'。只输出 JSON。"
    )
    data = _chat_json(client, model, system, convo, {})
    summary = (data.get("summary") or "").strip()
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(t).strip() for t in tags if str(t).strip()][:4]
    if not summary:
        first_user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        summary = first_user[:50] or "（空对话）"
    if not tags:
        tags = _fallback_tags(messages)
    return summary, tags


def merge_summary(client, model: str, existing: str, incoming: str, tag: str) -> str:
    """L2：把新 episode 摘要并入 structure 摘要"""
    if not existing:
        return incoming
    system = (
        f"你在维护一个关于「{tag}」的长期记忆结构。"
        "把已有摘要和新摘要融合成一句更完整的中文摘要(≤80字)，保留关键事实，不要重复。只输出摘要文本。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"已有摘要：{existing}\n新摘要：{incoming}"},
            ],
            temperature=0.2, max_tokens=200,
        )
        merged = resp.choices[0].message.content.strip()
        if merged:
            return merged
    except Exception:
        pass
    return f"{existing}；{incoming}"[:160]


def extract_skill_content(client, model: str, structure: Structure) -> str:
    """L2→L3：把 Structure 蒸馏成可复用的 skill 知识"""
    system = (
        f"你在把关于「{structure.tag}」的多段记忆提炼为一条可复用技能(skill)。"
        "输出中文，包含：适用场景、关键结论/操作步骤、注意事项。控制在 150 字内，条理清晰。只输出技能正文。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"标签：{structure.tag}\n累计摘要：{structure.summary}\n合并次数：{structure.merge_count}"},
            ],
            temperature=0.3, max_tokens=400,
        )
        content = resp.choices[0].message.content.strip()
        if content:
            return content
    except Exception:
        pass
    return f"【{structure.tag}】{structure.summary}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  标签相似度 + Structure 归并                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def tag_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard 相似度（含子串包含的宽松匹配）"""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    # 宽松：任一标签互为子串也算命中
    for x in sa:
        for y in sb:
            if x != y and (x in y or y in x):
                inter.add((x, y))
    union = sa | sb
    return len(inter) / max(len(union), 1)


def pick_primary_tag(tags: list[str]) -> str:
    return tags[0] if tags else "通用"


def find_similar_structure(structures: list[Structure], tags: list[str],
                           threshold: float) -> Structure | None:
    """在现有 structure 中找 tag 最相似且达阈值的一个"""
    best, best_score = None, 0.0
    for st in structures:
        score = tag_similarity(st.tags or [st.tag], tags)
        if score > best_score:
            best, best_score = st, score
    return best if best_score >= threshold else None


def make_skill(structure: Structure, content: str, require_approval: bool) -> Skill:
    """由 Structure 生成 Skill（默认进入待审批）"""
    return Skill(
        structure_id=structure.id,
        scope=structure.scope,
        owner=structure.owner,
        tag=structure.tag,
        content=content,
        status=SkillStatus.PENDING.value if require_approval else SkillStatus.APPROVED.value,
        source_episodes=list(structure.episode_ids),
    )
