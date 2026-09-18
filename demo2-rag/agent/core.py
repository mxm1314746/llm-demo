"""
Agent 编排 — 进程内工具调用（Function Calling）
把「文档 RAG」与「分层记忆」封装成工具，System Prompt 约束每轮主动调用：
  - recall_memory   每轮必先调用，取长期记忆(L3 skill + L2 structure)
  - search_knowledge 涉及文档时调用，取知识库片段
  - store_memory    用户明确要求记住 / 出现高价值事实时，直接落一条 L1 Episode

每轮用户/助手消息自动写入 MemoryManager 的 L0；满 N 轮或超时聚合为 Episode。
"""
from __future__ import annotations

import json
import concurrent.futures

from openai import OpenAI

from memory.manager import get_memory_manager
import rag_store


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "检索长期记忆（已审批的技能 L3 + 相关结构化记忆 L2）。每轮对话开始都应先调用，用于个性化/延续上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，通常是用户当前问题或其中的主题"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在用户上传的知识库文档中检索相关片段。当问题需要依据文档事实回答时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要在知识库中检索的问题"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认4"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "把一条重要事实/用户偏好主动写入长期记忆（立即生成一个 Episode 并参与后续归并）。用户说'记住…'或出现明确偏好时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要记住的内容（一句话事实/偏好）"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "2-4个主题标签"},
                },
                "required": ["content"],
            },
        },
    },
]


TOOL_ICONS = {"recall_memory": "🧠", "search_knowledge": "📚", "store_memory": "📝"}


def build_system_prompt(use_memory: bool = True) -> str:
    base = (
        "你是一个具备长期记忆的知识助手。回答要简洁、准确、有条理，使用中文。\n"
        "若依据了知识库文档，请在末尾用一行标注来源文件名。"
    )
    if not use_memory:
        return base + "\n\n（当前未启用记忆增强，可仅按需调用 search_knowledge。）"
    return base + (
        "\n\n# 记忆使用规则（重要）\n"
        "1. 每轮开始**必须先调用 recall_memory**，query 用用户问题的主题；把返回的长期记忆作为背景。\n"
        "2. 当问题需要文档事实支撑时，再调用 search_knowledge。\n"
        "3. 当用户明确要求'记住…'，或表达了稳定的个人偏好/事实时，调用 store_memory 落库。\n"
        "4. 综合「长期记忆 + 知识库 + 对话上下文」作答；若记忆与文档冲突，以文档为准并说明。\n"
        "5. 不要编造记忆里不存在的内容。"
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  工具执行                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def _exec_tool(name: str, args: dict, scope: str, owner: str) -> str:
    mgr = get_memory_manager()
    try:
        if name == "recall_memory":
            recalled = mgr.recall(args.get("query", ""), scope=scope, owner=owner)
            return mgr.format_recall(recalled)
        if name == "search_knowledge":
            hits = rag_store.search_knowledge(args.get("question", ""), int(args.get("top_k", 4) or 4))
            if not hits:
                return "（知识库为空或无相关片段）"
            lines = []
            for i, h in enumerate(hits):
                lines.append(f"[{i+1}] (来源:{h['source']}) {h['content']}")
            return "\n\n".join(lines)
        if name == "store_memory":
            ep = mgr.store_note(args.get("content", ""), args.get("tags") or [],
                                scope=scope, owner=owner)
            return f"已记住：{ep.summary}（标签：{'、'.join(ep.tags)}）"
        return f"未知工具: {name}"
    except Exception as e:
        return f"工具执行异常[{name}]: {e}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  ReAct 循环（Gradio generator）                              ║
# ╚══════════════════════════════════════════════════════════════╝

def chat(client: OpenAI, model: str, message: str, history: list,
         messages_state: list | None, session_id: str,
         scope: str = "private", owner: str = "default",
         use_memory: bool = True, temperature: float = 0.5,
         top_p: float = 0.8, max_iter: int = 5, top_k: int = 4):
    """
    yield (history, messages_state, mem_status_md)
    """
    mgr = get_memory_manager()

    # 1) 记录用户消息到 L0
    mgr.record_message(session_id, "user", message, scope, owner)

    history = history or []
    history.append({"role": "user", "content": message})
    assistant_msg = {"role": "assistant", "content": ""}
    history.append(assistant_msg)
    yield history, messages_state, ""

    # 2) 组装干净的 API 上下文
    messages = [{"role": "system", "content": build_system_prompt(use_memory)}]
    if messages_state:
        messages.extend(messages_state)
    messages.append({"role": "user", "content": message})

    tools = TOOL_DEFINITIONS if use_memory else TOOL_DEFINITIONS[1:2]  # 关记忆时只留 search_knowledge
    status = {"recalled": 0, "knowledge": 0, "stored": 0}

    for step in range(int(max_iter)):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=temperature, top_p=top_p,
            )
            choice = resp.choices[0].message
        except Exception as e:
            assistant_msg["content"] += f"\n\n⚠️ API 调用异常: {e}"
            yield history, _clean(messages), _status_md(status, mgr, session_id)
            return

        has_tool_calls = bool(choice.tool_calls)
        msg_dict = {"role": "assistant", "content": choice.content or None}
        if has_tool_calls:
            msg_dict["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.tool_calls
            ]
            messages.append(msg_dict)

            # 展示调用过程
            for tc in choice.tool_calls:
                icon = TOOL_ICONS.get(tc.function.name, "🔧")
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                args_str = "、".join(f"{k}={v}" for k, v in args.items())
                assistant_msg["content"] += f"{icon} **{tc.function.name}** `{args_str[:80]}`\n"
            yield history, _clean(messages), _status_md(status, mgr, session_id)

            # 并发执行工具
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {}
                for tc in choice.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    fut = pool.submit(_exec_tool, tc.function.name, args, scope, owner)
                    futures[fut] = (tc, args)
                for fut in concurrent.futures.as_completed(futures):
                    tc, args = futures[fut]
                    result = fut.result()
                    name = tc.function.name
                    if name == "recall_memory":
                        status["recalled"] += 1
                    elif name == "search_knowledge":
                        status["knowledge"] += 1
                    elif name == "store_memory":
                        status["stored"] += 1
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            yield history, _clean(messages), _status_md(status, mgr, session_id)
        else:
            # 无工具调用 → 收尾
            messages.append(msg_dict)
            final = choice.content or ""
            if assistant_msg["content"]:
                assistant_msg["content"] += "\n" + final
            else:
                assistant_msg["content"] = final or "（无回复）"
            break
    else:
        assistant_msg["content"] += "\n\n⚠️ 已达最大推理步数。"

    # 3) 记录助手消息到 L0 + 满 N 轮则聚合 Episode
    final_text = assistant_msg["content"]
    mgr.record_message(session_id, "assistant", final_text, scope, owner)
    closed_ep = mgr.close_episode(session_id, force=False)
    if closed_ep:
        status["episode"] = closed_ep  # type: ignore

    yield history, _clean(messages), _status_md(status, mgr, session_id, closed_ep)


def _clean(messages: list) -> list:
    """返回可安全存入 State 的消息（去掉 system，保留多轮上下文）"""
    return [m for m in messages if m.get("role") != "system"][-20:]


def _status_md(status: dict, mgr, session_id: str, closed_ep=None) -> str:
    s = mgr.stats()
    ep_line = ""
    if closed_ep is not None:
        ep_line = f" · 🆕 本轮聚合 Episode：{closed_ep.summary[:24]}（tags: {'、'.join(closed_ep.tags)}）"
    pending = s.get("L3_pending", 0)
    pend_line = f" · ⏳ 待审批 skill {pending}" if pending else ""
    return (
        f"🧠 recall×{status.get('recalled',0)} · 📚 knowledge×{status.get('knowledge',0)}"
        f" · 📝 store×{status.get('stored',0)}{ep_line}{pend_line}\n\n"
        f"记忆库 → L0 会话 {s['sessions']} · L1 Episode {s['L1_episodes']} · "
        f"L2 Structure {s['L2_structures']} · L3 Skill {s['L3_approved']}/{s['L3_skills']}(已批/总)"
    )
