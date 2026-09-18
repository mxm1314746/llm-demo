"""
项目2: 记忆增强 RAG 智能体
文档 RAG（Chroma）+ 四层记忆（L0 会话 / L1 Episode / L2 Structure / L3 Skill）
Agent 通过进程内工具调用，每轮主动 recall 长期记忆、按需检索知识库。

技术栈: LangChain + ChromaDB(双collection) + DeepSeek Function Calling + Gradio
启动: python app.py  →  http://localhost:7861
"""
import os
import sys
import uuid
import shutil

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

import ui_theme as ui
import rag_store
from memory.manager import init_memory_manager, get_memory_manager
from memory.pipeline import MemoryConfig
import agent.core as agent_core

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 初始化分层记忆管理器（默认配置，UI 滑块可实时覆盖）
init_memory_manager(client, MODEL, MemoryConfig(
    episode_rounds=3, session_timeout=300, debounce_threshold=3,
    tag_similarity=0.5, require_approval=True,
))


# ╔══════════════════════════════════════════════════════════════╗
# ║  文档处理（知识库 collection）                               ║
# ╚══════════════════════════════════════════════════════════════╝

SUPPORTED = {".pdf", ".txt", ".md"}


def process_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED:
        return False, f"不支持的文件格式: {ext}"
    try:
        loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        if not docs:
            return False, "文档内容为空"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return False, "分割后无有效内容"
        name = os.path.basename(file_path)
        for c in chunks:
            c.metadata["source_file"] = name
        n = rag_store.add_documents(chunks)
        return True, f"✅ 成功处理「{name}」— 共 {n} 个片段（已入 knowledge 库）"
    except Exception as e:
        return False, f"❌ 处理失败: {e}"


def upload_file(file):
    if file is None:
        return "请选择文件", _file_list_md()
    dst = os.path.join(rag_store.KNOWLEDGE_DIR, os.path.basename(file.name))
    shutil.copy2(file.name, dst)
    ok, msg = process_document(dst)
    if not ok:
        os.remove(dst)
    return msg, _file_list_md()


def _file_list_md():
    files = rag_store.list_document_files()
    if not files:
        return "（暂无文档）"
    return "\n".join(f"- {f}" for f in files)


def clear_knowledge():
    rag_store.clear_all_files()
    rag_store.clear_knowledge()
    return "✅ 知识库已清空", _file_list_md()


# ╔══════════════════════════════════════════════════════════════╗
# ║  对话（记忆增强 Agent）                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def respond(message, history, messages_state, session_id, scope, use_memory,
            temperature, top_p, max_iter, top_k, ep_rounds, debounce, timeout):
    if not message or not message.strip():
        yield history, messages_state, ""
        return
    mgr = get_memory_manager()
    # 应用 UI 配置
    mgr.cfg.episode_rounds = int(ep_rounds)
    mgr.cfg.debounce_threshold = int(debounce)
    mgr.cfg.session_timeout = int(timeout)
    mgr.check_timeout()  # 顺带刷新超时会话

    gen = agent_core.chat(
        client, MODEL, message, history, messages_state, session_id,
        scope=scope, owner="default", use_memory=use_memory,
        temperature=temperature, top_p=top_p, max_iter=int(max_iter), top_k=int(top_k),
    )
    for h, ms, st in gen:
        yield h, ms, st


def new_session():
    return [], None, f"sess-{uuid.uuid4().hex[:8]}", ""


# ╔══════════════════════════════════════════════════════════════╗
# ║  记忆分层可视化                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def _stats_md():
    mgr = get_memory_manager()
    mgr.check_timeout()
    s = mgr.stats()
    return (
        f"| 层级 | 内容 | 数量 |\n|---|---|:--:|\n"
        f"| L0 | 会话 / 原始消息 | {s['sessions']} / {s['L0_messages']} |\n"
        f"| L1 | Episode（N轮聚合+tags） | {s['L1_episodes']} |\n"
        f"| L2 | Structure（相似tag归并） | {s['L2_structures']} |\n"
        f"| L3 | Skill（已批/总/待审） | {s['L3_approved']}/{s['L3_skills']}/{s['L3_pending']} |\n"
        f"| RAG | 已索引 skill 向量 | {s['skills_indexed']} |"
    )


def _episodes_md():
    eps = get_memory_manager().list_episodes()[-12:][::-1]
    if not eps:
        return "_暂无 Episode_"
    rows = ["| 时间 | 轮 | tags | 摘要 |", "|---|:--:|---|---|"]
    import datetime as _dt
    for e in eps:
        t = _dt.datetime.fromtimestamp(e.end_ts).strftime("%m-%d %H:%M")
        rows.append(f"| {t} | {e.rounds} | {'、'.join(e.tags)} | {e.summary[:36]} |")
    return "\n".join(rows)


def _structures_md():
    sts = get_memory_manager().list_structures()
    if not sts:
        return "_暂无 Structure_"
    rows = ["| 主标签 | 合并次数 | episode数 | 摘要 | skill |", "|---|:--:|:--:|---|:--:|"]
    for st in sts:
        rows.append(f"| {st.tag} | {st.merge_count} | {len(st.episode_ids)} | {st.summary[:40]} | {'✅' if st.skill_id else '—'} |")
    return "\n".join(rows)


def _skills_md():
    sks = get_memory_manager().list_skills()
    if not sks:
        return "_暂无 Skill_"
    badge = {"pending": "⏳待审", "approved": "✅已批", "rejected": "❌已拒"}
    rows = ["| 标签 | 状态 | 内容 |", "|---|:--:|---|"]
    for s in sks:
        rows.append(f"| {s.tag} | {badge.get(s.status, s.status)} | {s.content[:50]} |")
    return "\n".join(rows)


def _pending_choices():
    pend = get_memory_manager().pending_skills()
    return gr.update(choices=[f"{s.id}｜{s.tag}" for s in pend],
                     value=(f"{pend[0].id}｜{pend[0].tag}" if pend else None))


def _review(skill_choice, action):
    if not skill_choice:
        return "请选择一个待审批 skill", _stats_md(), _skills_md(), _pending_choices()
    skill_id = skill_choice.split("｜")[0]
    mgr = get_memory_manager()
    if action == "approve":
        s = mgr.approve_skill(skill_id)
        msg = f"✅ 已批准并写入向量库：{s.tag}" if s else "未找到该 skill"
    else:
        s = mgr.reject_skill(skill_id)
        msg = f"❌ 已拒绝：{s.tag}" if s else "未找到该 skill"
    return msg, _stats_md(), _skills_md(), _pending_choices()


def refresh_memory():
    return _stats_md(), _episodes_md(), _structures_md(), _skills_md(), _pending_choices()


def wipe_memory():
    from memory import store as mstore
    mstore.wipe_all()
    # 清空 skills 向量库
    try:
        rag_store._collection(rag_store.SKILLS_COLLECTION).delete_collection()  # noqa: SLF001
    except Exception:
        pass
    return refresh_memory()


# ╔══════════════════════════════════════════════════════════════╗
# ║  UI                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

CUSTOM_CSS = ui.THEME_CSS + """
.mem-status { font-size: 12px; color: #555; background:#f5f8ff; border-radius:8px; padding:6px 10px; }
"""

with gr.Blocks(title="记忆增强 RAG 智能体", fill_height=True) as demo:
    gr.HTML(ui.header_html("D2", "记忆增强 RAG 智能体",
                           "文档RAG + 四层记忆(L0会话·L1 Episode·L2 Structure·L3 Skill) · Agent 每轮主动召回"))

    session_id = gr.State(f"sess-{uuid.uuid4().hex[:8]}")
    messages_state = gr.State(None)

    with gr.Tabs():
        # ── Tab1 对话 ──
        with gr.TabItem("💬 记忆增强对话"):
            with gr.Row():
                scope = gr.Radio(choices=[("私有 private", "private"), ("团队 team", "team")],
                                 value="private", label="记忆范围", scale=3)
                use_memory = gr.Checkbox(value=True, label="启用记忆增强", scale=2)
                new_btn = gr.Button("🆕 新会话", scale=1)
            sess_label = gr.Markdown(f"当前会话：`{session_id.value}`")

            chatbot = gr.Chatbot(label="对话", height=420, elem_classes=["ui-chat"],
                                 placeholder="提问试试，例如：退票手续费怎么算？/ 记住我只坐靠窗座位")
            mem_status = gr.Markdown("", elem_classes=["mem-status"])

            with gr.Row():
                msg = gr.Textbox(label="", placeholder="输入问题，Enter 发送…", scale=8,
                                 container=False, elem_classes=["ui-search"])
                send_btn = gr.Button("🚀 发送", scale=1, variant="primary", elem_classes=["ui-btn-primary"])

            with gr.Accordion("⚙️ 参数与记忆配置", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="温度")
                    top_p = gr.Slider(0.0, 1.0, 0.8, step=0.05, label="Top P")
                    max_iter = gr.Slider(1, 8, 5, step=1, label="最大推理步数")
                    top_k = gr.Slider(1, 10, 4, step=1, label="文档 Top-K")
                with gr.Row():
                    ep_rounds = gr.Slider(1, 10, 3, step=1, label="L1: 每 N 轮成 Episode")
                    debounce = gr.Slider(1, 10, 3, step=1, label="L3 防抖: 合并 N 次提取 Skill")
                    timeout = gr.Slider(30, 1800, 300, step=30, label="会话超时(秒)强制聚合")

            gr.HTML(ui.footer_html("Demo2 · 记忆增强 RAG · 独立开发 GitHub @mxm1314746"))

        # ── Tab2 知识库 ──
        with gr.TabItem("📁 知识库"):
            gr.Markdown("上传 PDF / TXT / Markdown，切分后写入 **knowledge** 向量库，供 `search_knowledge` 工具检索。")
            with gr.Row():
                with gr.Column(scale=3):
                    file_input = gr.File(label="上传文档", file_types=[".pdf", ".txt", ".md"], file_count="single")
                    upload_btn = gr.Button("📤 上传并处理", variant="primary", elem_classes=["ui-btn-primary"])
                with gr.Column(scale=2):
                    file_list = gr.Textbox(label="已上传文档", value=_file_list_md(), lines=6, interactive=False)
                    clear_kb_btn = gr.Button("🗑️ 清空知识库", variant="stop")
            upload_status = gr.Textbox(label="处理状态")

        # ── Tab3 记忆分层 ──
        with gr.TabItem("🧠 记忆分层"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新", variant="primary", elem_classes=["ui-btn-primary"])
                wipe_btn = gr.Button("♻️ 清空全部记忆", variant="stop")
            stats_md = gr.Markdown(_stats_md())

            with gr.Accordion("🧩 L3 Skill 人工审批（防抖达标后进入此队列）", open=True):
                pending_dd = gr.Dropdown(label="待审批 Skill", choices=[], interactive=True)
                with gr.Row():
                    approve_btn = gr.Button("✅ 批准并入库", variant="primary")
                    reject_btn = gr.Button("❌ 拒绝")
                review_msg = gr.Markdown("")

            gr.Markdown("### L1 · Episode")
            ep_md = gr.Markdown(_episodes_md())
            gr.Markdown("### L2 · Structure")
            st_md = gr.Markdown(_structures_md())
            gr.Markdown("### L3 · Skill")
            sk_md = gr.Markdown(_skills_md())

    # ══ 事件绑定 ══
    chat_inputs = [msg, chatbot, messages_state, session_id, scope, use_memory,
                   temperature, top_p, max_iter, top_k, ep_rounds, debounce, timeout]
    chat_outputs = [chatbot, messages_state, mem_status]

    send_btn.click(respond, chat_inputs, chat_outputs, concurrency_limit=1)
    msg.submit(respond, chat_inputs, chat_outputs, concurrency_limit=1)

    def _new():
        h, ms, sid, st = new_session()
        return h, ms, sid, st, f"当前会话：`{sid}`"
    new_btn.click(_new, None, [chatbot, messages_state, session_id, mem_status, sess_label])

    upload_btn.click(upload_file, [file_input], [upload_status, file_list])
    clear_kb_btn.click(clear_knowledge, None, [upload_status, file_list])

    mem_outputs = [stats_md, ep_md, st_md, sk_md, pending_dd]
    refresh_btn.click(refresh_memory, None, mem_outputs)
    wipe_btn.click(wipe_memory, None, mem_outputs)
    approve_btn.click(lambda c: _review(c, "approve"), [pending_dd], [review_msg, stats_md, sk_md, pending_dd])
    reject_btn.click(lambda c: _review(c, "reject"), [pending_dd], [review_msg, stats_md, sk_md, pending_dd])
    # 打开记忆 Tab 时自动刷新
    demo.load(refresh_memory, None, mem_outputs)


if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_api_key_here":
        print("⚠️  请先在 .env 中填写 DEEPSEEK_API_KEY")
        exit(1)
    print("[启动] 记忆增强 RAG 智能体")
    print("[地址] http://localhost:7861")
    demo.launch(server_name="127.0.0.1", server_port=7861,
                theme=gr.themes.Soft(), css=CUSTOM_CSS)
