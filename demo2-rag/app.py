"""
项目2: RAG 知识库问答系统
上传文档 → AI 基于文档内容回答你的问题
技术栈: LangChain + ChromaDB + DeepSeek + Gradio
"""

import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 国内 HuggingFace 镜像（用于下载嵌入模型）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------- 配置 ----------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

# 路径
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------- 初始化 Embedding ----------
def get_embeddings():
    """获取嵌入模型（小模型，约30MB）"""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ---------- 文档处理 ----------
SUPPORTED_EXTENSIONS = {".pdf": "PDF", ".txt": "文本", ".md": "Markdown"}


def process_document(file_path):
    """处理单个文档：加载 → 分割 → 向量化 → 存入 Chroma"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"不支持的文件格式: {ext}"

    try:
        # 1. 加载文档
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:  # .txt, .md
            loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        if not docs:
            return False, "文档内容为空，请检查文件"

        # 2. 分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return False, "分割后无有效内容"

        # 给每个 chunk 加上来源文件名
        filename = os.path.basename(file_path)
        for c in chunks:
            c.metadata["source_file"] = filename

        # 3. 向量化 + 存入 Chroma
        embeddings = get_embeddings()
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )

        return True, f"✅ 成功处理「{filename}」- 共 {len(chunks)} 个片段"

    except Exception as e:
        return False, f"❌ 处理失败: {str(e)}"


def get_document_list():
    """获取已上传文档列表"""
    files = []
    if os.path.exists(KNOWLEDGE_DIR):
        for f in os.listdir(KNOWLEDGE_DIR):
            fp = os.path.join(KNOWLEDGE_DIR, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
                files.append(f"{f} ({size_str})")
    return files if files else ["（暂无文档）"]


def clear_knowledge():
    """清空知识库"""
    for f in os.listdir(KNOWLEDGE_DIR):
        fp = os.path.join(KNOWLEDGE_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
    # 重建 Chroma
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR)
    return "✅ 知识库已清空", "（暂无文档）", [], ""


def upload_file(file):
    """上传并处理文件"""
    if file is None:
        return "请选择文件", "\n".join(get_document_list()), [], ""

    # 复制到知识库目录
    dst = os.path.join(KNOWLEDGE_DIR, os.path.basename(file.name))
    shutil.copy2(file.name, dst)

    # 处理
    success, msg = process_document(dst)
    if not success:
        # 处理失败，删除文件
        os.remove(dst)
        return msg, "\n".join(get_document_list()), [], ""

    return msg, "\n".join(get_document_list()), [], ""


# ---------- 检索 + 生成 ----------
def get_vectorstore():
    """获取 Chroma 向量库（如果存在）"""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return None
    try:
        embeddings = get_embeddings()
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    except Exception:
        return None


def query_knowledge(question, history, top_k):
    """RAG 问答：检索 + 生成"""
    vectorstore = get_vectorstore()
    if not vectorstore:
        history.append({"role": "assistant", "content": "📄 知识库为空，请先上传文档后再提问。"})
        yield history, ""
        return

    # 检索相关片段
    try:
        docs = vectorstore.similarity_search(question, k=top_k)
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ 检索失败: {str(e)}"})
        yield history, ""
        return

    if not docs:
        history.append({"role": "assistant", "content": "📄 未找到相关内容，请换一个问题或上传更多文档。"})
        yield history, ""
        return

    # 构建上下文
    context_parts = []
    source_info = {}
    for i, doc in enumerate(docs):
        context_parts.append(f"[片段{i + 1}] {doc.page_content}")
        src = doc.metadata.get("source_file", "未知来源")
        if src not in source_info:
            source_info[src] = []
        source_info[src].append(i + 1)

    context = "\n\n---\n\n".join(context_parts)
    sources_str = "、".join([f"「{s}」(片段{n})" for s, n in source_info.items()
                            for n in n])

    # 构建 RAG 提示
    system_prompt = f"""你是一个基于知识库的问答助手。请根据以下提供的资料回答问题。

## 参考资料
{context}

## 要求
1. 仅基于参考资料回答，如果资料中没有相关信息，请如实说"资料中未找到相关内容"
2. 回答要简洁清晰、有逻辑
3. 在回答末尾标注参考来源"""

    # 用户消息
    history.append({"role": "user", "content": question})
    yield history, ""

    # 调用 API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )

        history.append({"role": "assistant", "content": ""})
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                history[-1]["content"] += delta.content
            yield history, ""

        # 添加来源标注
        history[-1]["content"] += f"\n\n---\n📚 **参考来源**: {sources_str}"
        yield history, ""

    except Exception as e:
        error_msg = f"\n\n⚠️ 调用出错: {str(e)}"
        if history[-1]["role"] == "assistant":
            history[-1]["content"] += error_msg
        else:
            history.append({"role": "assistant", "content": error_msg})
        yield history, ""


# ---------- 清空对话 ----------
def clear_chat():
    return [], ""


# ==================== UI ====================

CUSTOM_CSS = """
footer { display: none !important; }
.upload-box { min-height: 100px; }
"""

with gr.Blocks(title="RAG 知识库问答 - DeepSeek", fill_height=True) as demo:
    gr.Markdown("""
        # 📚 RAG 知识库问答系统
        上传 **PDF / TXT / Markdown** 文档，AI 基于文档内容回答你的问题
    """)

    # ====== 知识库管理 ======
    with gr.Accordion("📁 知识库管理", open=True):
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="上传文档（支持 PDF / TXT / MD）",
                    file_types=[".pdf", ".txt", ".md"],
                    file_count="single",
                )
                upload_btn = gr.Button("📤 上传并处理", variant="primary")
            with gr.Column(scale=2):
                file_list = gr.Textbox(
                    label="已上传文档",
                    value="（暂无文档）",
                    lines=4,
                    interactive=False,
                )
                with gr.Row():
                    top_k = gr.Slider(1, 10, 4, step=1,
                                      label="检索参考数 (Top-K)")
                    clear_kb_btn = gr.Button("🗑️ 清空知识库", variant="stop")

        upload_status = gr.Textbox(label="处理状态")

    # ====== 对话区域 ======
    chatbot = gr.Chatbot(label="💬 问答对话", placeholder="上传文档后，在这里提问…", height=400)

    with gr.Row():
        msg = gr.Textbox(label="", placeholder="输入你的问题，按 Enter 发送…",
                         scale=8, container=False)
        send_btn = gr.Button("🚀 发送", scale=1, variant="primary")
        clear_chat_btn = gr.Button("🗑️ 清空对话", scale=1)

    gr.Markdown("---\n*项目2: RAG 知识库问答系统 · 大模型应用开发实战*")


    # ==================== 事件绑定 ====================

    # 上传文件
    upload_btn.click(
        upload_file, [file_input],
        [upload_status, file_list, chatbot, msg],
    )

    # 清空知识库
    clear_kb_btn.click(clear_knowledge, None, [upload_status, file_list, chatbot, msg])

    # 对话
    inputs = [msg, chatbot, top_k]
    outputs = [chatbot, msg]

    send_btn.click(query_knowledge, inputs, outputs, concurrency_limit=1)
    msg.submit(query_knowledge, inputs, outputs, concurrency_limit=1)

    # 清空对话
    clear_chat_btn.click(clear_chat, None, outputs)


if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_api_key_here":
        print("⚠️  请先在 .env 中填写 DEEPSEEK_API_KEY")
        exit(1)

    print(f"[启动] RAG 知识库问答系统")
    print(f"[地址] http://localhost:7860")
    print(f"[提示] 上传 PDF/TXT 文档后即可提问")
    demo.launch(server_name="127.0.0.1", server_port=7861,
                theme=gr.themes.Soft(),
                css=CUSTOM_CSS)
