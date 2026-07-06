"""
项目1: AI 对话助手
基于 DeepSeek API，支持流式输出、参数调节、智能追问推荐
"""

import os
import json
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# ---------- 配置 ----------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ==================== 核心功能 ====================

def respond(message, history, system_prompt, temperature, top_p, max_tokens,
            top_k, presence_penalty, frequency_penalty):
    """流式对话生成"""

    # 添加用户消息到对话
    history.append({"role": "user", "content": message})

    # 构建 API 消息
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    # 参数
    kwargs = dict(
        model=MODEL, messages=messages,
        temperature=temperature, top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stream=True,
    )
    if top_k is not None:
        kwargs["extra_body"] = {"top_k": top_k}

    try:
        response = client.chat.completions.create(**kwargs)
        history.append({"role": "assistant", "content": ""})
        yield history, ""  # 显示用户消息

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                history[-1]["content"] += delta.content
                yield history, ""

    except Exception as e:
        error_msg = f"\n\n⚠️ 调用出错: {str(e)}"
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] += error_msg
        else:
            history.append({"role": "assistant", "content": error_msg})
        yield history, ""


def gen_suggestions(chat_history):
    """根据对话历史生成 3 条追问建议"""
    if not chat_history or len(chat_history) < 2:
        return [gr.update(visible=False)] * 3

    # 取最近几轮对话生成建议
    recent = chat_history[-6:] if len(chat_history) > 6 else chat_history

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是一个对话助手。根据对话历史，列出3个用户最可能继续问的简短问题。"
                                              "只返回问题，用以下JSON格式：[\"问题1\", \"问题2\", \"问题3\"]"},
                *recent,
                {"role": "user", "content": "根据以上对话，推荐3个相关的追问，用JSON数组返回。"}
            ],
            temperature=0.7,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        # 尝试解析 JSON
        questions = json.loads(text)
        if not isinstance(questions, list) or len(questions) < 3:
            raise ValueError("格式不对")
    except Exception:
        # 如果 JSON 解析失败，回退方案
        questions = [
            "能展开讲讲吗？",
            "有什么实际应用？",
            "和别的方案相比呢？",
        ]

    return [
        gr.update(value=questions[0], visible=True),
        gr.update(value=questions[1], visible=True),
        gr.update(value=questions[2], visible=True),
    ]


def clear_all():
    """清空对话"""
    return [], "", *[gr.update(visible=False)] * 3


# ==================== UI ====================

CUSTOM_CSS = """
footer { display: none !important; }
.parameter-row { gap: 12px; }
.followup-row { gap: 8px; margin-top: -12px; }
.followup-row button {
    font-size: 12px !important; padding: 4px 12px !important;
    border-radius: 16px !important; min-height: 0 !important;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
"""

with gr.Blocks(title="AI 对话助手 - DeepSeek", fill_height=True, css=CUSTOM_CSS) as demo:
    gr.Markdown("""
        # 🤖 AI 对话助手
        基于 **DeepSeek** 大模型 · 流式输出 · 智能追问推荐
    """)

    # ====== 参数区（始终可见） ======
    with gr.Row(elem_classes="parameter-row"):
        temperature = gr.Slider(0.0, 2.0, 0.7, step=0.1,
                                label="🌡️ 温度", info="越高越有创意")
        top_p = gr.Slider(0.0, 1.0, 0.9, step=0.05,
                          label="🎯 Top P", info="核采样阈值")
        max_tokens = gr.Slider(256, 8192, 4096, step=128,
                               label="📏 最大 Token", info="回复长度上限")

    # ====== 高级设置（可折叠） ======
    with gr.Accordion("⚙️ 高级设置", open=False):
        with gr.Row(elem_classes="parameter-row"):
            top_k = gr.Slider(1, 20, 10, step=1,
                              label="🔝 Top K", info="取概率最高的 K 个词采样")
            presence_penalty = gr.Slider(-2.0, 2.0, 0.0, step=0.1,
                                         label="💬 话题新鲜度", info="正值鼓励谈论新话题")
            frequency_penalty = gr.Slider(-2.0, 2.0, 0.0, step=0.1,
                                          label="🔄 重复惩罚", info="正值减少重复用词")
        system_prompt = gr.Textbox(
            label="🎯 系统提示词",
            value="你是一个有帮助的AI助手。请用中文回答用户的问题。回答简洁清晰，有逻辑。",
            lines=3,
        )

    # ====== 聊天区域 ======
    chatbot = gr.Chatbot(label="💬 对话", placeholder="开始对话…", height=400)

    # ====== 追问推荐按钮 ======
    with gr.Row(elem_classes="followup-row") as followup_row:
        fb1 = gr.Button("", visible=False, size="sm")
        fb2 = gr.Button("", visible=False, size="sm")
        fb3 = gr.Button("", visible=False, size="sm")

    # ====== 输入区域 ======
    with gr.Row():
        msg = gr.Textbox(label="", placeholder="输入你的问题，按 Enter 发送…",
                         scale=8, container=False)
        send_btn = gr.Button("🚀 发送", scale=1, variant="primary")
        clear_btn = gr.Button("🗑️ 清空", scale=1)

    gr.Markdown("---\n*项目1: AI 对话助手 · 大模型应用开发实战*")


    # ==================== 事件绑定 ====================

    inputs = [msg, chatbot, system_prompt, temperature, top_p, max_tokens,
              top_k, presence_penalty, frequency_penalty]
    outputs = [chatbot, msg]
    suggest_outputs = [fb1, fb2, fb3]
    all_outputs = outputs + suggest_outputs

    # 发送 / Enter → 对话 → 生成追问
    send_btn.click(respond, inputs, outputs, concurrency_limit=1) \
            .then(gen_suggestions, [chatbot], suggest_outputs)

    msg.submit(respond, inputs, outputs, concurrency_limit=1) \
       .then(gen_suggestions, [chatbot], suggest_outputs)

    # 点击追问 → 填入输入框
    fb1.click(lambda q: q, [fb1], [msg])
    fb2.click(lambda q: q, [fb2], [msg])
    fb3.click(lambda q: q, [fb3], [msg])

    # 清空 → 重置所有
    clear_btn.click(clear_all, None, all_outputs)


# ==================== 启动 ====================

if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_api_key_here":
        print("⚠️  请先在 .env 中填写 DEEPSEEK_API_KEY")
        print("🔑 https://platform.deepseek.com/api_keys")
        exit(1)

    print(f"[启动] AI 对话助手")
    print(f"[地址] http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                theme=gr.themes.Soft())
