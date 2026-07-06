"""
项目3: AI Agent 智能助手（增强版）
核心机制: Function Calling + ReAct 循环 + 多轮对话记忆
技术栈: DeepSeek Function Calling + Gradio + httpx

面试亮点:
  1. 手写 ReAct 推理循环，而非调 LangChain 封装
  2. 支持多轮对话中保留工具调用历史
  3. 并发执行多个工具调用
  4. 透明展示推理链（Thinking Chain-of-Thought）
"""

import os
import json
import datetime
import re
import concurrent.futures
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. 工具定义：用 JSON Schema 描述每个工具的接口             ║
# ║  这是 Function Calling 的核心——LLM 通过 Schema 理解工具     ║
# ╚══════════════════════════════════════════════════════════════╝

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除、幂运算、三角函数等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '123 * 456' 或 '2 ** 10' 或 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "获取指定城市的当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名（中文或英文），如 '北京'、'New York'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索网络获取最新信息，如新闻、百科、资讯等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，越具体越准确",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认5",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. 工具执行：收到 LLM 的调用请求后，真正执行对应的函数      ║
# ╚══════════════════════════════════════════════════════════════╝

def safe_eval(expr: str) -> str:
    """安全计算数学表达式——白名单方式，禁止执行任意代码"""
    # 支持 math 模块的常用函数
    import math
    allowed_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("__")
    }
    # 替换常见写法
    expr = expr.replace("^", "**")
    try:
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"{expr} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def execute_tool(name: str, args: dict) -> str:
    """执行工具——每个工具独立实现，结果返回给 LLM 继续推理"""

    if name == "calculator":
        return safe_eval(args.get("expression", ""))

    elif name == "get_datetime":
        city = args.get("city", "北京")
        now = datetime.datetime.now()
        return (f"{city}的当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({now.strftime('%A')})")

    elif name == "web_search":
        query = args.get("query", "")
        max_results = min(args.get("max_results", 5), 10)
        results = []

        # 优先 DuckDuckGo（免费、无需 API Key）
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception:
            pass

        # 备选：Bing 解析
        if not results:
            try:
                import httpx
                resp = httpx.get(
                    f"https://www.bing.com/search?q={query}",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10, follow_redirects=True,
                )
                if resp.status_code == 200:
                    titles = re.findall(
                        r'<h2><a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', resp.text
                    )
                    for url, title in titles[:max_results]:
                        results.append({
                            "title": re.sub(r"<[^>]+>", "", title),
                            "body": "",
                            "href": url,
                        })
            except Exception:
                pass

        if not results:
            return "搜索功能在当前网络不可用，请稍后重试或使用其他工具。"

        output = []
        for i, r in enumerate(results[:max_results], 1):
            output.append(
                f"[{i}] {r.get('title', '无标题')}\n"
                f"    {r.get('body', r.get('snippet', ''))}\n"
                f"    链接: {r.get('href', '')}"
            )
        return "\n\n".join(output)

    return f"未知工具: {name}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. Agent 核心：ReAct 推理循环                               ║
# ║                                                              ║
# ║  ReAct = Reasoning + Acting                                  ║
# ║  1) LLM 思考 → 决定是直接回答还是调用工具                    ║
# ║  2) 如果要调用工具 → 解析工具名和参数 → 执行 → 返回结果     ║
# ║  3) 把工具结果送回 LLM → LLM 继续思考                       ║
# ║  4) 重复直到 LLM 认为可以给出最终答案                       ║
# ╚══════════════════════════════════════════════════════════════╝

def agent_chat(
    message, history, messages_state, system_prompt,
    max_iter, temperature, top_p
):
    """
    ReAct 循环核心函数

    参数:
        message: 用户当前输入
        history: Gradio 聊天记录（用于显示）
        messages_state: 完整 API 消息历史（含 tool_calls 和 tool 结果）
        system_prompt: 系统提示词
        max_iter: 最大推理步数
    """
    MAX_ITER = int(max_iter)

    # === 第一步：重建完整的消息上下文 ===
    # messages_state 保存了之前所有轮次的 tool_calls 和 tool 结果
    # 这样多轮对话中，Agent 能记住之前调用过什么工具、得到了什么结果
    messages = [{"role": "system", "content": system_prompt}]
    if messages_state:
        messages.extend(messages_state)
    messages.append({"role": "user", "content": message})

    # 更新显示
    history.append({"role": "user", "content": message})
    yield history, "", messages_state

    # 创建一个助理消息，逐步追加内容（展示推理链）
    assistant_msg = {"role": "assistant", "content": ""}
    history.append(assistant_msg)

    for step in range(MAX_ITER):
        # === 第二步：调用 LLM，传入工具定义 ===
        # 关键 API 参数: tools + tool_choice="auto"
        # LLM 会判断是否需要调工具，需要的话返回 tool_calls
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=temperature,
                top_p=top_p,
            )
            choice = resp.choices[0].message
        except Exception as e:
            assistant_msg["content"] += f"\n\n⚠️ API 调用出错: {str(e)}"
            yield history, "", messages
            return

        has_content = bool(choice.content)
        has_tool_calls = bool(choice.tool_calls)

        # === 第三步：记录 LLM 的思考过程 ===
        if has_content:
            assistant_msg["content"] += choice.content + "\n\n"

        # 构造 API 用的 assistant 消息（包含 tool_calls）
        msg_dict = {"role": "assistant", "content": choice.content or None}
        if has_tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.tool_calls
            ]

        if has_tool_calls:
            # === 第四步：LLM 决定调用工具 → 执行 ===
            messages.append(msg_dict)

            # 并发执行多个工具（如果 LLM 一次性请求多个）
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {}
                for tc in choice.tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    # 显示正在调用的工具
                    icon = {"calculator": "🔢", "get_datetime": "🕐", "web_search": "🔍"}.get(func_name, "🔧")
                    args_str = "、".join(f"{k}={v}" for k, v in args.items())
                    assistant_msg["content"] += (
                        f"{icon} **Step {step + 1} → 调用: {func_name}**\n"
                        f"> 参数: `{args_str}`\n"
                    )
                    yield history, "", messages

                    # 提交执行
                    future = pool.submit(execute_tool, func_name, args)
                    futures[future] = (tc, func_name, args)

                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    tc, func_name, args = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = f"执行异常: {e}"

                    result_short = result[:150] + "..." if len(result) > 150 else result
                    assistant_msg["content"] += f"> 📊 结果: {result_short}\n\n"
                    yield history, "", messages

                    # 工具结果送回 LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            yield history, "", messages

        else:
            # === 第四步(备选)：LLM 直接回答 → 完成 ===
            messages.append(msg_dict)
            if not has_content:
                assistant_msg["content"] = choice.content or "（无回复）"
            yield history, "", messages
            return

    # 超过最大迭代次数
    assistant_msg["content"] += "\n\n⚠️ 已达最大推理步数，请简化问题。"
    yield history, "", messages


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. 清空对话                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def clear_all():
    return [], "", None


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. Gradio UI                                               ║
# ╚══════════════════════════════════════════════════════════════╝

CUSTOM_CSS = """
footer { display: none !important; }
.thinking-box { background: #f0f7ff; border-radius: 8px; padding: 12px; }
"""

with gr.Blocks(title="AI Agent - DeepSeek", fill_height=True) as demo:
    gr.Markdown("""
        # 🤖 AI Agent 智能助手 · 增强版
        基于 **ReAct (Reasoning + Acting)** 模式 · 支持**多步推理**、**并发工具调用**
    """)

    with gr.Row():
        temperature = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="🌡️ 温度")
        top_p = gr.Slider(0.0, 1.0, 0.8, step=0.05, label="🎯 Top P")
        max_iter = gr.Slider(1, 10, 5, step=1, label="🔄 最大推理步数")

    with gr.Accordion("🎯 系统提示词", open=False):
        system_prompt = gr.Textbox(
            label="System Prompt",
            value="你是一个智能AI助手，可以使用工具来完成任务。"
                  "对于复杂任务，请一步步思考，选择合适的工具。"
                  "用中文回答，答案简洁清晰。",
            lines=2,
        )

    gr.Markdown("""
        💡 **示例任务** 👇
        `计算 2^10 等于多少？`
        `现在几点了？`
        `搜索一下2026年AI领域的最新进展`
        `帮我计算 3.14 * 25 再搜索一下圆周率的历史`
    """)

    chatbot = gr.Chatbot(
        label="💬 对话",
        placeholder="输入任务，AI 会调用工具来完成…",
        height=400,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="", placeholder="输入你的任务…",
            scale=8, container=False,
        )
        send_btn = gr.Button("🚀 执行", scale=1, variant="primary")
        clear_btn = gr.Button("🗑️ 清空", scale=1)

    gr.Markdown("---\n*原理: User → LLM(思考) → [调用工具 → 执行 → 结果回传]ⁿ → 最终答案*")

    # ====== 状态 ======
    messages_state = gr.State(None)  # 跨轮次保存完整消息历史

    # ====== 事件 ======
    inputs = [
        msg, chatbot, messages_state,
        system_prompt, max_iter, temperature, top_p,
    ]
    outputs = [chatbot, msg, messages_state]

    send_btn.click(agent_chat, inputs, outputs, concurrency_limit=1)
    msg.submit(agent_chat, inputs, outputs, concurrency_limit=1)
    clear_btn.click(clear_all, None, outputs)


if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_api_key_here":
        print("⚠️  请先配置 DEEPSEEK_API_KEY")
        exit(1)

    print(f"[启动] AI Agent 智能助手（增强版）")
    print(f"[地址] http://localhost:7862")
    print(f"[亮点] 多轮记忆 | 并发工具调用 | 透明推理链")
    demo.launch(
        server_name="127.0.0.1", server_port=7862,
        theme=gr.themes.Soft(), css=CUSTOM_CSS,
    )
