"""
Demo4: Multi-Agent 量化研究协作平台
多个专业 Agent 通过 Orchestrator 协作完成复杂任务
"""

# ====== 国内 HuggingFace 镜像（必须在最前面设置，否则 ChromaDB 下载慢）======
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import time
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from core.orchestrator import Orchestrator
from core.memory import SharedMemory
from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_agent import ReportAgent

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ==================== 初始化系统 ====================

def init_system():
    """初始化 Multi-Agent 系统"""
    persist_dir = os.path.join(os.path.dirname(__file__), "memory_data")
    memory = SharedMemory(persist_dir=persist_dir)

    orchestrator = Orchestrator(client, memory)

    # 注册专业 Agent
    orchestrator.register_agent(DataAgent(client, memory))
    orchestrator.register_agent(AnalysisAgent(client, memory))
    orchestrator.register_agent(ReportAgent(client, memory))

    return orchestrator


# 全局实例
orchestrator = init_system()


# ==================== 核心逻辑 ====================

def run_multi_agent(message, history):
    """运行 Multi-Agent 工作流，逐步输出执行过程"""
    if not message.strip():
        yield history, ""
        return

    # 显示用户消息
    history.append({"role": "user", "content": message})

    # 显示开始状态
    status_msg = {
        "role": "assistant",
        "content": "🤖 **Multi-Agent 系统启动**\n\n正在分析任务...",
    }
    history.append(status_msg)
    yield history, ""

    # 清空之前的状态
    orchestrator.execution_log = []
    for agent in orchestrator.agents.values():
        agent.status_history = []

    # 拆解任务
    task_defs = orchestrator.decompose_task(message)
    breakdown = "\n".join([f"  {i+1}. **{t['name']}** → {t['agent']}" for i, t in enumerate(task_defs)])
    status_msg["content"] = f"🤖 **Multi-Agent 系统启动**\n\n📋 **任务拆解** ({len(task_defs)} 个子任务):\n{breakdown}"
    yield history, ""

    # 逐个执行任务
    previous_results = {}
    intermediate_results = {}
    tasks = []

    for i, td in enumerate(task_defs):
        tasks.append(type('obj', (object,), {
            "name": td.get("name", f"任务{i+1}"),
            "description": td.get("description", ""),
            "agent": td.get("agent", ""),
            "status": "pending",
            "output": None,
        }))

    for i, task in enumerate(tasks):
        agent = orchestrator.agents.get(task.agent)
        if not agent:
            continue

        if previous_results:
            task.input = previous_results

        # 显示 Agent 开始工作
        status_msg["content"] += f"\n\n🔧 **[{agent.name}] 正在执行: {task.name}**"
        agent.update_status(agent.status.WORKING, task.name)
        yield history, ""

        try:
            # 执行
            result = agent.execute(task)
            task.status = "done"
            task.output = result
            intermediate_results[agent.name] = result

            # 保存到记忆
            orchestrator.memory.add_conversation("assistant", f"[{agent.name}] {result[:100]}...", agent.name)
            orchestrator.memory.remember(agent.name, f"task_{i}", result[:500])
            previous_results[agent.name] = result

            # 显示结果摘要
            preview = result[:200] + "..." if len(result) > 200 else result
            status_msg["content"] += f"\n✅ **完成!** 输出 {len(result)} 字符\n```\n{preview}\n```"

        except Exception as e:
            task.status = "failed"
            status_msg["content"] += f"\n❌ **执行失败**: {e}"
            intermediate_results[agent.name] = f"[失败] {e}"

        yield history, ""

    # 生成最终报告
    final_report = intermediate_results.get("报告撰写员", "")
    if not final_report:
        status_msg["content"] += "\n\n📊 **正在生成最终报告...**"
        yield history, ""
        final_report = orchestrator._summarize_results(message, intermediate_results)

    # 显示最终报告
    status_msg["content"] += f"\n\n---\n## 📊 最终报告\n\n{final_report}"
    yield history, ""

    # 保存到长期记忆
    orchestrator.memory.save_long_term("系统", f"task_{int(time.time())}",
                                        f"请求: {message}\n结果: {final_report[:500]}")


def clear_all():
    return [], ""


EXAMPLE_TASKS = [
    "搜索近期大模型领域的重要进展，分析技术趋势",
    "查一下Transformer架构的发展历程和关键里程碑",
    "分析大模型在金融领域的应用现状和未来趋势",
    "搜索量子计算的最新突破，评估其对AI的影响",
]

CUSTOM_CSS = """
footer { display: none !important; }
"""

with gr.Blocks(title="Multi-Agent 协作平台", fill_height=True, css=CUSTOM_CSS) as demo:
    gr.Markdown("""
        # 🤖 Multi-Agent 协作平台
        多个专业 AI Agent **协同工作** · Orchestrator 编排 · 共享记忆
    """)

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="💬 协作过程",
                placeholder="输入一个研究任务，多个 Agent 会协作完成…",
                height=500,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="", placeholder="例如: 搜索2026年AI融资事件并分析趋势",
                    scale=8, container=False,
                )
                send_btn = gr.Button("🚀 执行", scale=1, variant="primary")
                clear_btn = gr.Button("🗑️ 清空", scale=1)

            gr.Examples(
                examples=EXAMPLE_TASKS,
                inputs=msg,
                label="💡 试试这些研究任务",
            )

        with gr.Column(scale=3):
            gr.Markdown("### 🤖 Agent 团队")
            gr.Markdown("""
                | Agent | 职责 |
                |-------|------|
                | 🎯 **编排者** | 任务拆解、调度、汇总 |
                | 🔍 **数据采集员** | 搜索、收集信息 |
                | 📊 **数据分析师** | 计算、推理、分析 |
                | 📝 **报告撰写员** | 生成结构化报告 |
            """)

            gr.Markdown("### 🧠 记忆系统")
            memory_status = gr.Textbox(
                label="记忆状态",
                value=orchestrator.memory.get_status_summary(),
                lines=4,
                interactive=False,
            )

            gr.Markdown("### ⚙️ 工作原理")
            gr.Markdown("""
                ```
                用户请求
                   ↓
                Orchestrator 拆解任务
                   ↓
                并行/串行分配给 Agent
                   ↓
                每个 Agent 独立执行
                   ↓
                Orchestrator 汇总
                   ↓
                生成最终报告
                ```
            """)

    gr.Markdown("---\n*Demo4: Multi-Agent 协作 · 每个 Agent 都有自己的角色和工具*")

    # ====== 事件 ======
    inputs = [msg, chatbot]
    outputs = [chatbot, msg]

    send_btn.click(run_multi_agent, inputs, outputs, concurrency_limit=1)
    msg.submit(run_multi_agent, inputs, outputs, concurrency_limit=1)
    clear_btn.click(clear_all, None, outputs)


if __name__ == "__main__":
    print(f"[启动] Multi-Agent 协作平台")
    print(f"[地址] http://localhost:7863")
    print(f"[Agent] {' | '.join(orchestrator.agents.keys())}")
    print(f"[记忆] {orchestrator.memory.get_status_summary()}")
    demo.launch(server_name="127.0.0.1", server_port=7863,
                theme=gr.themes.Soft(), css=CUSTOM_CSS)
