"""
ReAct 推理循环 — 旅行 Agent 核心
复用 demo3-agent 的 agent_chat() 模式，适配 9 个航空工具
"""
import json
import re
import concurrent.futures
from openai import OpenAI
from .tools import execute_tool, TOOL_DEFINITIONS
from logger import logger


class TravelAgent:
    """
    南航智慧出行管家 Agent
    核心: ReAct 推理循环 + 工具编排

    工作流:
    User → LLM(思考) → [调用工具 → 执行 → 结果回传]ⁿ → 最终答案
    """

    def __init__(self, client: OpenAI, model: str = "deepseek-chat"):
        self.client = client
        self.model = model
        self.tracing = False
        self.trace = []

    # ═══ 对外接口 ═══

    def chat(
        self, message: str, history: list, messages_state: list | None,
        system_prompt: str, max_iter: int = 5,
        temperature: float = 0.5, top_p: float = 0.8,
    ):
        """
        ReAct 循环 generator（复用 demo3 的 yield 模式）

        参数:
            message: 用户当前输入
            history: Gradio Chatbot 显示用历史
            messages_state: 完整 API 消息历史（跨轮次）
            system_prompt: 系统提示词
            max_iter: 最大推理步数
        """
        MAX_ITER = int(max_iter)
        tool_icons = {
            "search_flights": "🔍", "book_ticket": "🎫", "check_in": "✅",
            "change_flight": "🔄", "cancel_ticket": "❌", "select_seat": "💺",
            "query_baggage": "🧳", "get_airport_info": "🏢", "query_policy": "📋",
            "query_order": "📦",
        }

        # 重建消息上下文
        messages = [{"role": "system", "content": system_prompt}]
        if messages_state:
            messages.extend(messages_state)
        messages.append({"role": "user", "content": message})

        # 更新 UI
        history.append({"role": "user", "content": message})
        yield history, "", messages_state

        # 创建 assistant 消息用于逐步追加
        assistant_msg = {"role": "assistant", "content": ""}
        history.append(assistant_msg)

        for step in range(MAX_ITER):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=temperature,
                    top_p=top_p,
                )
                choice = resp.choices[0].message
            except Exception as e:
                assistant_msg["content"] += f"\n\n⚠️ API 调用异常: {e}"
                yield history, "", messages
                return

            has_content = bool(choice.content)
            has_tool_calls = bool(choice.tool_calls)

            # 显示 LLM 的思考文本
            if has_content:
                assistant_msg["content"] += choice.content + "\n\n"

            # 构建 API 消息
            msg_dict = {"role": "assistant", "content": choice.content or None}
            if has_tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.tool_calls
                ]

            if has_tool_calls:
                messages.append(msg_dict)

                # 并发执行工具
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {}
                    for tc in choice.tool_calls:
                        func_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        icon = tool_icons.get(func_name, "🔧")
                        args_str = "、".join(f"{k}={v}" for k, v in args.items())
                        assistant_msg["content"] += (
                            f"{icon} **Step {step + 1} → {func_name}**\n"
                            f"> 参数: `{args_str}`\n"
                        )
                        yield history, "", messages

                        future = pool.submit(execute_tool, func_name, args)
                        futures[future] = (tc, func_name, args)

                    for future in concurrent.futures.as_completed(futures):
                        tc, func_name, args = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = f"执行异常: {e}"

                        result_short = result[:200] + "..." if len(result) > 200 else result
                        assistant_msg["content"] += f"> 📊 结果: {result_short}\n\n"
                        yield history, "", messages

                        # 工具结果送回 LLM
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                        # 记录 trace
                        if self.tracing:
                            self.trace.append({
                                "step": step + 1,
                                "tool": func_name,
                                "args": args,
                                "result": result[:500],
                            })

                yield history, "", messages

            else:
                # 无工具调用 → 完成
                messages.append(msg_dict)
                # 清理可能的 XML 残留
                if not has_content:
                    assistant_msg["content"] = choice.content or "（无回复）"
                content = assistant_msg["content"]
                content = re.sub(r"<tool_calls>.*?</tool_calls>", "", content, flags=re.DOTALL)
                content = re.sub(r"<invoke[^>]*>.*?</invoke>", "", content, flags=re.DOTALL)
                assistant_msg["content"] = content.strip()
                yield history, "", messages
                return

        # 超迭代次数
        assistant_msg["content"] += "\n\n⚠️ 已达最大推理步数，请尝试简化需求。"
        yield history, "", messages

    # ═══ 评测追踪 ═══

    def enable_tracing(self):
        self.tracing = True
        self.trace = []

    def disable_tracing(self):
        self.tracing = False

    def get_trace(self) -> list:
        return self.trace

    def clear_trace(self):
        self.trace = []


# ═══ 便捷函数（供 Gradio 使用） ═══

def create_agent(client: OpenAI, model: str = "deepseek-chat") -> TravelAgent:
    """创建 Agent 实例"""
    return TravelAgent(client, model)
