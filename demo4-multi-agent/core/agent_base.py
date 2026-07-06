"""
Agent 基类
所有专业 Agent 继承此类，实现自己的 execute() 方法
"""

import json
import re
import time
from typing import Optional
from openai import OpenAI
from .protocol import Message, MessageType, AgentStatus, Task


class BaseAgent:
    """Agent 基类"""

    def __init__(self, name: str, description: str, client: OpenAI, memory=None):
        self.name = name
        self.description = description
        self.client = client
        self.memory = memory           # 共享记忆系统
        self.status = AgentStatus.IDLE
        self.system_prompt = self._default_system_prompt()
        self.tools = []                # 子类注册自己的工具
        self.status_history = []       # 状态记录（用于UI展示）

    def _default_system_prompt(self) -> str:
        return f"你是 {self.name}，负责 {self.description}。"

    def update_status(self, status: AgentStatus, info: str = ""):
        """更新状态并记录"""
        self.status = status
        self.status_history.append({
            "time": time.strftime("%H:%M:%S"),
            "status": status.value,
            "info": info,
            "agent": self.name,
        })

    def execute(self, task: Task) -> str:
        """
        执行任务——子类必须实现此方法
        返回: 任务结果文本
        """
        raise NotImplementedError("子类必须实现 execute()")

    def llm_chat(self, messages: list, tools: list = None, temperature: float = 0.3) -> str:
        """调用 LLM 的通用方法"""
        try:
            kwargs = dict(model="deepseek-chat", messages=messages, temperature=temperature)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            resp = self.client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            # 如果有工具调用，解析并执行
            if msg.tool_calls:
                return self._handle_tool_calls(messages, msg)

            return msg.content or ""

        except Exception as e:
            return f"LLM 调用出错: {e}"

    def _handle_tool_calls(self, messages: list, response_msg) -> str:
        """处理 LLM 返回的工具调用，串联多轮直到最终文本"""
        # 最多重试 3 轮工具调用
        for _ in range(3):
            # 记录 assistant 消息
            assistant_msg = {
                "role": "assistant",
                "content": response_msg.content or None,
            }
            if response_msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in response_msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not response_msg.tool_calls:
                # 纯文本回复，清理可能混入的 <tool_calls> 标记
                content = response_msg.content or ""
                content = re.sub(r"<tool_calls>.*?</tool_calls>", "", content, flags=re.DOTALL)
                content = re.sub(r"<invoke[^>]*>.*?</invoke>", "", content, flags=re.DOTALL)
                return content.strip() or "（无内容）"

            # 执行每个工具
            for tc in response_msg.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = self._execute_tool(func_name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

            # 继续 LLM 推理
            resp2 = self.client.chat.completions.create(
                model="deepseek-chat", messages=messages, temperature=0.3,
            )
            response_msg = resp2.choices[0].message

        # 超过最大轮次，返回最终内容
        content = response_msg.content or "（工具调用超限）"
        content = re.sub(r"<tool_calls>.*?</tool_calls>", "", content, flags=re.DOTALL)
        content = re.sub(r"<invoke[^>]*>.*?</invoke>", "", content, flags=re.DOTALL)
        return content.strip()

    def _execute_tool(self, name: str, args: dict) -> str:
        """执行工具——子类重写以提供自己的工具"""
        return f"未知工具: {name}"

    def receive_message(self, msg: Message) -> Message:
        """接收并处理消息"""
        if msg.msg_type == MessageType.TASK:
            task = Task(**msg.content) if isinstance(msg.content, dict) else msg.content
            result = self.execute(task)
            return Message(
                msg_type=MessageType.RESULT,
                sender=self.name,
                receiver=msg.sender,
                task_id=task.task_id,
                content={"result": result},
            )
        return Message(
            msg_type=MessageType.ERROR,
            sender=self.name,
            receiver=msg.sender,
            content={"error": "不支持的消息类型"},
        )
