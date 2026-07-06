"""
Agent 间通信协议
定义消息格式、状态码、任务类型等
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any
import json
import uuid


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"           # 空闲
    WORKING = "working"     # 工作中
    DONE = "done"           # 完成
    FAILED = "failed"       # 失败


class MessageType(Enum):
    """消息类型"""
    TASK = "task"               # 分配任务
    RESULT = "result"           # 返回结果
    QUERY = "query"             # 查询信息
    RESPONSE = "response"       # 回复查询
    ERROR = "error"             # 错误
    STATUS = "status"           # 状态更新


@dataclass
class Message:
    """Agent 间通信消息"""
    msg_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    msg_type: MessageType = MessageType.TASK
    sender: str = ""
    receiver: str = ""
    task_id: str = ""
    content: Any = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "task_id": self.task_id,
            "content": self.content,
            "metadata": self.metadata,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Task:
    """任务定义"""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    agent: str = ""              # 分配给哪个 Agent
    status: str = "pending"      # pending / running / done / failed
    input: Any = None
    output: Any = None
    dependencies: list = field(default_factory=list)  # 依赖的任务ID列表
    subtasks: list = field(default_factory=list)      # 子任务列表
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "agent": self.agent,
            "status": self.status,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }
