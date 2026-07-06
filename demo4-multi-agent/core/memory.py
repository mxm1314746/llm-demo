"""
共享记忆系统
短期记忆（会话级）+ 长期记忆（JSON 文件持久化）
不依赖 ChromaDB —— 零额外下载，开箱即用
"""

import os
import json
import datetime


class SharedMemory:
    """
    所有 Agent 共享的记忆系统

    两级架构:
    1. short_term: 当前会话的记忆（dict）
    2. long_term: 跨会话持久化记忆（JSON 文件）
    """

    def __init__(self, persist_dir: str = None):
        self.short_term = {}         # {agent: {key: value}}
        self.conversation = []       # 当前会话记录
        self.agent_experience = {}   # {agent_name: [经验列表]}
        self.persist_dir = persist_dir

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    # ─── 短期记忆 ───

    def remember(self, agent: str, key: str, value: str):
        """Agent 记住一条信息（短期）"""
        if agent not in self.short_term:
            self.short_term[agent] = {}
        self.short_term[agent][key] = value

    def recall(self, agent: str, key: str) -> str:
        """Agent 回忆一条信息（短期）"""
        if agent in self.short_term and key in self.short_term[agent]:
            return self.short_term[agent][key]
        return ""

    def add_conversation(self, role: str, content: str, agent: str = "user"):
        """添加一条对话记录"""
        self.conversation.append({
            "role": role,
            "content": content,
            "agent": agent,
            "time": datetime.datetime.now().isoformat(),
        })

    def get_recent_context(self, n: int = 5) -> str:
        """获取最近 n 条对话记录，用于 LLM 上下文"""
        recent = self.conversation[-n:] if len(self.conversation) > n else self.conversation
        return "\n".join([f"[{c['agent']}] {c['content'][:100]}" for c in recent])

    # ─── 长期记忆 ───

    def save_long_term(self, agent: str, key: str, content: str):
        """保存到长期记忆（JSON 文件）"""
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, "long_term_memory.json")
        try:
            data = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data[f"{agent}__{key}"] = {
                "agent": agent, "key": key, "content": content,
                "time": datetime.datetime.now().isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def search_long_term(self, query: str, n: int = 3) -> list:
        """从长期记忆中检索（关键词匹配）"""
        results = []
        if not self.persist_dir:
            return results
        path = os.path.join(self.persist_dir, "long_term_memory.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    content = v.get("content", "")
                    # 简单关键词匹配
                    if any(word in content.lower() for word in query.lower().split()):
                        results.append({"content": content[:300], "metadata": v})
                        if len(results) >= n:
                            break
            except Exception:
                pass
        return results

    def add_experience(self, agent: str, task: str, result: str):
        """Agent 记录一次经验"""
        if agent not in self.agent_experience:
            self.agent_experience[agent] = []
        exp = {"task": task, "result": result[:200], "time": datetime.datetime.now().isoformat()}
        self.agent_experience[agent].append(exp)
        # 同时保存到长期记忆
        idx = len(self.agent_experience[agent])
        self.save_long_term(agent, f"experience_{idx}", f"任务: {task}\n结果: {result[:500]}")

    def get_status_summary(self) -> str:
        """获取记忆系统状态摘要（供UI展示）"""
        lines = [
            "短期记忆: {} 条".format(sum(len(v) for v in self.short_term.values())),
            "对话记录: {} 条".format(len(self.conversation)),
            "Agent经验: {} 条".format(sum(len(v) for v in self.agent_experience.values())),
        ]
        if self.persist_dir:
            path = os.path.join(self.persist_dir, "long_term_memory.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        count = len(json.load(f))
                    lines.append("长期记忆: {} 条（JSON文件）".format(count))
                except Exception:
                    lines.append("长期记忆: 文件存储")
        return "\n".join(lines)
