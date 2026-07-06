"""
Orchestrator（编排者）
负责: 理解用户请求 → 拆解任务 → 分配给专业 Agent → 汇总结果
"""

import json
import time
from openai import OpenAI
from .protocol import Task, Message, MessageType
from .memory import SharedMemory


class Orchestrator:
    """
    Multi-Agent 系统的核心编排者

    工作流程:
    1. 接收用户请求
    2. 用 LLM 拆解成子任务
    3. 分配给对应的专业 Agent
    4. 监控执行状态
    5. 汇总最终结果
    """

    def __init__(self, client: OpenAI, memory: SharedMemory):
        self.client = client
        self.memory = memory
        self.agents = {}           # {agent_name: agent_instance}
        self.tasks = []            # 当前任务列表
        self.execution_log = []    # 执行日志（用于UI展示）

    def register_agent(self, agent):
        """注册一个专业 Agent"""
        self.agents[agent.name] = agent
        print(f"  [OK] Agent: {agent.name}")

    def log(self, message: str, level: str = "info"):
        """记录执行日志"""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "message": message,
            "level": level,
        }
        self.execution_log.append(entry)
        print(f"  [{level.upper()}] {message}")

    # ─── 核心：任务拆解 ───

    def decompose_task(self, user_request: str) -> list:
        """
        用 LLM 将用户请求拆解为子任务列表

        返回: [{"name": "任务名", "description": "任务描述", "agent": "目标Agent"}, ...]
        """
        self.log("正在分析任务并拆解子任务...")

        prompt = f"""你是一个 Multi-Agent 系统的任务规划器。
请将以下用户请求拆解为多个可并行/串行执行的子任务。

可用 Agent:
{chr(10).join([f'- {name}: {agent.description}' for name, agent in self.agents.items()])}

用户请求: {user_request}

请以 JSON 数组格式返回，每个任务包含:
{{
    "name": "简短任务名",
    "description": "详细任务描述（告诉 Agent 要做什么）",
    "agent": "目标 Agent 名称（从上面选择）"
}}

注意：
- 串行任务（依赖前一个结果的）放在后面
- 可以并行执行的放在前面
- 一般先收集数据，再分析，最后生成报告"""

        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的任务规划助手。只返回 JSON，不要其他文字。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            text = resp.choices[0].message.content.strip()
            # 提取 JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            tasks = json.loads(text)
            if not isinstance(tasks, list):
                raise ValueError("不是数组")

            self.log(f"任务拆解完成: {len(tasks)} 个子任务")
            return tasks

        except Exception as e:
            self.log(f"任务拆解失败: {e}", "error")
            # 降级方案：默认拆解为三个步骤
            return [
                {"name": f"搜索相关信息", "description": f"搜索与「{user_request}」相关的信息", "agent": "数据采集员"},
                {"name": "分析信息", "description": f"分析搜索结果: {user_request}", "agent": "数据分析师"},
                {"name": "生成报告", "description": f"生成关于「{user_request}」的完整报告", "agent": "报告撰写员"},
            ]

    # ─── 核心：执行工作流 ───

    def execute(self, user_request: str) -> dict:
        """
        执行完整的 Multi-Agent 工作流

        返回:
        {
            "final_report": "最终报告",
            "intermediate_results": {"Agent名": "结果", ...},
            "execution_log": [...],
            "agent_statuses": {...},
        }
        """
        self.execution_log = []
        self.tasks = []

        # 1. 记录用户请求
        self.memory.add_conversation("user", user_request, "用户")
        self.log(f"收到用户请求: {user_request[:100]}...")

        # 2. 拆解任务
        task_defs = self.decompose_task(user_request)

        # 3. 创建任务对象
        previous_results = {}
        intermediate_results = {}

        for i, td in enumerate(task_defs):
            task = Task(
                name=td.get("name", f"任务{i+1}"),
                description=td.get("description", ""),
                agent=td.get("agent", ""),
                dependencies=[t.task_id for t in self.tasks] if td.get("depends_on") else [],
            )
            self.tasks.append(task)

        self.log(f"开始执行 {len(self.tasks)} 个子任务...")

        # 4. 按顺序执行任务（串行，因为多数任务有依赖关系）
        for i, task in enumerate(self.tasks):
            agent = self.agents.get(task.agent)
            if not agent:
                self.log(f"未找到 Agent: {task.agent}，跳过", "error")
                continue

            # 传递前置结果
            if previous_results:
                task.input = previous_results

            self.log(f"  → 分配给 [{agent.name}]: {task.name}")
            agent.update_status(agent.status.WORKING, task.name)

            try:
                # Agent 执行任务
                result = agent.execute(task)
                task.status = "done"
                task.output = result
                intermediate_results[agent.name] = result

                # 记录到共享记忆
                self.memory.add_conversation("assistant", f"[{agent.name}] {result[:100]}...", agent.name)
                self.memory.remember(agent.name, f"task_{i}", result[:500])

                # 传递给下一个任务
                previous_results[agent.name] = result
                self.log(f"  ✅ [{agent.name}] 完成 → {len(result)} 字符")

            except Exception as e:
                task.status = "failed"
                self.log(f"  ❌ [{agent.name}] 失败: {e}", "error")
                intermediate_results[agent.name] = f"[执行失败] {e}"

        # 5. 汇总最终结果
        self.log("所有任务完成，生成最终汇总...")

        # 如果有报告 Agent 的结果，作为最终报告
        final_report = intermediate_results.get("报告撰写员", "")
        if not final_report:
            final_report = self._summarize_results(user_request, intermediate_results)

        self.memory.add_conversation("assistant", final_report[:200], "系统")
        self.log("✅ 全部完成！")

        return {
            "final_report": final_report,
            "intermediate_results": intermediate_results,
            "execution_log": self.execution_log,
            "task_breakdown": [t.to_dict() for t in self.tasks],
        }

    def _summarize_results(self, user_request: str, results: dict) -> str:
        """当没有报告 Agent 时，自己汇总结果"""
        try:
            parts = [f"# {user_request}\n"]
            for agent_name, result in results.items():
                parts.append(f"## {agent_name}\n{result[:500]}\n")

            prompt = f"请根据以下信息生成一份完整的报告:\n\n{''.join(parts)}"
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content or "（无结果）"
        except Exception as e:
            return f"汇总失败: {e}"
