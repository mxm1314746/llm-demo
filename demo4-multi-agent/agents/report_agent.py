"""
报告生成 Agent
负责将分析结果整理成结构化报告
"""

from core.agent_base import BaseAgent
from core.protocol import Task


class ReportAgent(BaseAgent):
    """报告 Agent — 整合信息、生成结构化报告"""

    def __init__(self, client, memory):
        super().__init__(
            name="报告撰写员",
            description="整合多源信息，生成结构化、可读性强的报告",
            client=client,
            memory=memory,
        )
        self.system_prompt = """你是报告撰写员，负责生成结构化报告。
请按照以下步骤工作：
1. 梳理所有输入信息
2. 按逻辑组织内容结构
3. 生成包含标题、段落、要点、数据支撑的完整报告
4. 报告格式要清晰易读，适当使用 Markdown

报告要求：
- 有明确的标题和分层结构
- 每个观点有数据或来源支撑
- 语言简洁专业
- 最后附上总结和结论"""

    def execute(self, task: Task) -> str:
        self.update_status(self.status.WORKING, f"开始撰写: {task.name}")

        prompt = f"报告主题: {task.description}\n\n"
        if task.input:
            if isinstance(task.input, dict):
                for k, v in task.input.items():
                    prompt += f"## {k}\n{v}\n\n"
            else:
                prompt += str(task.input)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = self.llm_chat(messages)
        self.update_status(self.status.DONE, "报告完成")
        return result
