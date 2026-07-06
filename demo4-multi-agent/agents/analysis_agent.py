"""
数据分析 Agent — 纯 LLM 推理，不调工具
"""

from core.agent_base import BaseAgent
from core.protocol import Task


class AnalysisAgent(BaseAgent):
    """分析 Agent — 数据分析、推理、计算"""

    def __init__(self, client, memory):
        super().__init__(
            name="数据分析师",
            description="分析数据、计算量化指标、推理得出结论",
            client=client,
            memory=memory,
        )
        self.system_prompt = """你是资深数据分析师。请做以下工作:

1. 仔细阅读输入数据
2. 提取所有可用的数值、事实
3. 进行计算和分析（直接在思考中计算，不必调用工具）
4. 得出有数据支撑的结论和洞察
5. 输出结构化的分析结果

输出格式:
## 数据摘要
- 列出关键数据点

## 量化分析
- 具体计算过程和结果
- 百分比、排名、增长率等

## 趋势洞察
- 从数据中推导的趋势
- 有数据支撑的判断

## 结论
- 最重要的 3 个发现

如果数据不足，明确说明「当前数据不完整，以下分析基于有限数据」。"""

    def execute(self, task: Task) -> str:
        self.update_status(self.status.WORKING, f"分析: {task.name}")

        # 从记忆获取上下文
        context = ""
        if self.memory:
            context = self.memory.get_recent_context(3)
            related = self.memory.search_long_term(task.description)
            if related:
                context += "\n" + "\n".join([r["content"][:200] for r in related[:2]])

        # 构建输入
        input_text = ""
        if task.input:
            if isinstance(task.input, dict):
                for k, v in task.input.items():
                    input_text += f"\n### {k}的输出\n{v}\n"
            else:
                input_text = str(task.input)

        prompt = f"""分析任务: {task.description}

原始数据:
{input_text}

{("参考历史: " + context) if context else ""}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = self.llm_chat(messages)  # 不传 tools

        if self.memory:
            self.memory.add_experience(self.name, task.description, result)

        self.update_status(self.status.DONE, "分析完成")
        return result
