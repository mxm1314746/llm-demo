"""
数据采集 Agent
搜狗搜索 + LLM 知识双重数据源
"""

import re
import time
import random
import requests
from core.agent_base import BaseAgent
from core.protocol import Task


class DataAgent(BaseAgent):
    """数据采集 Agent — 搜索+知识整理"""

    def __init__(self, client, memory):
        super().__init__(
            name="数据采集员",
            description="搜索网络信息、整理结构化数据",
            client=client,
            memory=memory,
        )
        self.system_prompt = """你是专业的信息研究员。请根据提供的「信息来源」整理出有价值的分析素材。

整理规则:
1. 提取所有事实、数字、趋势、案例
2. 按主题分组，结构化输出
3. 每条信息标注来源（搜索结果 或 模型知识）
4. 输出干净的中文，不要任何 XML/HTML 标记

输出格式:
## 来源概况
- 来源: 搜索结果 N 条 + 模型知识补充

## 核心事实
- [事实1]（来源: xxx）
- [事实2]

## 数据点
列出具体数字、百分比、年份等

## 关键趋势
列出观察到的趋势和方向"""

    def execute(self, task: Task) -> str:
        self.update_status(self.status.WORKING, f"信息采集: {task.name}")

        # 1. 提取搜索关键词
        queries = self._extract_queries(task.description)

        # 2. 搜狗搜索
        search_text = ""
        for i, q in enumerate(queries):
            if i > 0:
                time.sleep(random.uniform(1.0, 2.0))  # 避免限流
            results = self._sogou_search(q)
            if results:
                search_text += f"\n## 搜狗搜索结果（关键词: {q}）\n"
                search_text += self._format_results(results)
                self.update_status(self.status.WORKING, f"搜索 {q}: {len(results)}条")

        has_search = bool(search_text.strip())
        if not has_search:
            search_text = "## 信息来源\n（网络搜索不可用，以下均基于模型内置知识。已特别标注。）"

        # 3. LLM 整理结果 + 补充知识
        prompt = f"""研究任务: {task.description}

请根据以下信息来源整理研究成果:

{search_text}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = self.llm_chat(messages)  # 不传 tools

        # 4. 如果搜索成功，缓存结论
        if self.memory:
            self.memory.remember(self.name, task.name, result[:500])
            if has_search:
                self.memory.save_long_term(self.name, task.name, result[:1000])

        self.update_status(self.status.DONE, f"采集完成: {task.name}")
        return result

    # --- 关键词提取 ---

    def _extract_queries(self, task_desc: str) -> list:
        """用 LLM 提取搜索引擎友好的关键词"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是搜索专家。把任务描述转成 1-2 个高质量的搜索关键词。\n"
                        "规则: 自然连贯的中文短语；用行业术语；不要空格分隔。\n"
                        "只返回关键词，一行一个，不超过 2 行。"
                    )},
                    {"role": "user", "content": task_desc},
                ],
                temperature=0.3, max_tokens=80,
            )
            lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n")
                     if len(l.strip()) >= 6 and len(l.strip()) <= 60]
            return lines[:2] if lines else [task_desc[:40]]
        except Exception:
            return [task_desc[:40]]

    # --- 搜狗搜索（国内可用，无需VPN，质量好）---

    def _sogou_search(self, query: str, n: int = 5) -> list:
        """搜狗搜索（国内直连，加反反爬机制）"""
        ua_list = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
        ]
        try:
            headers = {
                "User-Agent": random.choice(ua_list),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            resp = requests.get(
                "https://www.sogou.com/web",
                params={"query": query},
                headers=headers, timeout=10,
            )
            resp.encoding = "utf-8"

            # 页面太短说明被反爬了
            if len(resp.text) < 10000:
                return []

            h3s = re.findall(r'<h3[^>]*>(.*?)</h3>', resp.text, re.DOTALL)
            results = []
            for h3 in h3s[:n]:
                t = re.sub(r"<[^>]+>", "", h3).replace("&nbsp;", " ").strip()
                t = re.sub(r"\s+", " ", t)
                if len(t) >= 8:
                    results.append({"title": t, "body": "", "href": ""})
            return results
        except Exception:
            return []

    def _format_results(self, results: list) -> str:
        """格式化搜索结果"""
        return "\n".join(
            f"- [{r['title']}]({r.get('href', '')})" for r in results
        ) if results else "（无结果）"
