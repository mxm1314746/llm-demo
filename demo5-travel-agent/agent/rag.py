"""
航空知识库 RAG — 基于 JSON 文件的关键词匹配
不依赖 ChromaDB，避免 embedding 模型下载问题
"""
import json
import os
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class PolicyRAG:
    """航空政策检索器 — 轻量级，不依赖 ChromaDB"""

    def __init__(self):
        self.policies = self._load_policies()
        self._category_keywords = {
            "行李": ["行李", "托运", "携带", "液体", "充电宝", "宠物", "运动器材", "物品"],
            "退改签": ["退票", "改签", "退改", "手续费", "延误", "退款", "转让"],
            "值机": ["值机", "选座", "登机", "座位", "错过"],
            "证件": ["身份证", "证件", "护照", "丢失"],
            "特殊旅客": ["老人", "儿童", "孕妇", "轮椅", "药品", "婴儿", "无人陪伴"],
            "会员": ["会员", "明珠", "积分", "里程"],
            "航班": ["航班", "动态", "舱位", "头等舱", "中转", "查询"],
        }

    def _load_policies(self) -> list:
        filepath = os.path.join(DATA_DIR, "policies.json")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)["policies"]

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """
        关键词 + 分类匹配检索
        返回最相关的 top_k 条政策
        """
        scored = []
        for p in self.policies:
            score = 0.0

            # 1. 分类关键词匹配 + 加权
            for cat, keywords in self._category_keywords.items():
                for kw in keywords:
                    if kw in question and cat == p["category"]:
                        score += 0.5
                    if kw in question and kw in p["question"]:
                        score += 0.8
                    if kw in question and kw in p["answer"]:
                        score += 0.4

            # 2. 字符级 Jaccard 相似度
            q_set = set(question)
            a_all = set(p["question"] + p["answer"])
            if q_set:
                jaccard = len(q_set & a_all) / len(q_set | a_all)
                score += jaccard * 2.0

            # 3. 公共子串奖励
            for i in range(len(question) - 1):
                sub = question[i:i + 2]
                if sub in p["question"] or sub in p["answer"]:
                    score += 0.1

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def query(self, question: str) -> Optional[str]:
        """查询政策并返回格式化结果"""
        results = self.search(question, top_k=3)

        if not results:
            return None

        output = []
        for i, p in enumerate(results, 1):
            output.append(f"📋 [{p['category']}] {p['question']}\n{p['answer']}")

        return "\n\n".join(output)


# 全局单例
_rag_instance: Optional[PolicyRAG] = None


def get_rag() -> PolicyRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PolicyRAG()
    return _rag_instance


def search_policy(question: str) -> str:
    """对外接口 — 搜索航空政策"""
    rag = get_rag()
    result = rag.query(question)
    if result:
        return result
    return "未找到相关政策信息。建议联系南航客服热线 95539 获取帮助。"
