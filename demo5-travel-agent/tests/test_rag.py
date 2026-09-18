"""
测试 RAG 知识库检索
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.rag import PolicyRAG


class TestPolicyRAG:
    """PolicyRAG 知识库检索测试"""

    @pytest.fixture
    def rag(self):
        return PolicyRAG()

    def test_exact_match(self, rag):
        """精确匹配"""
        results = rag.search("经济舱免费行李额是多少")
        assert len(results) >= 1
        assert results[0]["category"] == "行李"

    def test_semantic_match(self, rag):
        """语义匹配（关键词有重叠）"""
        results = rag.search("退票手续费怎么算")
        assert len(results) >= 1
        assert results[0]["category"] == "退改签"

    def test_pet_query(self, rag):
        results = rag.search("宠物可以带上飞机吗")
        assert len(results) >= 1
        assert "宠物" in results[0]["answer"]

    def test_id_lost_query(self, rag):
        results = rag.search("身份证丢了")
        assert len(results) >= 1
        assert results[0]["category"] == "证件"

    def test_query_with_price(self, rag):
        results = rag.search("改签费用多少")
        assert len(results) >= 1

    def test_no_match(self, rag):
        """不相关查询"""
        results = rag.search("今天天气怎么样")
        # 返回空列表或最弱匹配
        assert isinstance(results, list)

    def test_query_method(self, rag):
        """query() 接口返回格式化文本"""
        result = rag.query("退票手续费怎么算")
        assert result is not None
        assert "手续费" in result
