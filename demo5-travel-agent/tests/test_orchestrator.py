"""
测试 ReAct 推理循环（Mock API 调用）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import Mock, patch, MagicMock
from agent.orchestrator import TravelAgent


class TestTravelAgent:
    """TravelAgent 核心逻辑测试（不调真实 API）"""

    @pytest.fixture
    def agent(self):
        """创建 Mock 客户端"""
        mock_client = MagicMock()
        return TravelAgent(mock_client, model="test-model")

    def test_init(self, agent):
        """初始化测试"""
        assert agent.model == "test-model"
        assert agent.tracing is False
        assert agent.trace == []

    def test_enable_tracing(self, agent):
        """追踪开关测试"""
        agent.enable_tracing()
        assert agent.tracing is True
        agent.disable_tracing()
        assert agent.tracing is False

    def test_clear_trace(self, agent):
        """清空追踪测试"""
        agent.trace = [{"step": 1}]
        agent.clear_trace()
        assert agent.trace == []

    def test_chat_with_tool_call(self, agent):
        """模拟 LLM 返回 tool_call 的情况"""
        # Mock LLM 返回含 tool_calls 的响应
        mock_choice = MagicMock()
        mock_choice.message.content = "我来查航班"
        mock_choice.message.tool_calls = [
            MagicMock(
                id="call_test123",
                function=MagicMock(
                    name="search_flights",
                    arguments='{"from": "北京", "to": "上海", "date": "2026-07-09"}',
                ),
            )
        ]

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        agent.client.chat.completions.create.return_value = mock_resp

        # 执行
        gen = agent.chat(
            message="帮我查航班",
            history=[],
            messages_state=None,
            system_prompt="你是南航管家",
            max_iter=1,
        )

        results = list(gen)
        assert len(results) > 0

    def test_chat_direct_answer(self, agent):
        """模拟 LLM 直接回答（无 tool_calls）"""
        mock_choice = MagicMock()
        mock_choice.message.content = "好的，已为您查询"
        mock_choice.message.tool_calls = None

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        agent.client.chat.completions.create.return_value = mock_resp

        gen = agent.chat(
            message="你好",
            history=[],
            messages_state=None,
            system_prompt="你是南航管家",
        )

        results = list(gen)
        assert len(results) > 0
        # 最终输出应该包含回复内容
        final_history = results[-1][0]
        assert len(final_history) >= 2  # user + assistant

    def test_max_iter_limit(self, agent):
        """模拟 LLM 一直返回 tool_calls 达到最大迭代限制"""
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [
            MagicMock(id="call_x", function=MagicMock(name="search_flights", arguments='{}'))
        ]

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        agent.client.chat.completions.create.return_value = mock_resp

        gen = agent.chat(
            message="测试循环",
            history=[],
            messages_state=None,
            system_prompt="你是南航管家",
            max_iter=2,
        )

        results = list(gen)
        # 应该触发超迭代提示
        final_history = results[-1][0]
        last_content = final_history[-1]["content"]
        assert "最大推理步数" in last_content

    def test_chat_with_exception(self, agent):
        """模拟 API 异常"""
        agent.client.chat.completions.create.side_effect = Exception("API 超时")

        gen = agent.chat(
            message="测试异常",
            history=[],
            messages_state=None,
            system_prompt="测试",
        )

        results = list(gen)
        assert len(results) > 0
