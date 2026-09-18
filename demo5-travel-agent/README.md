# ✈️ 南航「智慧出行管家」AI Agent

基于 **DeepSeek + ReAct 推理循环** 的航空出行智能助手，模拟南方航空智慧出行管家的核心功能。

## 项目亮点

- **9 大航空工具**: 航班查询、订票、值机、改签、退票、选座、行李查询、机场信息、政策查询
- **自然语言交互**: "一句话交互"——用户用口语描述需求，Agent 自动完成多步操作
- **ReAct 推理循环**: 手写 Reasoning + Acting 循环，支持多步工具编排和并发执行
- **分层 System Prompt**: 角色层 → 工具Schema层 → 行为用例层 → 安全策略层
- **评测框架**: 三维度×三层级评测体系 + 50条测试用例 + 4类Bad Case归因
- **面试友好**: 代码结构清晰，关键逻辑带注释，核心概念和面试题对应

## 快速启动

```bash
cd demo5-travel-agent
pip install -r requirements.txt
python app.py
# 打开 http://localhost:7864
```

## 项目结构

```
demo5-travel-agent/
├── app.py                  # Gradio 双Tab界面 (7864端口)
├── agent/
│   ├── tools.py            # 9个航空工具 Schema + 模拟实现
│   ├── orchestrator.py     # ReAct 推理循环核心
│   ├── prompt.py           # 分层 System Prompt
│   └── rag.py              # 航空知识库检索
├── eval/
│   ├── test_cases.py       # 50条测试用例
│   ├── runner.py           # 批量跑批器
│   ├── metrics.py          # 三维度×三层级评分
│   ├── attribution.py      # Bad Case 4类归因
│   └── report.py           # Markdown 报告生成
├── data/
│   ├── flights.json        # 20条模拟航班
│   ├── airports.json       # 5个机场信息
│   ├── policies.json       # 25条航空政策FAQ
│   └── test_cases/         # 测试用例JSON
└── requirements.txt
```

## 面试话术要点

**Q: 这个项目的核心架构是什么？**
> 基于 ReAct 推理循环的 AI Agent，底层调用 DeepSeek API 的 Function Calling 能力。
> Agent 接收用户自然语言输入后，LLM 自主判断需要调用哪些工具，
> 我实现了并发工具执行引擎（ThreadPoolExecutor），支持多工具并行调用。
> 核心循环最大 5 轮迭代，每轮 LLM 根据工具返回结果继续推理直到给出最终答案。

**Q: 评测体系是怎么设计的？**
> 三维度 × 三层级：结果正确性（意图准确率/槽位填充率/工具成功率）、
> 过程合规性（步骤顺序/参数准确性/降级规范）、
> 成本效率（p50/p99延迟/Token消耗/工具调用次数）。
> 测试用例 50 条，按 60% 常规 + 25% 边缘 + 15% 对抗分布。

**Q: Bad Case 怎么归因优化的？**
> 四类失败模式：工具选错、参数缺失、链路断裂、安全风险。
> 跑完批量测试后自动归类，按影响面排序给出优化建议。
> 优化后重新跑回归集，对比通过率和延迟变化。
