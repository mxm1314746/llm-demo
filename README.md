# 缪训铭 · AI 应用开发作品集

> 景德镇陶瓷大学 · 人工智能专业 · 2027届  
> 求职方向: **AI应用开发 / 大模型应用 / AI Agent 实习生**

一套循序渐进的大模型应用实战项目：从对话 → RAG → Agent → Multi-Agent → 垂直领域 Agent，
每个项目均可独立运行，统一 UI 主题（`ui_theme.py`）。

---

## 📂 项目列表

| # | 项目 | 技术栈 | 一句话 | 端口 |
|---|------|--------|--------|:--:|
| 1 | [AI 对话助手](demo1-chatbot/) | DeepSeek + Gradio | 流式输出、参数调节、智能追问推荐 | 7860 |
| 2 | [记忆增强 RAG 智能体](demo2-rag/) | LangChain + ChromaDB + Function Calling | 文档RAG + **四层记忆(L0-L3)** + Agent 每轮主动召回 | 7861 |
| 3 | [AI Agent 智能助手](demo3-agent/) | Function Calling + ReAct | 手写 ReAct 循环，工具并发调用 + 联网搜索 | 7862 |
| 4 | [Multi-Agent 协作平台](demo4-multi-agent/) | Orchestrator + 共享记忆 | 数据/分析/报告多 Agent 协作完成量化研究 | 7863 |
| 5 | [南航智慧出行管家](demo5-travel-agent/) | ReAct + 10 工具 + 评测框架 | 航旅垂直 Agent，实时日期解析 + 订单状态机 + 三维度评测 | 7864 |

---

## ✨ 亮点：Demo2 四层记忆增强 RAG

在经典「上传文档 → 检索 → 回答」之上，加入了一套**可落地的分层长期记忆**，并用 Agent 工具调用打通：

```
L0 会话    原始 user/assistant 消息，按 session 落盘
  │  每 N 轮（或会话超时）聚合
L1 Episode  LLM 生成 summary + tags
  │  按 tag 相似度(Jaccard)归并，每种 tag 保留一个
L2 Structure 滚动合并摘要，记 merge_count
  │  合并达阈值(防抖)才蒸馏，避免噪声升级
L3 Skill   → 人工审批通过 → 写入 skills 向量库，供检索增强
```

- **Agent 工具调用**：`recall_memory`（每轮必先调用）/ `search_knowledge` / `store_memory`（主动记忆），System Prompt 强约束每轮召回。
- **防抖 + 人工审批**：L3 skill 需 structure 合并达 N 次才生成，并进入待审批队列，批准后才入向量库。
- **数据范围**：`scope = private / team`，recall 按范围过滤。
- **双向量库**：`knowledge`（文档）与 `skills`（长期记忆）隔离。
- 目录：`rag_store.py` · `memory/{schema,store,pipeline,manager}.py` · `agent/core.py` · `tests/test_memory.py`

## ✨ 亮点：Demo5 南航智慧出行管家

- **实时日期解析**：以真实当前时间为基准解析「今天/明天/后天/周X/M月D日/YYYY-MM-DD」，航班按请求日期动态生成（时刻稳定、余票价格确定性变化），过期日期拒绝。
- **订单状态机 + 持久化**：已出票 → 已值机/已选座 → 已改签/已退票，落盘 `orders.json`，新增 `query_order` 工具；退票费按距起飞真实时间计算，值机校验时间窗。
- **评测框架**：三维度 × 三层级指标、50 条用例、Bad Case 四类归因。

---

## 🚀 快速体验

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 2. 安装依赖（各项目目录下均有 requirements.txt）
pip install -r demo2-rag/requirements.txt

# 3. 启动任意项目
cd demo1-chatbot && python app.py        # → http://localhost:7860
cd demo2-rag && python app.py            # → http://localhost:7861
cd demo3-agent && python app.py          # → http://localhost:7862
cd demo4-multi-agent && python app.py    # → http://localhost:7863
cd demo5-travel-agent && python app.py   # → http://localhost:7864
```

> 需要 Python 3.10+。首次运行 demo2 会自动下载嵌入模型 `BAAI/bge-small-zh-v1.5`（已配置国内 HF 镜像）。

## 🧪 测试

```bash
cd demo2-rag && python tests/test_memory.py       # 记忆管线全链路（假 LLM，不耗 API）
cd demo5-travel-agent && python -m pytest tests/   # 出行管家 64 项单测
```

## 📧 联系我

- 邮箱: 2602032749@qq.com
- GitHub: [@mxm1314746](https://github.com/mxm1314746)
