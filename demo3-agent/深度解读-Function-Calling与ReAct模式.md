# 🧠 AI Agent 项目深度解读

## Function Calling 原理 + ReAct 模式 + 面试攻略

---

# 第一部分：Function Calling 是什么？

## 1.1 传统 LLM 的局限

普通 LLM 只能 **"说"** 不能 **"做"**。你问它：

> "今天北京的天气怎么样？"

它可能回答：
> "抱歉，我没有实时获取天气的能力，请您自己查一下。"

**原因**：LLM 的知识截止于训练数据，无法访问实时信息、无法执行计算、无法操作外部系统。

## 1.2 Function Calling 的解决方案

Function Calling（函数调用）让 LLM **能调用外部工具**：

```
用户提问 → LLM 判断需要工具 → 返回工具名+参数 → 代码执行工具 → 结果送回 LLM → LLM 给出最终回答
```

**不是 LLM 自己去执行代码**，而是 LLM **说**："我需要调用 calculator，参数是 123 * 456"
→ **你的代码**去执行计算 → 把结果还给 LLM → LLM 基于结果组织回答。

## 1.3 核心概念：工具描述（Tool Schema）

要让 LLM 知道有什么工具可用，需要用 **JSON Schema** 描述每个工具：

```json
{
    "type": "function",
    "function": {
        "name": "calculator",              // 工具名（LLM 通过这个名字调用）
        "description": "计算数学表达式",    // 描述（LLM 靠这个理解工具用途）
        "parameters": {                     // 参数定义（LLM 知道要传什么参数）
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"  // 每个参数的描述也很重要
                }
            },
            "required": ["expression"]
        }
    }
}
```

**关键**：
- `name` 和 `description` 越清晰，LLM 越能正确选择工具
- `parameters` 描述越详细，LLM 传参越准确
- LLM **不会真正执行函数**，它只负责决定"要调用哪个函数、传什么参数"

---

# 第二部分：ReAct 模式详解

## 2.1 什么是 ReAct？

**ReAct = Reasoning + Acting**（推理 + 行动）

这是 Google 2023 年提出的 Agent 框架，核心思想：

```
思考(Thought) → 行动(Action) → 观察(Observation) → 思考 → 行动 → 观察 → ... → 最终答案
```

## 2.2 ReAct 循环图解

```
                    ┌─────────────────────────────────────┐
                    │             用户提问                  │
                    │   "计算 2^10 然后搜索 AI 新闻"        │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Step 1: LLM 思考                │
                    │  "用户要求两件事，先计算再搜索"        │
                    │  返回: tool_calls=[calculator, search]│
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Step 2: 执行工具                │
                    │  calculator("2^10") → "1024"        │
                    │  web_search("AI 2026") → [结果...]  │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Step 3: LLM 继续思考            │
                    │  "计算结果是1024，搜索到了AI新闻..."  │
                    │  返回: content="2^10=1024, AI新闻..."│
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      最终答案                        │
                    │  "2^10 = 1024。关于AI的最新进展..."  │
                    └─────────────────────────────────────┘
```

## 2.3 项目中 ReAct 的代码实现

在 `agent_chat()` 函数中，这个循环就是 ReAct：

```python
for step in range(MAX_ITER):           # 循环：允许 LLM 多次思考+行动
    # 1. 调用 LLM（传入工具定义）
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,        # ← 告诉 LLM 有什么工具
        tool_choice="auto",             # ← 让 LLM 自主决定是否调工具
    )
    choice = resp.choices[0].message

    if choice.tool_calls:              # → 2. LLM 决定调用工具
        for tc in choice.tool_calls:
            result = execute_tool(...)  # → 3. 执行工具
            messages.append({           # → 4. 结果送回 LLM
                "role": "tool",
                "content": result
            })
        # → 继续下一轮循环（Step 2, 3, ...）
    else:
        # LLM 直接回答 → 完成
        return choice.content
```

---

# 第三部分：本项目架构逐层拆解

## 3.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    Gradio UI                         │
│         聊天界面 · 参数调节 · 推理过程展示             │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              agent_chat()  ReAct 循环                 │
│     LLM 调用 → 判断 → 工具执行 → 继续 → 回答         │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│             消息管理 (messages_state)                │
│     跨轮次保存 tool_calls 和 tool 结果               │
└──────┬─────────┬──────────┬─────────────────────────┘
       │         │          │
┌──────▼──┐ ┌───▼────┐ ┌───▼──────────┐
│ 计算器   │ │ 时间   │ │ 网络搜索      │
│ safe_eval│ │datetime│ │ DuckDuckGo   │
└─────────┘ └────────┘ └──────────────┘
```

## 3.2 核心模块说明

| 模块 | 文件位置 | 职责 |
|------|---------|------|
| **工具定义** | `TOOL_DEFINITIONS` | 用 JSON Schema 描述工具接口 |
| **工具执行** | `execute_tool()` | 真正执行工具逻辑 |
| **ReAct 循环** | `agent_chat()` | 编排 LLM 调用→工具执行→继续推理 |
| **消息管理** | `messages_state` | 跨轮次保存对话+工具调用历史 |
| **并发执行** | `ThreadPoolExecutor` | 同时执行多个工具调用 |
| **UI 展示** | Gradio Blocks | 显示推理过程链 |

## 3.3 相比 LangChain Agent 的差异（面试加分点）

```
本项目:                              LangChain Agent:
- 手写 ReAct 循环                   - 封装好的 AgentExecutor
- 完全控制每一步逻辑                 - 黑盒，难调试
- 工具结果直接展示在 UI              - 需要额外配置回调
- 消息历史精确控制                   - 自动管理，但不可控
- 代码量 ~300 行，完全可读           - 依赖重，源码复杂
```

**面试时可以说**：
> "我没有直接使用 LangChain Agent 的黑盒封装，而是手写了 ReAct 推理循环。
> 这样我对每一步的推理过程都有完全的控制，也更容易调试和扩展。"

---

# 第四部分：关键代码逐行解读

## 4.1 工具定义 —— LLM 和代码之间的"契约"

```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",       # 工具标识符
            "description": "计算...",    # 告诉 LLM 这个工具做什么
            "parameters": {...}          # 告诉 LLM 需要什么参数
        }
    },
    ...
]
```

**为什么需要 description？**
LLM 靠 description 理解工具的用途。description 写得好，LLM 才能正确选择工具。

## 4.2 工具执行 —— 真正干活的地方

```python
def execute_tool(name: str, args: dict) -> str:
    if name == "calculator":
        return safe_eval(args["expression"])
    elif name == "web_search":
        return search_web(args["query"])
    ...

# 每个工具返回字符串，这个字符串会被送回给 LLM
```

**关键设计**：`execute_tool` 的返回值一定是**字符串**，因为要放进 `messages` 里送回给 LLM。

## 4.3 ReAct 循环 —— Agent 的大脑

```python
for step in range(MAX_ITER):
    # 1. LLM 思考
    resp = client.chat.completions.create(..., tools=TOOL_DEFINITIONS)

    if resp.choices[0].message.tool_calls:
        # 2. LLM 决定：需要调用工具
        #    → 提取工具名和参数
        #    → 执行工具
        #    → 结果放回 messages
        #    → 继续循环（让 LLM 基于结果继续思考）
    else:
        # 3. LLM 决定：可以直接回答了
        #    → 返回内容，结束循环
```

## 4.4 多轮对话记忆 —— 让 Agent 记住上一步

```python
# ★ 面试高频考点：多轮对话中如何保持 Agent 的"记忆"

# messages_state 是一个 gr.State，跨轮次保存
# 里面存了所有历史 tool_calls 和 tool 结果

messages = [{"role": "system", "content": system_prompt}]
if messages_state:
    messages.extend(messages_state)  # ← 恢复历史工具调用记录
messages.append({"role": "user", "content": message})

# 每一轮结束后，把 messages 存回 messages_state
# 下一轮用户发新消息时，之前的工具调用历史还在
```

---

# 第五部分：面试可能问的问题 & 标准回答

## Q1: Function Calling 的原理是什么？

> **答**：Function Calling 不是 LLM 真的去执行函数，而是 LLM 在生成回复时，
> 通过工具描述（JSON Schema）知道有哪些工具可用。当 LLM 觉得需要外部信息或计算时，
> 它会返回一个特殊的 tool_calls 字段，包含**工具名和参数**。
> 真正的执行由开发者代码完成，执行结果再送回 LLM，让它基于结果继续推理。
>
> 可以理解为：LLM 是"指挥官"，决定用什么工具、传什么参数；
> 代码是"执行者"，真正干活；结果返回给"指挥官"做下一步决策。

## Q2: ReAct 和传统的 Chain-of-Thought 有什么区别？

> **答**：Chain-of-Thought 只是让 LLM 一步步思考，但思考完了还是只能"说"。
> ReAct 在 CoT 的基础上加了"行动"环节——LLM 不仅可以思考，还可以调用工具、
> 获取外部信息、执行计算，然后基于结果继续推理。
>
> 简单说：CoT 是"想完再说"，ReAct 是"想→做→观察→再想→再说"。

## Q3: 你的 Agent 和 LangChain Agent 比有什么优势？

> **答**：我没有直接使用 LangChain 的 AgentExecutor 黑盒，而是手写了 ReAct 循环。
> 这样有三个好处：
> 1. **完全可控**——每一步的 messages 结构、工具调用逻辑、错误处理都自己掌控
> 2. **透明可观测**——工具调用过程直接展示在 UI 上，用户可以看见 AI 的"思考链"
> 3. **轻量可扩展**——加一个新工具只需要写一个执行函数 + 一行工具定义，不用理解 LangChain 的复杂抽象

## Q4: 工具调用失败怎么办？你的错误处理策略是什么？

> **答**：两个层面的处理：
> 1. **工具执行层面**——每个工具都有 try-catch，出错返回错误描述而不是崩溃
> 2. **LLM 层面**——错误描述作为 tool 结果送回 LLM，LLM 可以判断：
>    - 是参数错了？→ 重新生成正确的参数再试
>    - 是工具不可用？→ 告诉用户换个方式
>    
> 比如计算器返回 "计算错误: 除以零"，LLM 会意识到问题并告诉用户。

## Q5: 你的项目有哪些可以改进的地方？

> **答**：（展示思考深度）
> 1. **流式输出**——目前是整个 ReAct 循环结束后才展示全部，可以改成流式逐步显示
> 2. **工具结果缓存**——同样的问题重复搜索时，可以缓存结果减少 API 调用
> 3. **更复杂的工具**——比如接入数据库查询、调用外部 API
> 4. **任务规划**——对于超复杂任务，可以先让 LLM 做任务分解，再逐步执行

---

# 第六部分：面试话术模板

## 30 秒自我介绍（项目部分）

> "我独立完成了三个大模型应用项目：
> 一个 AI 对话助手、一个 RAG 知识库问答系统、
> 以及一个基于 Function Calling 的 AI Agent。
> 其中 Agent 项目我没有用 LangChain 封装，
> 而是手写了 ReAct 推理循环，支持多工具并发调用和多轮对话记忆。"

## 项目介绍（1 分钟版本）

> "这个 Agent 项目的核心是 ReAct 模式——让 LLM 不仅能思考，还能行动。
> 我给 LLM 注册了计算器、时间查询、网络搜索三个工具，
> 用 JSON Schema 描述每个工具的接口。当用户提问时，
> LLM 会自主判断是否需要调用工具、需要哪个工具、传什么参数。
> 我的代码执行工具后把结果送回 LLM，让它继续推理，直到给出最终答案。
>
> 相比直接用 LangChain Agent，我这个方案更透明——用户能看到 AI 的每一步推理和工具调用过程，
> 也更容易扩展和调试。"

## 遇到不会的问题怎么办？

> "这个问题我没有深入研究过，但我可以从我的项目经验出发，谈一下我的理解……
> 另外如果给我一点时间，我可以查阅资料后给出更准确的回答。"

---

# 第七部分：面试官可能会深挖的点

| 问题方向 | 准备好这些 |
|---------|-----------|
| **原理** | Function Calling 的整个流程、tool_choice 参数的含义 |
| **对比** | 和 LangChain Agent 的区别、和 OpenAI GPTs 的区别 |
| **工程** | tool_calls 的消息格式、tool 消息的 role 含义 |
| **设计** | 为什么用 JSON Schema？为什么返回 string？ |
| **问题** | 如果 LLM 无限循环调用工具怎么办？（答案：max_iter） |
| **扩展** | 如果要加一个新工具怎么做？（答案：三步：写函数、加定义、加映射） |

---

*把这篇文章读透，Function Calling 方向的面试你就能应对自如了。*
