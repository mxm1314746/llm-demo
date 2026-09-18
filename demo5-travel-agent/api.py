"""
FastAPI 封装 — 将 Agent 暴露为 REST API

启动: uvicorn api:app --reload --port 8000
访问: http://localhost:8000/docs (Swagger)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from agent.orchestrator import TravelAgent
from agent.prompt import build_system_prompt
from logger import logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
agent = TravelAgent(client, model=MODEL)
system_prompt = build_system_prompt()

logger.info("FastAPI 服务启动")
logger.info(f"模型: {MODEL}")
logger.info(f"Base URL: {BASE_URL}")

app = FastAPI(
    title="南航智慧出行管家 API",
    description="AI Agent REST API — 航班查询、订票、值机、改签、退票、选座",
    version="1.0.0",
)


# ══════════════════════════════════════
# 请求/响应模型
# ══════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入", example="帮我查明天北京到上海的航班")
    history: list = Field(default=[], description="对话历史（可选）")
    temperature: float = Field(default=0.5, ge=0.0, le=2.0, description="温度")
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="Top P")
    max_iter: int = Field(default=5, ge=1, le=10, description="最大推理步数")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent 回复")
    tool_calls: list = Field(default=[], description="工具调用记录")
    tool_call_count: int = Field(default=0, description="工具调用次数")

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = MODEL
    tools: int = 10


# ══════════════════════════════════════
# API 端点
# ══════════════════════════════════════

@app.get("/", tags=["系统"])
async def root():
    """API 根路径"""
    return {"service": "南航智慧出行管家", "version": "1.0.0", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health():
    """健康检查"""
    return HealthResponse()

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(req: ChatRequest):
    """
    Agent 对话接口

    发送用户消息，Agent 自动调用工具并返回结果。
    支持多轮对话（传入 history 参数）。
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    agent.enable_tracing()
    agent.clear_trace()

    try:
        # 执行 Agent 对话
        gen = agent.chat(
            message=req.message,
            history=list(req.history) if req.history else [],
            messages_state=None,
            system_prompt=system_prompt,
            max_iter=req.max_iter,
            temperature=req.temperature,
            top_p=req.top_p,
        )

        # 消费所有 yield
        final_response = ""
        for history, _, _ in gen:
            if history and history[-1]["role"] == "assistant":
                final_response = history[-1]["content"]

        # 获取工具调用记录
        trace = agent.get_trace()
        tool_calls = [t["tool"] for t in trace]

        return ChatResponse(
            response=final_response,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        agent.disable_tracing()

@app.post("/chat/stream", tags=["Agent"])
async def chat_stream(req: ChatRequest):
    """
    Agent 流式对话接口（Server-Sent Events）

    返回 SSE 格式的流式响应，每步推理实时推送。
    """
    from fastapi.responses import StreamingResponse
    import json
    import time

    async def event_generator():
        try:
            gen = agent.chat(
                message=req.message,
                history=[],
                messages_state=None,
                system_prompt=system_prompt,
                max_iter=req.max_iter,
                temperature=req.temperature,
                top_p=req.top_p,
            )

            for history_val, _, _ in gen:
                if history_val and len(history_val) > 0:
                    last = history_val[-1]
                    if last["role"] == "assistant":
                        yield f"data: {json.dumps({'type': 'partial', 'content': last['content']}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
