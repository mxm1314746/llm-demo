"""
结构化日志系统 — 替换代码中所有的 print

使用方式:
  from logger import logger
  logger.info("Agent 启动")
  logger.error("API 调用失败", exc_info=True)
"""
import sys
import os
from loguru import logger

# 移除默认 handler
logger.remove()

# 日志格式
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 控制台输出
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=os.getenv("LOG_LEVEL", "DEBUG"),
    colorize=True,
    enqueue=True,
)

# 文件输出（保留最近 7 天，每天轮转）
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger.add(
    os.path.join(LOG_DIR, "agent_{time:YYYY-MM-DD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    enqueue=True,
)

# 错误日志单独文件
logger.add(
    os.path.join(LOG_DIR, "error_{time:YYYY-MM-DD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
    level="ERROR",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    enqueue=True,
)

__all__ = ["logger"]
