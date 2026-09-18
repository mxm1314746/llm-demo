# ============================================
# 南航智慧出行管家 — Docker 镜像
# 基于 Python 3.12 slim
# ============================================
FROM python:3.12-slim

LABEL maintainer="mxm1314746"
LABEL description="南航智慧出行管家 AI Agent"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建非 root 用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 7864

# 启动命令
CMD ["python", "app.py"]
