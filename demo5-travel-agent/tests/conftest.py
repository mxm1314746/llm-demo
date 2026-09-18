"""
conftest.py — pytest 配置
"""
import sys, os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
