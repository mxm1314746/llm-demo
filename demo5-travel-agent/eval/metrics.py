"""
三维度 × 三层级 评分计算
复用南航文档中的评测指标体系
"""
import statistics
from .runner import EvalResult


class MetricsCalculator:
    """三维度 × 三层级 评测指标计算器"""

    def compute_all(self, results: list[EvalResult]) -> dict:
        """计算全部指标"""
        return {
            "summary": self._summary(results),
            "result_correctness": self._result_correctness(results),
            "process_compliance": self._process_compliance(results),
            "cost_efficiency": self._cost_efficiency(results),
        }

    def _summary(self, results: list[EvalResult]) -> dict:
        """北极星指标 — 总体通过率"""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total * 100, 1) if total else 0,
        }

    def _result_correctness(self, results: list[EvalResult]) -> dict:
        """
        维度1: 结果正确性
        L1(北极星): 端到端任务成功率
        L2(上卷): 意图识别准确率、槽位填充完整率
        L3(下钻): 单工具调用成功率
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        # L2: 意图识别准确率（有工具调用且通过 = 意图正确）
        with_tools = [r for r in results if r.tool_call_count > 0]
        intent_accurate = sum(1 for r in with_tools if r.passed)

        # L3: 按分类统计通过率
        by_category = {}
        for cat in ["normal", "edge", "adversarial"]:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.passed)
                by_category[cat] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "pass_rate": round(cat_passed / len(cat_results) * 100, 1),
                }

        return {
            "L1_task_success_rate": round(passed / total * 100, 1) if total else 0,
            "L2_intent_accuracy": round(intent_accurate / len(with_tools) * 100, 1) if with_tools else 0,
            "L3_by_category": by_category,
        }

    def _process_compliance(self, results: list[EvalResult]) -> dict:
        """
        维度2: 过程合规性
        L1: 步骤顺序正确率
        L2: 参数准确率
        L3: 降级/兜底规范性
        """
        total = len(results)

        # 简单估计: 有工具调用且无错误 = 过程合规
        compliant = sum(1 for r in results if r.tool_call_count > 0 and not r.error)

        # 有错误但仍有工具调用 = 做了降级
        with_errors = [r for r in results if r.error and r.tool_call_count > 0]
        degraded = len(with_errors)

        return {
            "L1_step_order_rate": round(compliant / total * 100, 1) if total else 0,
            "L2_parameter_accuracy": "N/A (需人工评估参数准确性)",
            "L3_degradation_count": degraded,
        }

    def _cost_efficiency(self, results: list[EvalResult]) -> dict:
        """
        维度3: 成本效率
        L1: 端到端延迟 p50/p99
        L2: Token 消耗量（估算）
        L3: 工具调用次数分布
        """
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        tool_counts = [r.tool_call_count for r in results]

        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
            p99 = latencies[p99_idx]
            avg_lat = statistics.mean(latencies)
        else:
            p50 = p99 = avg_lat = 0

        return {
            "L1_latency": {
                "p50_ms": round(p50, 0),
                "p99_ms": round(p99, 0),
                "avg_ms": round(avg_lat, 0),
            },
            "L2_token_estimate": "N/A（需要 API usage 返回）",
            "L3_tool_calls": {
                "avg_per_task": round(statistics.mean(tool_counts), 1) if tool_counts else 0,
                "max": max(tool_counts) if tool_counts else 0,
                "min": min(tool_counts) if tool_counts else 0,
                "distribution": {
                    "0_tools": sum(1 for t in tool_counts if t == 0),
                    "1_tool": sum(1 for t in tool_counts if t == 1),
                    "2_tools": sum(1 for t in tool_counts if t == 2),
                    "3+_tools": sum(1 for t in tool_counts if t >= 3),
                },
            },
        }
