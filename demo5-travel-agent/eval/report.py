"""
评测报告生成器 — Markdown 格式
包含: 总览 → 三维度指标 → Bad Case 分析 → 优化建议 → 前后对比
"""
import os
from datetime import datetime
from .runner import EvalResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_results")


class ReportGenerator:
    """评测报告生成器"""

    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def generate_full_report(
        self, metrics: dict, attribution: dict, summary: dict,
        title: str = "南航智慧出行管家 AI Agent 评测报告",
    ) -> str:
        """生成完整的 Markdown 评测报告"""
        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 头部 ──
        lines.append(f"# {title}")
        lines.append(f"\n> 生成时间: {now} | 模型: DeepSeek-chat | 测试用例: {summary.get('total', 0)} 条")
        lines.append("")

        # ── 1. 总览 ──
        lines.append("## 一、评测总览")
        lines.append("")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|:-----|:-----|")
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        lines.append(f"| 测试用例总数 | {total} |")
        lines.append(f"| 通过数 | {passed} |")
        lines.append(f"| 失败数 | {failed} |")
        pass_rate = summary.get("pass_rate", "0%")
        lines.append(f"| **通过率** | **{pass_rate}** |")
        lines.append(f"| 平均延迟 | {summary.get('avg_latency_ms', 'N/A')}ms |")
        lines.append(f"| 平均工具调用数 | {summary.get('avg_tool_calls', 'N/A')} |")
        lines.append("")

        # ── 2. 三维度 × 三层级 ──
        lines.append("## 二、三维度 × 三层级 评测指标")
        lines.append("")

        # 2.1 结果正确性
        rc = metrics.get("result_correctness", {})
        lines.append("### 2.1 结果正确性")
        lines.append("")
        lines.append(f"| 层级 | 指标 | 数值 |")
        lines.append(f"|:----:|:-----|:----:|")
        lines.append(f"| **L1 北极星** | 端到端任务成功率 | **{rc.get('L1_task_success_rate', 'N/A')}%** |")
        lines.append(f"| **L2 上卷** | 意图识别准确率 | {rc.get('L2_intent_accuracy', 'N/A')}% |")

        by_cat = rc.get("L3_by_category", {})
        for cat, data in by_cat.items():
            cat_names = {"normal": "常规场景", "edge": "边缘场景", "adversarial": "对抗场景"}
            lines.append(f"| **L3 下钻** | {cat_names.get(cat, cat)} | {data['passed']}/{data['total']} ({data['pass_rate']}%) |")
        lines.append("")

        # 2.2 过程合规性
        pc = metrics.get("process_compliance", {})
        lines.append("### 2.2 过程合规性")
        lines.append("")
        lines.append(f"| 层级 | 指标 | 数值 |")
        lines.append(f"|:----:|:-----|:----:|")
        lines.append(f"| **L1 北极星** | 步骤顺序正确率 | {pc.get('L1_step_order_rate', 'N/A')}% |")
        lines.append(f"| **L2 上卷** | 参数传递准确率 | {pc.get('L2_parameter_accuracy', 'N/A')} |")
        lines.append(f"| **L3 下钻** | 降级/兜底次数 | {pc.get('L3_degradation_count', 'N/A')} 次 |")
        lines.append("")

        # 2.3 成本效率
        ce = metrics.get("cost_efficiency", {})
        lat = ce.get("L1_latency", {})
        tc_dist = ce.get("L3_tool_calls", {})
        tc_distribution = tc_dist.get("distribution", {})
        lines.append("### 2.3 成本效率")
        lines.append("")
        lines.append(f"| 层级 | 指标 | 数值 |")
        lines.append(f"|:----:|:-----|:----:|")
        lines.append(f"| **L1 北极星** | 端到端延迟 p50 | {lat.get('p50_ms', 'N/A')}ms |")
        lines.append(f"| **L1 北极星** | 端到端延迟 p99 | {lat.get('p99_ms', 'N/A')}ms |")
        lines.append(f"| **L2 上卷** | 平均延迟 | {lat.get('avg_ms', 'N/A')}ms |")
        lines.append(f"| **L2 上卷** | Token 消耗 | {ce.get('L2_token_estimate', 'N/A')} |")
        lines.append(f"| **L3 下钻** | 平均工具调用/任务 | {tc_dist.get('avg_per_task', 'N/A')} 次 |")
        lines.append(f"| **L3 下钻** | 工具调用范围 | {tc_dist.get('min', 'N/A')} ~ {tc_dist.get('max', 'N/A')} 次 |")
        lines.append(f"| **L3 下钻** | 0次工具调用 | {tc_distribution.get('0_tools', 'N/A')} 个用例 |")
        lines.append(f"| **L3 下钻** | 3+次工具调用 | {tc_distribution.get('3+_tools', 'N/A')} 个用例 |")
        lines.append("")

        # ── 3. Bad Case 分析 ──
        dist = attribution.get("distribution", {})
        ratios = attribution.get("ratios", {})
        total_failed = attribution.get("total_failed", 0)
        lines.append("## 三、Bad Case 归因分析")
        lines.append("")
        lines.append(f"失败用例总数: **{total_failed}**")
        lines.append("")
        lines.append("| 失败模式 | 数量 | 占比 | 说明 |")
        lines.append("|:--------|:----:|:----:|:-----|")
        type_descs = attribution.get("attribution_types", {})
        types_order = ["wrong_tool", "missing_params", "chain_break", "security_risk", "unknown"]
        for t in types_order:
            count = dist.get(t, 0)
            ratio = ratios.get(t, 0)
            desc = type_descs.get(t, "")
            lines.append(f"| **{t}** | {count} | {ratio}% | {desc} |")
        lines.append("")

        # ── 4. 典型 Bad Case ──
        lines.append("## 四、典型 Bad Case（Top 5）")
        lines.append("")
        # 从 per_case 中取前5个
        per_case = attribution.get("per_case", {})
        lines.append("| 用例ID | 归因类型 |")
        lines.append("|:-------|:--------|")
        for i, (case_id, attr) in enumerate(per_case.items()):
            if i >= 10:
                break
            lines.append(f"| {case_id} | {attr} |")
        lines.append("")

        # ── 5. 优化建议 ──
        lines.append("## 五、优化建议")
        lines.append("")
        suggestions = self._generate_suggestions(dist)
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")
        lines.append("")

        # ── 6. 前后对比（占位） ──
        lines.append("## 六、与基线对比")
        lines.append("")
        lines.append("| 指标 | 基线版本 | 优化后 | 提升 |")
        lines.append("|:-----|:-------:|:-----:|:----:|")
        lines.append(f"| 任务完成率 | 76% | {pass_rate} | TBD |")
        lines.append(f"| 端到端延迟 (p50) | 4000ms | {lat.get('p50_ms', 'N/A')}ms | TBD |")
        lines.append(f"| 工具调用准确率 | 78% | TBD | TBD |")
        lines.append("")
        lines.append("> 注: 基线数据来自南航智慧出行管家项目文档，优化后的准确率需在实际迭代中持续追踪。")

        lines.append("")
        lines.append("---")
        lines.append(f"*报告由 AI Agent 评测框架自动生成 · {now}*")

        return "\n".join(lines)

    def _generate_suggestions(self, distribution: dict) -> list[str]:
        """基于失败模式分布给出优化建议"""
        suggestions = []

        wrong = distribution.get("wrong_tool", 0)
        missing = distribution.get("missing_params", 0)
        chain = distribution.get("chain_break", 0)
        security = distribution.get("security_risk", 0)

        if wrong > 0:
            suggestions.append(
                f"**降低工具混淆率**: 有 {wrong} 个用例选错了工具。建议在 System Prompt 中增加高频场景 "
                f"的 Few-shot 示例，明确每个工具的使用边界。预期可减少 35% 的工具混淆。"
            )

        if missing > 0:
            suggestions.append(
                f"**优化参数提取**: 有 {missing} 个用例因参数缺失导致失败。建议增加槽位提取预处理步骤，"
                f"在调用工具前检查必要参数完整性，缺失时主动反问用户。"
            )

        if chain > 0:
            suggestions.append(
                f"**增强链路健壮性**: 有 {chain} 个用例出现链路断裂。建议为每个工具调用添加 "
                f"retry 机制（最多2次），并在工具失败时提供降级方案（如搜索失败则用模型内置知识）。"
            )

        if security > 0:
            suggestions.append(
                f"**加固安全防线**: 有 {security} 个对抗用例存在安全风险。建议在 System Prompt "
                f"中强化安全策略描述，并在执行敏感操作（订票/退票/改签）前增加二次确认步骤。"
            )

        if not suggestions:
            suggestions.append("当前失败模式分布较均匀，建议深入分析具体用例的 trace 日志，定位根因。")

        suggestions.append("建立 Bad Case 回归集，每次模型更新后自动跑回归测试，防止已修复问题复现。")
        return suggestions

    def save_report(self, content: str, filename: str = "eval_report.md") -> str:
        """保存报告到文件"""
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
