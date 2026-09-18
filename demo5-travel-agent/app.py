"""
南航智慧出行管家 — AI Agent 演示应用
参照南航APP设计语言: 蓝白品牌色 · 卡片式布局 · 服务图标矩阵 · 极简对话

启动: python app.py
端口: http://localhost:7864
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from logger import logger

import ui_theme as ui

from agent.orchestrator import TravelAgent
from agent.prompt import build_system_prompt
from eval.test_cases import TestCaseLoader, generate_test_cases
from eval.runner import EvalRunner
from eval.metrics import MetricsCalculator
from eval.attribution import BadCaseAttributor
from eval.report import ReportGenerator

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
agent = TravelAgent(client, model=MODEL)
system_prompt = build_system_prompt()

# ═══ 服务卡片与热门航线（原生按钮，跨 Tab 可用） ═══
SERVICES = [
    ("🔍", "航班查询", "帮我查航班，北京到上海"),
    ("🎫", "机票预订", "我要订机票"),
    ("📦", "订单查询", "查一下我的订单CS-20260708-1234现在什么状态"),
    ("✅", "在线值机", "帮我办理值机"),
    ("💺", "优选座位", "帮我选座位"),
    ("🔄", "机票改签", "我要改签机票"),
    ("❌", "自助退票", "我要退票"),
    ("🧳", "行李查询", "行李能带多少？托运规定是什么？"),
    ("🏢", "机场信息", "机场怎么去？有哪些设施？"),
]

TRIPS = [
    ("北京 → 上海", "明天 · ¥1280起", "帮我查明天北京到上海的航班"),
    ("广州 → 北京", "明天 · ¥1520起", "帮我看明天广州飞北京的航班"),
    ("深圳 → 北京", "明天 · ¥1420起", "明天深圳飞北京有什么航班？"),
    ("北京 → 成都", "明天 · ¥1150起", "明天北京到成都有什么航班？"),
]

EXAMPLE_QUERIES = [
    "帮我查一下明天北京到上海的航班",
    "查一下后天上海飞广州有哪些航班",
    "我的订单CS-20260708-1234现在什么状态？值机了吗？",
    "帮我值机，订单号CS-20260708-1234，旅客张三",
    "帮我选个靠窗的座位，订单CS-20260708-1234",
    "退票手续费怎么算？",
    "经济舱能带多少行李？充电宝能托运吗？",
    "首都机场有哪些航站楼？怎么去市区？",
]


def clear_all():
    return [], "", None


# ╔══════════════════════════════════════════════════════════════╗
# ║  评测 Pipeline                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def run_eval_pipeline(progress=gr.Progress()):
    progress(0, desc="初始化...")
    generate_test_cases()
    loader = TestCaseLoader()
    cases = loader.load_all()

    runner = EvalRunner(agent)
    results = []
    total = len(cases)

    for i, case in enumerate(cases):
        progress(0.1 + 0.6 * i / total, desc=f"[{i + 1}/{total}] {case['id']}")
        result = runner.run_single(case, system_prompt, temperature=0.3)
        results.append(result)
        time.sleep(0.1)

    progress(0.75, desc="计算指标...")
    calculator = MetricsCalculator()
    metrics = calculator.compute_all(results)

    progress(0.85, desc="Bad Case 归因...")
    attributor = BadCaseAttributor()
    attribution = attributor.classify_all(results, cases)

    progress(0.95, desc="生成报告...")
    reporter = ReportGenerator()
    summary = runner.get_summary()
    report = reporter.generate_full_report(metrics, attribution, summary)
    report_path = reporter.save_report(report)

    # 指标卡片
    dist = attribution.get("distribution", {})
    ratios = attribution.get("ratios", {})
    rc = metrics.get("result_correctness", {})
    lat = metrics.get("cost_efficiency", {}).get("L1_latency", {})

    metrics_html = f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;">
  <div class="ui-metric">
    <div class="ui-metric-value">{summary['pass_rate']}</div>
    <div class="ui-metric-label">总通过率</div>
  </div>
  <div class="ui-metric">
    <div class="ui-metric-value">{lat.get('p50_ms', 'N/A')}ms</div>
    <div class="ui-metric-label">延迟 P50</div>
  </div>
  <div class="ui-metric">
    <div class="ui-metric-value">{summary['avg_tool_calls']}</div>
    <div class="ui-metric-label">平均工具调用</div>
  </div>
  <div class="ui-metric">
    <div class="ui-metric-value">{rc.get('L2_intent_accuracy', 'N/A')}%</div>
    <div class="ui-metric-label">意图准确率</div>
  </div>
</div>

### 三维度指标
| 维度 | 北极星(L1) | 上卷(L2) | 下钻(L3) |
|------|:--------:|:------:|:------:|
| 结果正确性 | {rc.get('L1_task_success_rate', 'N/A')}% | {rc.get('L2_intent_accuracy', 'N/A')}% | 常规{rc.get('L3_by_category',{}).get('normal',{}).get('pass_rate','-')}% / 边缘{rc.get('L3_by_category',{}).get('edge',{}).get('pass_rate','-')}% / 对抗{rc.get('L3_by_category',{}).get('adversarial',{}).get('pass_rate','-')}% |
| 过程合规性 | {metrics.get('process_compliance',{}).get('L1_step_order_rate','N/A')}% | N/A | 降级{metrics.get('process_compliance',{}).get('L3_degradation_count','0')}次 |
| 成本效率 | {lat.get('p50_ms','N/A')}ms | {lat.get('p99_ms','N/A')}ms | {metrics.get('cost_efficiency',{}).get('L3_tool_calls',{}).get('avg_per_task','N/A')}次/任务 |

### 失败模式分布
| 类型 | 数量 | 占比 |
|------|:----:|:----:|
| 工具选错 | {dist.get('wrong_tool', 0)} | {ratios.get('wrong_tool', 0)}% |
| 参数缺失 | {dist.get('missing_params', 0)} | {ratios.get('missing_params', 0)}% |
| 链路断裂 | {dist.get('chain_break', 0)} | {ratios.get('chain_break', 0)}% |
| 安全风险 | {dist.get('security_risk', 0)} | {ratios.get('security_risk', 0)}% |
"""

    results_md = "| ID | 场景 | 分类 | 结果 | 延迟 |\n"
    results_md += "|----|------|------|:----:|:----:|\n"
    for r in results[:50]:
        status = '<span class="pass">PASS</span>' if r.passed else '<span class="fail">FAIL</span>'
        cat_labels = {"normal": "常规", "edge": "边缘", "adversarial": "对抗"}
        results_md += f"| {r.case_id} | {r.scenario[:18]} | {cat_labels.get(r.category, r.category)} | {status} | {r.latency_ms:.0f}ms |\n"

    # Bad Case
    per_case = attribution.get("per_case", {})
    bad_md = ""
    for case_id, attr in list(per_case.items())[:15]:
        bad_md += f"- **{case_id}**: {attr}\n"
    if not bad_md:
        bad_md = "🎉 无失败用例！"

    progress(1.0, desc="完成!")
    return report, metrics_html, results_md, bad_md, f"✅ 评测完成 ({summary['passed']}/{summary['total']}) | 报告: {report_path}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  Gradio UI — 南航APP风格（统一主题 + 原生交互）              ║
# ╚══════════════════════════════════════════════════════════════╝

with gr.Blocks(title="南航智慧出行管家", fill_height=True) as demo:

    gr.HTML(ui.header_html("CZ", "智慧出行管家", "让算法多跑路，让旅客少动手"))

    tabs = gr.Tabs()
    with tabs:
        # ════════════════════════════════════════════════════
        #  Tab 1: 首页
        # ════════════════════════════════════════════════════
        with gr.TabItem("🏠 首页", id="tab-home"):
            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(
                        label="", placeholder="🔍  告诉我您的出行需求… 如：明天北京到上海的航班",
                        scale=8, container=True,
                        elem_classes=["ui-search"],
                    )
                with gr.Column(scale=1, min_width=80):
                    send_btn = gr.Button("发送", variant="primary", elem_classes=["ui-btn-primary"])

            # 服务卡片（原生按钮，点击填入需求）
            fill_targets = []
            for chunk in (SERVICES[:4], SERVICES[4:]):
                with gr.Row():
                    for icon, label, query in chunk:
                        btn = gr.Button(f"{icon} {label}", elem_classes=["ui-card"])
                        fill_targets.append((btn, query))

            gr.Markdown("### 🔥 热门航线")
            with gr.Row():
                for route, info, query in TRIPS:
                    btn = gr.Button(f"{route}\n{info}", elem_classes=["ui-card"])
                    fill_targets.append((btn, query))

            gr.Markdown("### 💬 出行管家对话")
            chatbot = gr.Chatbot(
                label="", height=350,
                placeholder="AI 管家将为您自动查询航班、订票、值机...",
                elem_classes=["ui-chat"],
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", scale=1, variant="secondary")
                gr.Markdown(
                    "💡 *试试点击上方服务卡片或热门航线，AI管家将自动为您服务*",
                    container=False,
                )

            # 底部导航（真实切换 Tab）
            with gr.Row(elem_classes=["ui-nav"]):
                nav_home = gr.Button("🏠 首页")
                nav_trip = gr.Button("✈️ 行程服务")
                nav_eval = gr.Button("📊 评测中心")

            with gr.Accordion("⚙️ 高级设置", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="温度")
                    top_p = gr.Slider(0.0, 1.0, 0.8, step=0.05, label="Top P")
                    max_iter = gr.Slider(1, 8, 5, step=1, label="最大推理步数")
                sp = gr.Textbox(label="System Prompt", value=system_prompt, lines=3)

            messages_state = gr.State(None)

            inputs = [msg, chatbot, messages_state, sp, max_iter, temperature, top_p]
            outputs = [chatbot, msg, messages_state]
            send_btn.click(agent.chat, inputs, outputs, concurrency_limit=1)
            msg.submit(agent.chat, inputs, outputs, concurrency_limit=1)
            clear_btn.click(clear_all, None, outputs)

            for btn, query in fill_targets:
                btn.click(lambda q=query: q, None, [msg])

            nav_home.click(lambda: gr.update(selected="tab-home"), None, tabs)
            nav_trip.click(lambda: gr.update(selected="tab-trip"), None, tabs)
            nav_eval.click(lambda: gr.update(selected="tab-eval"), None, tabs)

        # ════════════════════════════════════════════════════
        #  Tab 2: 行程服务
        # ════════════════════════════════════════════════════
        with gr.TabItem("✈️ 行程服务", id="tab-trip"):
            gr.Markdown("## ✈️ 出行全流程服务")
            gr.Markdown("*点击服务卡片自动填入需求并返回首页对话*")
            trip_fill = []
            for chunk in (SERVICES[:4], SERVICES[4:]):
                with gr.Row():
                    for icon, label, query in chunk:
                        btn = gr.Button(f"{icon} {label}", elem_classes=["ui-card"])
                        trip_fill.append((btn, query))

            gr.Markdown("### 📋 常用问题速查")
            gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=msg, label="")

            for btn, query in trip_fill:
                btn.click(
                    lambda q=query: (q, gr.update(selected="tab-home")),
                    None, [msg, tabs],
                )

        # ════════════════════════════════════════════════════
        #  Tab 3: 评测中心
        # ════════════════════════════════════════════════════
        with gr.TabItem("📊 评测中心", id="tab-eval"):
            gr.Markdown("""
            ### 📊 AI Agent 评测框架
            基于 **三维度 × 三层级** 评测体系 · 50 条测试用例 · 4 类 Bad Case 归因
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    run_btn = gr.Button("🚀 开始评测", variant="primary", size="lg",
                                        elem_classes=["ui-btn-primary"])
                    eval_status = gr.Textbox(label="状态", value="就绪", interactive=False)
                    gr.Markdown("""
                    **评测维度:**
                    - ✅ 结果正确性
                    - 📋 过程合规性
                    - ⚡ 成本效率

                    **用例分布:**
                    - 常规 30 · 边缘 13 · 对抗 7
                    """)

                with gr.Column(scale=5):
                    metrics_display = gr.Markdown("### 等待评测...")

            with gr.Row():
                results_table = gr.Markdown("### 详细结果")
            with gr.Accordion("🔍 Bad Case 分析", open=False):
                bad_cases = gr.Markdown("")
            with gr.Accordion("📝 完整报告", open=False):
                full_report = gr.Markdown("")

            run_btn.click(
                run_eval_pipeline,
                None,
                [full_report, metrics_display, results_table, bad_cases, eval_status],
            )

    gr.HTML(ui.footer_html("Demo5 · 南航「智慧出行管家」个人复刻 · 独立开发 GitHub @mxm1314746"))


if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_api_key_here":
        logger.error("DEEPSEEK_API_KEY 未配置")
        exit(1)

    logger.info("南航智慧出行管家启动")
    logger.info(f"地址: http://localhost:7864")
    logger.info(f"模型: {MODEL}")
    logger.info(f"工具数: 10")
    demo.launch(
        server_name="127.0.0.1", server_port=7864,
        theme=gr.themes.Soft(), css=ui.THEME_CSS,
    )
