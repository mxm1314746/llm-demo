"""llm-demo 作品集统一前端主题。

提供品牌色 CSS、头部/页脚 HTML 与通用样式类，供 demo1–demo5 复用，
保证五个项目视觉语言一致。仅依赖字符串拼接，不引入额外依赖。
"""

BRAND = "#003d9e"
BRAND_LIGHT = "#0055c4"

THEME_CSS = """
/* ── 全局 ── */
footer { display: none !important; }
.gradio-container {
    max-width: 100% !important;
    background: #f2f4f8 !important;
}

/* ── 顶部品牌导航栏 ── */
.ui-header {
    background: linear-gradient(135deg, #003d9e 0%, #0055c4 100%);
    padding: 16px 24px;
    border-radius: 0 0 16px 16px;
    margin: -16px -16px 8px -16px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 12px rgba(0,61,158,0.25);
}
.ui-header-logo {
    background: white;
    border-radius: 10px;
    padding: 6px 14px;
    font-weight: 900;
    font-size: 16px;
    color: #003d9e;
    letter-spacing: 1px;
}
.ui-header-title { color: white; font-size: 18px; font-weight: 600; line-height: 1.3; }
.ui-header-sub { color: rgba(255,255,255,0.75); font-size: 12px; }

/* ── 服务/功能卡片网格（响应式） ── */
.ui-card-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 8px 0 16px 0;
}
@media (max-width: 900px) { .ui-card-grid { grid-template-columns: repeat(2, 1fr); } }
.ui-card {
    background: white !important;
    border: 1px solid #edf0f5 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    color: #333 !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    min-height: 0 !important;
}
.ui-card:hover {
    background: #e8f0ff !important;
    border-color: #003d9e !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,61,158,0.12) !important;
}

/* ── 搜索框 ── */
.ui-search input, .ui-search textarea {
    border: none !important;
    box-shadow: none !important;
    font-size: 14px;
}
.ui-search {
    background: white;
    border-radius: 20px !important;
    border: 1px solid #e0e4ea !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── 对话区域 ── */
.ui-chat {
    background: white;
    border-radius: 12px;
    border: 1px solid #edf0f5;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.ui-chat .chatbot { border: none !important; }

/* ── 主按钮 ── */
.ui-btn-primary {
    background: linear-gradient(135deg, #003d9e, #0055c4) !important;
    border: none !important;
    border-radius: 20px !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── 底部导航（可点击切换） ── */
.ui-nav { gap: 8px; }
.ui-nav button {
    background: white !important;
    border: 1px solid #edf0f5 !important;
    border-radius: 12px !important;
    color: #666 !important;
    font-size: 13px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.ui-nav button:hover { color: #003d9e !important; border-color: #003d9e !important; }

/* ── 页脚 ── */
.ui-footer {
    text-align: center;
    color: #9aa3ad;
    font-size: 12px;
    padding: 12px 0 4px 0;
}

/* ── 指标卡片（评测） ── */
.ui-metric {
    text-align: center;
    padding: 16px;
    background: #f5f8ff;
    border-radius: 8px;
}
.ui-metric-value { font-size: 28px; font-weight: 700; color: #003d9e; }
.ui-metric-label { font-size: 12px; color: #666; }
.pass { color: #0d904f; font-weight: 600; }
.fail { color: #d93025; font-weight: 600; }
"""


def header_html(logo, title, subtitle):
    return f"""
    <div class="ui-header">
      <span class="ui-header-logo">{logo}</span>
      <div>
        <div class="ui-header-title">{title}</div>
        <div class="ui-header-sub">{subtitle}</div>
      </div>
    </div>
    """


def footer_html(text):
    return f'<div class="ui-footer">{text}</div>'
