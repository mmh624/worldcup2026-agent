"""
Streamlit 应用入口 — 2026世界杯冠军预测Agent
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="2026世界杯冠军预测 Agent",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局CSS样式（深色世界杯风格）─────────────────────
st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #0a0e1a; color: #e0e0e0; }
    
    /* 侧边栏 */
    section[data-testid="stSidebar"] { background-color: #0f1628; }
    section[data-testid="stSidebar"] .stMarkdown { color: #b0b8d0; }
    
    /* 标题颜色 */
    h1, h2, h3 { color: #FFD700 !important; }
    h4, h5, h6 { color: #FFA500 !important; }
    
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #1a2040, #0f1628);
        border: 1px solid #FFD700;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 0 15px rgba(255,215,0,0.15);
    }
    
    /* 球队卡片 */
    .team-card {
        background: #1a2040;
        border-left: 4px solid #FFD700;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    
    /* 比赛卡片 */
    .match-card {
        background: linear-gradient(90deg, #1a2040 0%, #0d1530 50%, #1a2040 100%);
        border: 1px solid #2a3560;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
        text-align: center;
    }
    
    /* 冠军横幅 */
    .champion-banner {
        background: linear-gradient(135deg, #1a1400, #3d2c00, #1a1400);
        border: 2px solid #FFD700;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(255,215,0,0.3);
    }
    
    /* 进度条颜色 */
    .stProgress > div > div > div { background-color: #FFD700; }
    
    /* 表格样式 */
    .dataframe { background-color: #1a2040 !important; color: #e0e0e0 !important; }
    
    /* 按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #0a0e1a;
        font-weight: bold;
        border: none;
        border-radius: 8px;
    }
    
    /* 分隔线 */
    hr { border-color: #FFD700; opacity: 0.3; }
    
    /* 选择框 */
    .stSelectbox > div { background-color: #1a2040; }
    
    /* 展开框 */
    .streamlit-expanderHeader { color: #FFD700 !important; background-color: #1a2040; }
</style>
""", unsafe_allow_html=True)

# ── 数据预计算（只运行一次，缓存结果）────────────────
@st.cache_resource(show_spinner="正在初始化预测模型...")
def load_all_predictions():
    from utils.data_loader import load_teams
    from models.team_rating import compute_overall_scores
    from models.group_stage import simulate_all_groups, get_group_qualifiers
    from models.knockout import build_bracket_r32, simulate_full_knockout
    from models.champion import run_monte_carlo, get_champion_prediction

    teams   = load_teams()
    scored  = compute_overall_scores(teams)

    # 确定性预测（用于展示）
    group_results = simulate_all_groups(scored, mode="deterministic")
    top2, best_thirds = get_group_qualifiers(group_results)
    bracket = build_bracket_r32(top2, best_thirds, scored)
    knockout = simulate_full_knockout(bracket, scored, mode="deterministic")

    # 蒙特卡洛（N=5000，平衡速度和精度）
    mc_results = run_monte_carlo(scored, n_sims=5000)
    champion_pred = get_champion_prediction(scored, mc_results)

    return {
        "teams": teams,
        "scored": scored,
        "group_results": group_results,
        "top2": top2,
        "best_thirds": best_thirds,
        "bracket": bracket,
        "knockout": knockout,
        "mc_results": mc_results,
        "champion_pred": champion_pred,
    }


# ── 侧边栏导航 ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <span style='font-size:48px'>⚽</span>
        <h2 style='color:#FFD700; margin:8px 0'>世界杯预测</h2>
        <p style='color:#888; font-size:12px'>2026 FIFA World Cup</p>
        <p style='color:#888; font-size:12px'>美国·加拿大·墨西哥</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "选择页面",
        ["🏠 首页总览", "📊 球队实力榜", "🏟️ 小组赛预测", "🏆 淘汰赛对阵", "🥇 冠军预测", "🔍 单场预测"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("<p style='color:#666; font-size:11px; text-align:center'>四层可解释预测架构<br>泊松+蒙特卡洛×10000</p>", unsafe_allow_html=True)

    # API KEY 输入
    with st.expander("⚙️ Qwen API设置"):
        api_key = st.text_input("DashScope API Key", type="password",
                                value=os.environ.get("DASHSCOPE_API_KEY", ""))
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
            from config import settings
            settings.QWEN_API_KEY = api_key
            st.success("API Key已设置")

# ── 加载数据 ──────────────────────────────────────────
data = load_all_predictions()

# ── 页面路由 ──────────────────────────────────────────
if page == "🏠 首页总览":
    from app.home import render_home
    render_home(data)
elif page == "📊 球队实力榜":
    from app.analysis import render_analysis
    render_analysis(data)
elif page == "🏟️ 小组赛预测":
    from app.groups import render_groups
    render_groups(data)
elif page == "🏆 淘汰赛对阵":
    from app.bracket import render_bracket
    render_bracket(data)
elif page == "🥇 冠军预测":
    from app.champion_page import render_champion
    render_champion(data)
elif page == "🔍 单场预测":
    from app.match_detail import render_match_detail
    render_match_detail(data)
