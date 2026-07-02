"""冠军预测页面"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def render_champion(data: dict):
    champion = data["champion_pred"]
    mc = data["mc_results"]
    probs_df = mc["probs_df"]
    scored = data["scored"]

    st.markdown("## 🥇 冠军预测")
    st.markdown("<p style='color:#888'>10,000次蒙特卡洛模拟 | Wilson 95%置信区间 | 固定随机种子42保证可复现</p>", unsafe_allow_html=True)

    # ── 冠军大展示 ─────────────────────────────────
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div class='champion-banner'>
            <div style='font-size:80px; margin:10px 0'>{champion['champion_flag']}</div>
            <h1 style='font-size:40px; color:#FFD700; margin:0'>{champion['champion_name']}</h1>
            <p style='color:#FFA500; font-size:24px; font-weight:bold; margin:8px 0'>夺冠概率 {champion['champion_prob']:.1f}%</p>
            <p style='color:#888'>95% Wilson CI: [{champion['ci_lower']:.1f}%, {champion['ci_upper']:.1f}%]</p>
            <p style='color:#aaa'>综合实力评分：{champion['overall_score']:.1f} / 100</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 夺冠路径 ──────────────────────────────────
    col_path, col_gauge = st.columns([2, 1])
    with col_path:
        st.markdown("### 🗺️ 预测夺冠路径")
        path = champion.get("path", [])
        for i, p in enumerate(path):
            is_final = (p["round"] == "决赛")
            icon = "🏆" if is_final else f"{i+1}️⃣"
            border_color = "#FFD700" if is_final else "#4a6fa5"
            st.markdown(f"""
            <div style='background:#1a2040; border-left:4px solid {border_color}; 
                        border-radius:8px; padding:12px 16px; margin:8px 0;
                        display:flex; justify-content:space-between; align-items:center'>
                <span style='color:{border_color}; font-weight:bold; font-size:15px'>{icon} {p['round']}</span>
                <span style='color:#e0e0e0'>击败 <b>{p['opponent']}</b></span>
                <span style='color:#FFD700; font-size:22px; font-weight:bold'>{p['score']}</span>
                <span style='color:#888; font-size:12px'>{p['decided_by']}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown("### 📊 概率仪表盘")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=champion['champion_prob'],
            delta={"reference": 100 / 48, "valueformat": ".1f"},
            title={"text": f"夺冠概率<br><span style='font-size:12px'>（均等概率={100/48:.1f}%）</span>",
                   "font": {"color": "#FFD700"}},
            number={"suffix": "%", "font": {"color": "#FFD700", "size": 32}},
            gauge={
                "axis": {"range": [0, 40], "tickcolor": "#888"},
                "bar": {"color": "#FFD700"},
                "bgcolor": "#1a2040",
                "bordercolor": "#FFD700",
                "steps": [
                    {"range": [0, 100/48], "color": "#1a2040"},
                    {"range": [100/48, 20], "color": "#2a3560"},
                    {"range": [20, 40], "color": "#3a4570"},
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── TOP 10 夺冠概率详情 ──────────────────────
    st.markdown("### 📈 完整概率排行榜 (TOP 10)")
    top10 = probs_df.head(10).copy()

    col_table, col_funnel = st.columns([2, 1])
    with col_table:
        display = top10[["rank","flag","name","group","champion_prob","final_prob","semi_prob","quarter_prob","ci_lower","ci_upper","overall_score"]].copy()
        display.columns = ["排名","旗帜","球队","小组","夺冠%","决赛%","四强%","八强%","CI下限","CI上限","综合评分"]
        st.dataframe(
            display.style.background_gradient(subset=["夺冠%"], cmap="YlOrBr"),
            use_container_width=True,
            hide_index=True,
        )

    with col_funnel:
        # 漏斗图
        c = probs_df.iloc[0]
        fig_funnel = go.Figure(go.Funnel(
            y=["八强概率", "四强概率", "决赛概率", "夺冠概率"],
            x=[c["quarter_prob"], c["semi_prob"], c["final_prob"], c["champion_prob"]],
            textinfo="value+percent initial",
            marker=dict(color=["#4a6fa5", "#FFA500", "#FF8C00", "#FFD700"]),
        ))
        fig_funnel.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            title=dict(text=f"{c['flag']} {c['name']} 各阶段概率", font=dict(color="#FFD700")),
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.divider()

    # ── Qwen推理报告 ──────────────────────────────
    st.markdown("### 🤖 AI推理分析报告")
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        gen_report = st.button("生成 Qwen 推理报告", type="primary")
    with col_info:
        from config.settings import QWEN_API_KEY
        if QWEN_API_KEY:
            st.success("Qwen API 已配置，将生成AI分析")
        else:
            st.info("未配置API Key，将使用规则模板生成报告")

    if gen_report or "champion_report" not in st.session_state:
        with st.spinner("正在生成推理报告..."):
            from backend.services.qwen_service import generate_champion_report
            report = generate_champion_report(champion, scored)
            st.session_state["champion_report"] = report

    if "champion_report" in st.session_state:
        st.markdown(st.session_state["champion_report"])

        # 下载按钮
        st.download_button(
            label="📥 下载预测报告 (Markdown)",
            data=st.session_state["champion_report"],
            file_name="world_cup_2026_champion_prediction.md",
            mime="text/markdown",
        )
