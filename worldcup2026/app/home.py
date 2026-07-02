"""首页总览页面"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def render_home(data: dict):
    champion = data["champion_pred"]
    mc = data["mc_results"]
    probs_df = mc["probs_df"]
    scored = data["scored"]

    # ── 顶部英雄区 ─────────────────────────────────
    st.markdown(f"""
    <div class='champion-banner'>
        <h1 style='font-size:42px; margin:0; color:#FFD700'>⚽ 2026 FIFA 世界杯</h1>
        <p style='color:#aaa; font-size:16px; margin:8px 0'>美国 · 加拿大 · 墨西哥 | 2026.06.11 - 07.19</p>
        <p style='color:#FFD700; font-size:18px; margin:4px 0'>四层可解释预测 Agent | 蒙特卡洛 × 10,000 次模拟</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 冠军预测速览 ─────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"""
        <div class='champion-banner' style='padding:20px'>
            <p style='color:#aaa; font-size:14px; margin:0'>🏆 预测冠军</p>
            <h2 style='font-size:36px; margin:8px 0; color:#FFD700'>{champion['champion_flag']} {champion['champion_name']}</h2>
            <p style='color:#FFA500; font-size:20px; font-weight:bold'>{champion['champion_prob']:.1f}% 夺冠概率</p>
            <p style='color:#888; font-size:12px'>95% CI: [{champion['ci_lower']:.1f}%, {champion['ci_upper']:.1f}%]</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#888; font-size:13px; margin:0'>参赛球队</p>
            <h2 style='color:#FFD700; font-size:32px; margin:4px 0'>48</h2>
            <p style='color:#aaa; font-size:12px'>12个小组</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#888; font-size:13px; margin:0'>模拟次数</p>
            <h2 style='color:#FFD700; font-size:32px; margin:4px 0'>10K</h2>
            <p style='color:#aaa; font-size:12px'>蒙特卡洛</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color:#888; font-size:13px; margin:0'>预测维度</p>
            <h2 style='color:#FFD700; font-size:32px; margin:4px 0'>5</h2>
            <p style='color:#aaa; font-size:12px'>白盒可解释</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 夺冠概率TOP10图表 ──────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 夺冠概率 TOP 10")
        top10 = probs_df.head(10).copy()
        top10["display"] = top10["flag"] + " " + top10["name"]

        fig = go.Figure()
        colors = ["#FFD700" if i == 0 else "#FFA500" if i < 3 else "#4a6fa5"
                  for i in range(len(top10))]
        fig.add_trace(go.Bar(
            x=top10["champion_prob"],
            y=top10["display"],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1f}%" for p in top10["champion_prob"]],
            textposition="outside",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(top10["ci_upper"] - top10["champion_prob"]).tolist(),
                arrayminus=(top10["champion_prob"] - top10["ci_lower"]).tolist(),
                color="rgba(255,255,255,0.4)",
            )
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(title="夺冠概率 (%)", gridcolor="#1a2040", color="#e0e0e0"),
            yaxis=dict(autorange="reversed", gridcolor="#1a2040", color="#e0e0e0"),
            height=380,
            margin=dict(l=10, r=60, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🗺️ 赛事路线图")
        # 淘汰赛路径
        path_items = champion.get("path", [])
        if path_items:
            for i, p in enumerate(path_items):
                icon = "🏆" if p["round"] == "决赛" else "✅"
                st.markdown(f"""
                <div class='team-card'>
                    <span style='color:#FFD700'>{icon} {p['round']}</span><br>
                    <span style='color:#e0e0e0'>击败 <b>{p['opponent']}</b></span>
                    <span style='color:#FFA500; float:right; font-size:18px'>{p['score']}</span><br>
                    <span style='color:#888; font-size:12px'>{p['decided_by']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("运行预测后显示路径")

    st.divider()

    # ── 架构说明 ───────────────────────────────────
    st.markdown("### ⚙️ 预测模型架构")
    a1, a2, a3, a4 = st.columns(4)
    arch_items = [
        ("Layer 0", "五维评分", "历史底蕴×20% + FIFA排名×30% + 攻防效率×20% + 球员班底×15% + 近期状态×15%", "#FFD700"),
        ("Layer 1", "泊松预测", "λ=1.32×攻击力×防守力×东道主加成，独立泊松分布计算胜平负概率", "#FFA500"),
        ("Layer 2", "赛程推演", "小组赛积分榜→32强淘汰赛三阶段（90分钟/加时/点球）", "#FF8C00"),
        ("Layer 3", "蒙特卡洛", "N=10,000次模拟聚合，Wilson 95%置信区间，随机种子42可复现", "#FF6B00"),
    ]
    for col, (layer, title, desc, color) in zip([a1, a2, a3, a4], arch_items):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; min-height:140px'>
                <p style='color:{color}; font-weight:bold; margin:0'>{layer}</p>
                <h4 style='color:{color}; margin:4px 0'>{title}</h4>
                <p style='color:#aaa; font-size:12px; margin:0'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── 各大洲球队分布 ──────────────────────────────
    st.markdown("### 🌍 参赛球队各大洲分布")
    conf_counts = scored["confederation"].value_counts().reset_index()
    conf_counts.columns = ["confederation", "count"]
    conf_map = {"UEFA": "欧洲UEFA", "CONMEBOL": "南美CONMEBOL", "CONCACAF": "北中美CONCACAF",
                "CAF": "非洲CAF", "AFC": "亚洲AFC", "OFC": "大洋洲OFC"}
    conf_counts["confederation_cn"] = conf_counts["confederation"].map(conf_map)

    fig2 = px.pie(
        conf_counts, values="count", names="confederation_cn",
        color_discrete_sequence=["#FFD700", "#FFA500", "#FF8C00", "#4a6fa5", "#2d8a6e", "#8a4a8a"],
        hole=0.4,
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(font=dict(color="#e0e0e0")),
    )
    fig2.update_traces(textfont_color="#e0e0e0")
    st.plotly_chart(fig2, use_container_width=True)
