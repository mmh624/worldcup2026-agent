"""球队实力分析页面"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def render_analysis(data: dict):
    scored = data["scored"]
    mc = data["mc_results"]
    probs_df = mc["probs_df"]

    st.markdown("## 📊 球队实力分析")
    st.markdown("<p style='color:#888'>五维度综合评分 | 基于FIFA排名、历史战绩、攻防效率、球员班底、近期状态</p>", unsafe_allow_html=True)

    # ── 过滤器 ─────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        groups_filter = st.multiselect("按小组筛选", sorted(scored["group"].unique()),
                                        default=sorted(scored["group"].unique()))
    with col_f2:
        confs = st.multiselect("按洲际联合会筛选", sorted(scored["confederation"].unique()),
                                default=sorted(scored["confederation"].unique()))

    filtered = scored[scored["group"].isin(groups_filter) & scored["confederation"].isin(confs)].copy()
    filtered = filtered.sort_values("overall_score", ascending=False).reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)

    # ── 雷达图（TOP5球队对比）────────────────────────
    st.markdown("### 🕸️ TOP 5 球队五维度雷达图")
    top5 = filtered.head(5)
    dims = ["score_history", "score_fifa", "score_ad", "score_player", "score_form"]
    dim_labels = ["历史底蕴", "FIFA排名", "攻防效率", "球员班底", "近期状态"]
    colors_radar = ["#FFD700", "#FFA500", "#4a6fa5", "#2d8a6e", "#8a4a8a"]

    fig_radar = go.Figure()
    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[d] for d in dims] + [row[dims[0]]]
        labels_loop = dim_labels + [dim_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=labels_loop,
            fill="toself", name=f"{row['flag']} {row['name']}",
            line_color=colors_radar[i % len(colors_radar)],
            fillcolor=colors_radar[i % len(colors_radar)].replace("#", "rgba(") + ",0.1)",
            opacity=0.8,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], color="#888", gridcolor="#2a3560"),
            angularaxis=dict(color="#e0e0e0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=420,
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── 气泡图（进攻 vs 防守 vs 综合评分）───────────
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("### ⚡ 攻防平衡分析")
        merged = filtered.merge(probs_df[["id", "champion_prob"]], on="id", how="left")
        merged["champion_prob"] = merged["champion_prob"].fillna(0)
        merged["display"] = merged["flag"] + " " + merged["name"]

        fig_bubble = px.scatter(
            merged, x="attack_score", y="defense_score",
            size="overall_score", color="champion_prob",
            hover_name="display",
            color_continuous_scale=[[0, "#1a2040"], [0.5, "#4a6fa5"], [1, "#FFD700"]],
            labels={"attack_score": "进攻评分", "defense_score": "防守评分",
                    "champion_prob": "夺冠概率(%)"},
            text="flag",
        )
        fig_bubble.update_traces(textposition="middle center")
        fig_bubble.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.8)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(gridcolor="#1a2040", color="#e0e0e0"),
            yaxis=dict(gridcolor="#1a2040", color="#e0e0e0"),
            height=380,
            margin=dict(l=30, r=30, t=10, b=30),
            coloraxis_colorbar=dict(tickfont=dict(color="#e0e0e0")),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col_b2:
        st.markdown("### 📈 综合评分排行")
        top15 = filtered.head(15).copy()
        top15["display"] = top15["flag"] + " " + top15["name"]
        colors_bar = ["#FFD700" if i < 3 else "#4a6fa5" for i in range(len(top15))]

        fig_bar = go.Figure(go.Bar(
            x=top15["overall_score"], y=top15["display"],
            orientation="h", marker_color=colors_bar,
            text=[f"{s:.1f}" for s in top15["overall_score"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(range=[0, 100], gridcolor="#1a2040", color="#e0e0e0"),
            yaxis=dict(autorange="reversed", color="#e0e0e0"),
            height=400,
            margin=dict(l=10, r=50, t=10, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── 详细数据表 ─────────────────────────────────
    st.markdown("### 📋 详细评分数据")
    display_df = filtered[[
        "rank", "flag", "name", "group", "confederation",
        "overall_score", "score_history", "score_fifa", "score_ad",
        "score_player", "score_form", "fifa_rank"
    ]].copy()
    display_df.columns = ["排名", "旗帜", "球队", "小组", "联合会",
                           "综合评分", "历史底蕴", "FIFA排名分", "攻防效率",
                           "球员班底", "近期状态", "FIFA排名"]
    st.dataframe(
        display_df.style.background_gradient(subset=["综合评分"], cmap="YlOrBr"),
        use_container_width=True, height=400,
    )

    # ── 评分权重说明 ───────────────────────────────
    with st.expander("📖 评分体系说明"):
        st.markdown("""
        | 维度 | 权重 | 计算方式 |
        |------|------|---------|
        | 历史底蕴 | 20% | 世界杯冠军×10 + 亚军×5 + 四强×2，再归一化 |
        | FIFA排名 | 30% | 1/排名的倒数归一化（排名越高得分越高） |
        | 攻防效率 | 20% | (进攻评分+防守评分)/2，归一化到0-100 |
        | 球员班底 | 15% | 球员综合实力评分归一化 |
        | 近期状态 | 15% | 近期表现评分归一化 |
        
        > 所有维度均归一化到0-100分，加权求和得到综合评分。
        """)
