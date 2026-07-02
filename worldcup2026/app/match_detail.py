"""单场比赛预测页面"""
import streamlit as st
import plotly.graph_objects as go


def render_match_detail(data: dict):
    scored = data["scored"]

    st.markdown("## 🔍 单场比赛预测分析")
    st.markdown("<p style='color:#888'>选择任意两支球队，实时预测比赛结果，并生成Qwen推理分析</p>", unsafe_allow_html=True)

    # ── 球队选择 ──────────────────────────────────
    all_teams = sorted(scored.to_dict("records"), key=lambda x: x["overall_score"], reverse=True)
    team_options = [f"{t['flag']} {t['name']} (小组{t['group']}, FIFA#{t['fifa_rank']})" for t in all_teams]
    id_map = {opt: t for opt, t in zip(team_options, all_teams)}

    col1, col2 = st.columns(2)
    with col1:
        sel_a = st.selectbox("球队 A", team_options, index=0)
    with col2:
        sel_b = st.selectbox("球队 B", team_options, index=1)

    if sel_a == sel_b:
        st.warning("请选择不同的两支球队")
        return

    ta = id_map[sel_a]
    tb = id_map[sel_b]

    # ── 预测结果 ──────────────────────────────────
    from models.group_stage import compute_lambdas, predict_score, win_draw_loss_probs
    lam_a, lam_b = compute_lambdas(
        ta["attack_lambda"], ta["defense_factor"],
        tb["attack_lambda"], tb["defense_factor"],
        a_is_home=ta.get("home_boost", False),
        b_is_home=tb.get("home_boost", False),
    )
    ga, gb = predict_score(lam_a, lam_b)
    pw, pd_, pl = win_draw_loss_probs(lam_a, lam_b)

    # ── 比赛卡片展示 ──────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        _team_card(ta, is_winner=(ga > gb))
    with c2:
        winner_text = "平局" if ga == gb else ("A胜" if ga > gb else "B胜")
        st.markdown(f"""
        <div style='text-align:center; padding:20px'>
            <div style='color:#FFD700; font-size:52px; font-weight:bold; margin:10px 0'>{ga} - {gb}</div>
            <div style='color:#888; font-size:14px'>{winner_text}</div>
            <div style='color:#888; font-size:12px; margin-top:8px'>
                λ_A={lam_a:.2f} | λ_B={lam_b:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        _team_card(tb, is_winner=(gb > ga))

    # ── 概率条 ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _render_prob_bar(pw, pd_, pl, ta["name"], tb["name"])

    st.divider()

    # ── 比分热力图 ─────────────────────────────────
    col_heat, col_radar = st.columns(2)
    with col_heat:
        st.markdown("### 🌡️ 比分概率热力图")
        _render_score_heatmap(lam_a, lam_b, ta["name"], tb["name"])

    with col_radar:
        st.markdown("### 🕸️ 实力对比雷达图")
        _render_comparison_radar(ta, tb)

    st.divider()

    # ── Qwen推理分析 ──────────────────────────────
    st.markdown("### 🤖 AI推理分析")
    if st.button("生成此场比赛分析", type="primary"):
        with st.spinner("正在生成分析..."):
            match_result = {
                "winner_name": ta["name"] if ga > gb else (tb["name"] if gb > ga else "平局"),
                "score_a": ga, "score_b": gb,
                "decided_by": "regular",
            }
            from backend.services.qwen_service import analyze_match
            analysis = analyze_match(ta, tb, match_result)
            st.markdown(analysis)


def _team_card(team: dict, is_winner: bool):
    border = "#FFD700" if is_winner else "#2a3560"
    bg = "rgba(255,215,0,0.1)" if is_winner else "#1a2040"
    st.markdown(f"""
    <div style='background:{bg}; border:2px solid {border}; border-radius:12px; 
                padding:16px; text-align:center'>
        <div style='font-size:48px'>{team['flag']}</div>
        <div style='color:#FFD700; font-size:20px; font-weight:bold'>{team['name']}</div>
        <div style='color:#888; font-size:13px'>FIFA 排名 #{team['fifa_rank']}</div>
        <div style='color:#FFA500; font-size:15px; margin-top:8px'>综合 {team['overall_score']:.1f}</div>
        {'<div style="color:#FFD700; font-size:11px; margin-top:4px">🏠 东道主加成</div>' if team.get('home_boost') else ''}
    </div>
    """, unsafe_allow_html=True)


def _render_prob_bar(pw: float, pd_: float, pl: float, name_a: str, name_b: str):
    """胜平负概率条形图"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[pw * 100], y=["概率"],
        orientation="h", name=f"{name_a}胜",
        marker_color="#90EE90",
        text=[f"{name_a}胜 {pw*100:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[pd_ * 100], y=["概率"],
        orientation="h", name="平局",
        marker_color="#87CEEB",
        text=[f"平局 {pd_*100:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[pl * 100], y=["概率"],
        orientation="h", name=f"{name_b}胜",
        marker_color="#FF8888",
        text=[f"{name_b}胜 {pl*100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="h", font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_score_heatmap(lam_a: float, lam_b: float, name_a: str, name_b: str):
    """比分热力图（前7×7）"""
    from models.group_stage import _poisson_pmf
    n = 7
    z = []
    for j in range(n):
        row = [_poisson_pmf(i, lam_a) * _poisson_pmf(j, lam_b) * 100 for i in range(n)]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{name_a[:4]}{i}" for i in range(n)],
        y=[f"{name_b[:4]}{j}" for j in range(n)],
        colorscale=[[0, "#0a0e1a"], [0.5, "#4a6fa5"], [1, "#FFD700"]],
        text=[[f"{v:.2f}%" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=10),
        xaxis=dict(color="#e0e0e0"),
        yaxis=dict(color="#e0e0e0"),
        height=300,
        margin=dict(l=50, r=20, t=10, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_comparison_radar(ta: dict, tb: dict):
    """双队实力对比雷达图"""
    dims = ["score_history", "score_fifa", "score_ad", "score_player", "score_form"]
    labels = ["历史底蕴", "FIFA排名", "攻防效率", "球员班底", "近期状态"]

    fig = go.Figure()
    for team, color in [(ta, "#FFD700"), (tb, "#4a6fa5")]:
        vals = [team.get(d, 50) for d in dims] + [team.get(dims[0], 50)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]],
            fill="toself", name=f"{team['flag']} {team['name']}",
            line_color=color, opacity=0.8,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], color="#888", gridcolor="#2a3560"),
            angularaxis=dict(color="#e0e0e0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
