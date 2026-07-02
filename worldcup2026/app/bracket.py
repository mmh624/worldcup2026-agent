"""淘汰赛对阵树页面"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_bracket(data: dict):
    knockout = data["knockout"]
    scored = data["scored"]

    st.markdown("## 🏆 淘汰赛对阵预测")
    st.markdown("<p style='color:#888'>32强→16强→8强→半决赛→决赛 | 三阶段制胜：90分钟/加时赛/点球大战</p>", unsafe_allow_html=True)

    # ── 决赛高亮 ──────────────────────────────────
    champion = data["champion_pred"]
    final_match = knockout.get("final", [{}])[0]

    if final_match:
        st.markdown(f"""
        <div class='champion-banner' style='margin-bottom:16px'>
            <p style='color:#888; margin:0; font-size:13px'>🏆 预测决赛</p>
            <div style='display:flex; justify-content:center; align-items:center; gap:24px; margin:12px 0'>
                <div style='text-align:center'>
                    <div style='font-size:32px'>{final_match.get('team_a_flag','')}</div>
                    <div style='color:#e0e0e0'>{final_match.get('team_a_name','')}</div>
                </div>
                <div style='text-align:center'>
                    <div style='color:#FFD700; font-size:36px; font-weight:bold'>{final_match.get('score_a',0)} - {final_match.get('score_b',0)}</div>
                    <div style='color:#888; font-size:12px'>{'正常时间' if final_match.get('decided_by')=='regular' else '加时赛' if final_match.get('decided_by')=='extra_time' else '点球大战'}</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:32px'>{final_match.get('team_b_flag','')}</div>
                    <div style='color:#e0e0e0'>{final_match.get('team_b_name','')}</div>
                </div>
            </div>
            <p style='color:#FFD700; font-size:18px; font-weight:bold'>🏆 冠军：{champion['champion_flag']} {champion['champion_name']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── 淘汰赛可视化树（用Plotly绘制） ───────────────
    st.markdown("### 🌲 赛程对阵树")
    _render_bracket_tree(knockout)

    # ── 各轮详情 ──────────────────────────────────
    st.markdown("### 📋 各轮详情")
    round_tabs = st.tabs(["32强", "16强", "八强", "半决赛", "决赛"])

    round_keys = [("r32", "32强"), ("r16", "16强"), ("qf", "八强"), ("sf", "半决赛"), ("final", "决赛")]
    for tab, (rk, rl) in zip(round_tabs, round_keys):
        with tab:
            matches = knockout.get(rk, [])
            if not matches:
                st.info(f"暂无{rl}数据")
                continue

            cols = st.columns(min(len(matches), 2))
            for i, match in enumerate(matches):
                with cols[i % min(len(matches), 2)]:
                    _render_match_card(match)


def _render_match_card(match: dict):
    """单场淘汰赛卡片"""
    decided_cn = {"regular": "⏱️ 正常时间", "extra_time": "⏰ 加时赛", "penalty": "🎯 点球大战"}.get(
        match.get("decided_by", "regular"), ""
    )
    winner_id = match.get("winner_id", "")
    a_win = winner_id == match.get("team_a", "")
    b_win = winner_id == match.get("team_b", "")

    a_style = "color:#FFD700; font-weight:bold" if a_win else "color:#888"
    b_style = "color:#FFD700; font-weight:bold" if b_win else "color:#888"

    st.markdown(f"""
    <div class='match-card'>
        <div style='display:flex; justify-content:space-between; align-items:center; padding:4px 0'>
            <div style='text-align:left; flex:1'>
                <span style='font-size:20px'>{match.get('team_a_flag','')}</span><br>
                <span style='{a_style}; font-size:13px'>{match.get('team_a_name','')}</span>
            </div>
            <div style='text-align:center; flex:0 0 100px'>
                <span style='color:#FFD700; font-size:26px; font-weight:bold'>
                    {match.get('score_a',0)} - {match.get('score_b',0)}
                </span><br>
                <span style='color:#888; font-size:11px'>{decided_cn}</span>
            </div>
            <div style='text-align:right; flex:1'>
                <span style='font-size:20px'>{match.get('team_b_flag','')}</span><br>
                <span style='{b_style}; font-size:13px'>{match.get('team_b_name','')}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_bracket_tree(knockout: dict):
    """用Plotly绘制淘汰赛括号树"""
    fig = go.Figure()

    round_defs = [
        ("r32",  "32强",   0.05),
        ("r16",  "16强",   0.25),
        ("qf",   "八强",   0.45),
        ("sf",   "半决赛", 0.65),
        ("final","决赛",   0.85),
    ]

    max_y = 34  # 32场+间距

    for round_key, round_name, x_pos in round_defs:
        matches = knockout.get(round_key, [])
        if not matches:
            continue

        n = len(matches)
        y_step = max_y / max(n, 1)

        fig.add_annotation(
            x=x_pos, y=max_y + 1,
            text=f"<b>{round_name}</b>",
            showarrow=False,
            font=dict(color="#FFD700", size=13),
            xref="paper", yref="y",
        )

        for i, match in enumerate(matches):
            y_center = max_y - (i + 0.5) * y_step
            winner_id = match.get("winner_id", "")

            for j, (tid, tname, tflag, score) in enumerate([
                (match.get("team_a"), match.get("team_a_name",""), match.get("team_a_flag",""), match.get("score_a",0)),
                (match.get("team_b"), match.get("team_b_name",""), match.get("team_b_flag",""), match.get("score_b",0)),
            ]):
                y = y_center + (0.6 if j == 0 else -0.6)
                is_winner = tid == winner_id
                color = "#FFD700" if is_winner else "#6a7090"
                bg_color = "rgba(255,215,0,0.15)" if is_winner else "rgba(26,32,64,0.8)"

                fig.add_shape(
                    type="rect",
                    x0=x_pos - 0.09, y0=y - 0.5,
                    x1=x_pos + 0.09, y1=y + 0.5,
                    line=dict(color=color, width=1),
                    fillcolor=bg_color,
                    xref="paper", yref="y",
                )
                fig.add_annotation(
                    x=x_pos, y=y,
                    text=f"{tflag} {tname[:6]} <b>{score}</b>",
                    showarrow=False,
                    font=dict(color=color, size=9),
                    xref="paper", yref="y",
                )

            # 连接线到下一轮
            if round_key != "final":
                fig.add_shape(
                    type="line",
                    x0=x_pos + 0.09, y0=y_center,
                    x1=x_pos + 0.14, y1=y_center,
                    line=dict(color="#FFD700", width=1, dash="dot"),
                    xref="paper", yref="y",
                )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.9)",
        font=dict(color="#e0e0e0"),
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[0, max_y + 2]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
