"""小组赛预测页面"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_groups(data: dict):
    group_results = data["group_results"]
    scored = data["scored"]

    st.markdown("## 🏟️ 小组赛预测")
    st.markdown("<p style='color:#888'>泊松分布比分预测 | 12组×6场 = 72场预测 | 每组前2名+8支最佳第三名出线</p>", unsafe_allow_html=True)

    # ── 小组选择器 ─────────────────────────────────
    group_list = sorted(group_results.keys())
    selected_groups = st.multiselect("选择要查看的小组", group_list, default=group_list[:4])

    if not selected_groups:
        st.warning("请至少选择一个小组")
        return

    # ── 按组显示积分表 ─────────────────────────────
    for g in selected_groups:
        table = group_results[g]

        with st.expander(f"📋 **{g} 组**  ({' | '.join(table['flag'].tolist() + [''])[:60]})", expanded=True):
            col_left, col_right = st.columns([3, 2])

            with col_left:
                # 积分表
                display = table[["rank","flag","name","played","wins","draws","losses","gf","ga","gd","points"]].copy()
                display.columns = ["名次","旗帜","球队","场次","胜","平","负","进球","失球","净胜球","积分"]

                def highlight_qualifiers(row):
                    """高亮前两名（绿色）"""
                    if row.name < 2:
                        return ["background-color: rgba(45,138,110,0.3); color: #90EE90"] * len(row)
                    elif row.name == 2:
                        return ["background-color: rgba(74,111,165,0.2); color: #87CEEB"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    display.style.apply(highlight_qualifiers, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )

                # 出线说明
                t1 = table.iloc[0]
                t2 = table.iloc[1]
                t3 = table.iloc[2] if len(table) > 2 else None
                st.markdown(f"""
                <div style='padding:8px 0'>
                    <span style='color:#90EE90'>✅ 直接出线：{t1['flag']} {t1['name']} ({t1['points']}分)
                    ，{t2['flag']} {t2['name']} ({t2['points']}分)</span>
                    {'<br><span style="color:#87CEEB">🔵 可能最佳第三：' + t3["flag"] + " " + t3["name"] + f' ({t3["points"]}分)</span>' if t3 is not None else ""}
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                # 进球统计柱状图
                _render_group_chart(table)

            # ── 小组内场次预测 ──────────────────────
            _render_group_matches(g, table, scored)


def _render_group_chart(table: pd.DataFrame):
    """小组积分/进球图表"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{r['flag']} {r['name']}" for _, r in table.iterrows()],
        y=table["points"],
        name="积分",
        marker_color=["#FFD700" if i < 2 else "#4a6fa5" for i in range(len(table))],
        text=table["points"],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=[f"{r['flag']} {r['name']}" for _, r in table.iterrows()],
        y=table["gf"],
        name="进球数",
        mode="lines+markers",
        line_color="#FFA500",
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=11),
        xaxis=dict(color="#e0e0e0"),
        yaxis=dict(title="积分", color="#e0e0e0", gridcolor="#1a2040"),
        yaxis2=dict(title="进球", overlaying="y", side="right", color="#FFA500"),
        height=220,
        margin=dict(l=10, r=40, t=10, b=40),
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_group_matches(group_label: str, table: pd.DataFrame, scored: pd.DataFrame):
    """展示小组内各场次预测比分"""
    from models.group_stage import compute_lambdas, predict_score, win_draw_loss_probs

    teams_dict = {row["id"]: row.to_dict() for _, row in scored.iterrows() if row["id"] in table["id"].values}

    team_ids = table["id"].tolist()
    st.markdown("**📅 场次预测：**")
    match_cols = st.columns(3)
    match_idx = 0

    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            ta = teams_dict.get(team_ids[i])
            tb = teams_dict.get(team_ids[j])
            if not ta or not tb:
                continue

            lam_a, lam_b = compute_lambdas(
                ta["attack_lambda"], ta["defense_factor"],
                tb["attack_lambda"], tb["defense_factor"],
                a_is_home=ta.get("home_boost", False),
                b_is_home=tb.get("home_boost", False),
            )
            ga, gb = predict_score(lam_a, lam_b)
            pw, pd_, pl = win_draw_loss_probs(lam_a, lam_b)

            with match_cols[match_idx % 3]:
                winner_color = "#90EE90" if ga > gb else ("#FFA500" if ga == gb else "#FF6B6B")
                st.markdown(f"""
                <div class='match-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center'>
                        <span style='font-size:13px'>{ta['flag']} {ta['name']}</span>
                        <span style='color:#FFD700; font-size:22px; font-weight:bold; margin:0 8px'>{ga} - {gb}</span>
                        <span style='font-size:13px'>{tb['name']} {tb['flag']}</span>
                    </div>
                    <div style='margin-top:6px; display:flex; gap:4px'>
                        <span style='background:rgba(45,138,110,0.4); padding:2px 6px; border-radius:4px; font-size:11px; color:#90EE90'>{pw*100:.0f}%胜</span>
                        <span style='background:rgba(74,111,165,0.4); padding:2px 6px; border-radius:4px; font-size:11px; color:#87CEEB'>{pd_*100:.0f}%平</span>
                        <span style='background:rgba(255,107,107,0.3); padding:2px 6px; border-radius:4px; font-size:11px; color:#FF8888'>{pl*100:.0f}%负</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            match_idx += 1
