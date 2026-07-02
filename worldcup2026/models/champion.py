"""
Layer 3 — 蒙特卡洛冠军概率聚合
─────────────────────────────────────────────────────────
对全赛程进行 N 次独立模拟，统计每支球队夺冠次数，
计算：
  - 夺冠概率
  - 进入决赛概率
  - 进入四强概率
  - Wilson 95% 置信区间（控制小样本下的不确定性）
─────────────────────────────────────────────────────────
"""
import math
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import MONTE_CARLO_N, RANDOM_SEED
from models.group_stage import simulate_all_groups, get_group_qualifiers
from models.knockout import build_bracket_r32, simulate_full_knockout


def _wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson得分区间（95% CI），适合小概率事件。
    返回 (lower, upper)
    """
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_monte_carlo(
    scored_df: pd.DataFrame,
    n_sims: int = MONTE_CARLO_N,
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    进行 n_sims 次全赛程蒙特卡洛模拟。

    返回:
    {
        "champion_counts":    {team_id: count},
        "final_counts":       {team_id: count},
        "semifinal_counts":   {team_id: count},
        "quarterfinal_counts":{team_id: count},
        "n_sims":             n_sims,
        "probs_df":           DataFrame（完整概率表，按夺冠概率排序）
    }
    """
    np.random.seed(seed)
    rng = random.Random(seed)

    champion_counts    = defaultdict(int)
    final_counts       = defaultdict(int)
    semifinal_counts   = defaultdict(int)
    quarterfinal_counts= defaultdict(int)

    for sim_i in range(n_sims):
        # 小组赛
        group_results = simulate_all_groups(scored_df, mode="sample")
        top2, best_thirds = get_group_qualifiers(group_results)

        # 构建32强对阵
        bracket = build_bracket_r32(top2, best_thirds, scored_df)

        # 淘汰赛
        knockout = simulate_full_knockout(bracket, scored_df, mode="sample", rng=rng)

        champion_id = knockout.get("champion", {}).get("id")
        if champion_id:
            champion_counts[champion_id] += 1

        # 统计各阶段出现次数
        for match in knockout.get("final", []):
            final_counts[match["team_a"]] += 1
            final_counts[match["team_b"]] += 1

        for match in knockout.get("sf", []):
            semifinal_counts[match["team_a"]] += 1
            semifinal_counts[match["team_b"]] += 1

        for match in knockout.get("qf", []):
            quarterfinal_counts[match["team_a"]] += 1
            quarterfinal_counts[match["team_b"]] += 1

    # ── 构建概率DataFrame ────────────────────────────
    rows = []
    for _, row in scored_df.iterrows():
        tid = row["id"]
        c_cnt = champion_counts[tid]
        f_cnt = final_counts[tid]
        s_cnt = semifinal_counts[tid]
        q_cnt = quarterfinal_counts[tid]

        p_champ = c_cnt / n_sims
        p_final = f_cnt / n_sims
        p_semi  = s_cnt / n_sims
        p_qf    = q_cnt / n_sims

        ci_lo, ci_hi = _wilson_ci(p_champ, n_sims)

        rows.append({
            "id":              tid,
            "name":            row["name"],
            "flag":            row["flag"],
            "group":           row["group"],
            "overall_score":   row["overall_score"],
            "champion_prob":   round(p_champ * 100, 2),
            "final_prob":      round(p_final * 100, 2),
            "semi_prob":       round(p_semi  * 100, 2),
            "quarter_prob":    round(p_qf    * 100, 2),
            "ci_lower":        round(ci_lo   * 100, 2),
            "ci_upper":        round(ci_hi   * 100, 2),
            "champion_count":  c_cnt,
        })

    probs_df = pd.DataFrame(rows).sort_values("champion_prob", ascending=False).reset_index(drop=True)
    probs_df["rank"] = range(1, len(probs_df) + 1)

    return {
        "champion_counts":    dict(champion_counts),
        "final_counts":       dict(final_counts),
        "semifinal_counts":   dict(semifinal_counts),
        "quarterfinal_counts":dict(quarterfinal_counts),
        "n_sims":             n_sims,
        "probs_df":           probs_df,
    }


def get_champion_prediction(scored_df: pd.DataFrame, mc_results: Dict) -> Dict:
    """
    基于蒙特卡洛结果，返回最终冠军预测及完整推理依据。
    """
    probs_df = mc_results["probs_df"]
    champion_row = probs_df.iloc[0]

    # 跑一次确定性预测获取完整赛程路径
    from models.group_stage import simulate_all_groups, get_group_qualifiers
    group_results = simulate_all_groups(scored_df, mode="deterministic")
    top2, best_thirds = get_group_qualifiers(group_results)
    bracket = build_bracket_r32(top2, best_thirds, scored_df)
    knockout = simulate_full_knockout(bracket, scored_df, mode="deterministic")

    # 提取冠军在淘汰赛中的对手路径
    champion_id = knockout.get("champion", {}).get("id", "")
    path = []
    round_labels = [
        ("r32", "32强"),
        ("r16", "16强"),
        ("qf",  "四强"),
        ("sf",  "半决赛"),
        ("final","决赛"),
    ]
    for round_key, round_name in round_labels:
        for match in knockout.get(round_key, []):
            if match["winner_id"] == champion_id:
                loser_flag = ""
                if match["team_a"] == champion_id:
                    opponent = match["team_b_name"]
                    score = f"{match['score_a']}-{match['score_b']}"
                else:
                    opponent = match["team_a_name"]
                    score = f"{match['score_b']}-{match['score_a']}"
                decided = {"regular": "正常时间", "extra_time": "加时赛", "penalty": "点球大战"}.get(match["decided_by"], "")
                path.append({"round": round_name, "opponent": opponent, "score": score, "decided_by": decided})

    return {
        "champion_id":      champion_id,
        "champion_name":    champion_row["name"],
        "champion_flag":    champion_row["flag"],
        "champion_prob":    champion_row["champion_prob"],
        "ci_lower":         champion_row["ci_lower"],
        "ci_upper":         champion_row["ci_upper"],
        "overall_score":    champion_row["overall_score"],
        "path":             path,
        "knockout_full":    knockout,
        "group_results":    {g: t.to_dict("records") for g, t in group_results.items()},
        "top10_probs":      probs_df.head(10).to_dict("records"),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.data_loader import load_teams
    from models.team_rating import compute_overall_scores

    print("加载数据与评分...")
    teams   = load_teams()
    scored  = compute_overall_scores(teams)

    print(f"运行蒙特卡洛模拟 (N=1000, 快速测试)...")
    mc = run_monte_carlo(scored, n_sims=1000, seed=RANDOM_SEED)

    print("\n=== 冠军概率排行榜 TOP10 ===")
    print(mc["probs_df"][["rank","flag","name","champion_prob","final_prob","ci_lower","ci_upper"]]
          .head(10).to_string(index=False))

    pred = get_champion_prediction(scored, mc)
    print(f"\n🏆 预测冠军: {pred['champion_flag']} {pred['champion_name']}")
    print(f"   夺冠概率: {pred['champion_prob']}% (95%CI: [{pred['ci_lower']}%, {pred['ci_upper']}%])")
    print("\n淘汰赛路径:")
    for p in pred["path"]:
        print(f"  {p['round']}: 击败 {p['opponent']} {p['score']} ({p['decided_by']})")
