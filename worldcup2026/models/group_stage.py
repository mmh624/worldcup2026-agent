"""
Layer 1 — 泊松单场比赛预测
Layer 2 — 小组赛积分榜模拟
─────────────────────────────────────────────────────────
泊松模型核心公式：
  λ_home = AVG × attack_home × (1/defense_away) × home_boost
  λ_away = AVG × attack_away × (1/defense_home)

  P(进X球) = λ^X * e^(-λ) / X!
  P(胜) = ΣΣ P(i>j)，P(平) = ΣΣ P(i=j)，P(负) = ΣΣ P(i<j)
─────────────────────────────────────────────────────────
"""
import math
import random
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    LEAGUE_AVG_GOALS, POISSON_K_MAX, HOME_BOOST_FACTOR, RANDOM_SEED
)

# ── 基础泊松工具 ──────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) = λ^k * e^(-λ) / k!"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def compute_lambdas(
    atk_a: float, def_a: float,
    atk_b: float, def_b: float,
    a_is_home: bool = False,
    b_is_home: bool = False,
) -> Tuple[float, float]:
    """
    计算A、B两队的期望进球数λ（泊松参数）。

    λ_A = AVG × atk_A × (1/def_B) × [home_boost if A is host]
    λ_B = AVG × atk_B × (1/def_A) × [home_boost if B is host]
    """
    base = LEAGUE_AVG_GOALS
    lam_a = base * atk_a * (1.0 / max(def_b, 0.1))
    lam_b = base * atk_b * (1.0 / max(def_a, 0.1))
    if a_is_home:
        lam_a *= HOME_BOOST_FACTOR
    if b_is_home:
        lam_b *= HOME_BOOST_FACTOR
    return round(lam_a, 4), round(lam_b, 4)


def win_draw_loss_probs(lam_a: float, lam_b: float) -> Tuple[float, float, float]:
    """
    从两个独立泊松分布计算 P(A胜), P(平), P(A负)。
    截断到 POISSON_K_MAX 个进球。
    """
    p_win, p_draw, p_loss = 0.0, 0.0, 0.0
    for i in range(POISSON_K_MAX + 1):
        pi = _poisson_pmf(i, lam_a)
        for j in range(POISSON_K_MAX + 1):
            pj = _poisson_pmf(j, lam_b)
            prob = pi * pj
            if i > j:
                p_win  += prob
            elif i == j:
                p_draw += prob
            else:
                p_loss += prob
    total = p_win + p_draw + p_loss
    if total <= 0:
        return 0.33, 0.34, 0.33
    return p_win / total, p_draw / total, p_loss / total


def predict_score(lam_a: float, lam_b: float) -> Tuple[int, int]:
    """
    用泊松分布的众数（最大概率比分）作为预测比分。
    返回 (goals_a, goals_b)
    """
    best_prob = -1
    best_score = (1, 1)
    for i in range(POISSON_K_MAX + 1):
        for j in range(POISSON_K_MAX + 1):
            p = _poisson_pmf(i, lam_a) * _poisson_pmf(j, lam_b)
            if p > best_prob:
                best_prob = p
                best_score = (i, j)
    return best_score


def sample_score(lam_a: float, lam_b: float, rng: random.Random) -> Tuple[int, int]:
    """
    蒙特卡洛用：从泊松分布随机采样一个比分。
    """
    # 用numpy泊松采样更高效
    goals_a = int(np.random.poisson(lam_a))
    goals_b = int(np.random.poisson(lam_b))
    return goals_a, goals_b


def predict_match(team_a: dict, team_b: dict, a_home: bool = False, b_home: bool = False) -> dict:
    """
    给定两支球队的数据字典，返回完整的比赛预测结果。

    返回dict字段:
        team_a, team_b, lam_a, lam_b,
        prob_win, prob_draw, prob_loss,
        predicted_score_a, predicted_score_b,
        win_pct_a, win_pct_b  (仅供参考)
    """
    lam_a, lam_b = compute_lambdas(
        team_a["attack_lambda"], team_a["defense_factor"],
        team_b["attack_lambda"], team_b["defense_factor"],
        a_is_home=a_home, b_is_home=b_home,
    )
    p_win, p_draw, p_loss = win_draw_loss_probs(lam_a, lam_b)
    score_a, score_b = predict_score(lam_a, lam_b)

    return {
        "team_a":             team_a["id"],
        "team_a_name":        team_a["name"],
        "team_a_flag":        team_a["flag"],
        "team_b":             team_b["id"],
        "team_b_name":        team_b["name"],
        "team_b_flag":        team_b["flag"],
        "lam_a":              lam_a,
        "lam_b":              lam_b,
        "prob_win":           round(p_win,  3),
        "prob_draw":          round(p_draw, 3),
        "prob_loss":          round(p_loss, 3),
        "predicted_score_a":  score_a,
        "predicted_score_b":  score_b,
    }


# ── 小组赛积分榜 ─────────────────────────────────────

def _points(win: bool, draw: bool) -> int:
    return 3 if win else (1 if draw else 0)


def simulate_group(group_teams: List[dict], mode: str = "deterministic") -> pd.DataFrame:
    """
    模拟一组小组赛（6场轮循赛），返回积分榜DataFrame。

    mode:
      'deterministic' — 用预测比分（众数）计算积分
      'sample'        — 从泊松采样比分（蒙特卡洛用）

    积分榜字段：
        id, name, flag, played, wins, draws, losses,
        gf, ga, gd, points
    """
    stats = {t["id"]: {"id": t["id"], "name": t["name"], "flag": t["flag"],
                       "played": 0, "wins": 0, "draws": 0, "losses": 0,
                       "gf": 0, "ga": 0, "gd": 0, "points": 0}
             for t in group_teams}

    n = len(group_teams)
    rng = random.Random(RANDOM_SEED)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = group_teams[i], group_teams[j]
            a_home = a.get("home_boost", False)
            b_home = b.get("home_boost", False)
            lam_a, lam_b = compute_lambdas(
                a["attack_lambda"], a["defense_factor"],
                b["attack_lambda"], b["defense_factor"],
                a_is_home=a_home, b_is_home=b_home,
            )
            if mode == "sample":
                ga, gb = sample_score(lam_a, lam_b, rng)
            else:
                ga, gb = predict_score(lam_a, lam_b)

            # 更新统计
            for tid, gf, ga_opp in [(a["id"], ga, gb), (b["id"], gb, ga)]:
                s = stats[tid]
                s["played"] += 1
                s["gf"] += gf
                s["ga"] += ga_opp
                s["gd"] += (gf - ga_opp)
                if gf > ga_opp:
                    s["wins"] += 1
                    s["points"] += 3
                elif gf == ga_opp:
                    s["draws"] += 1
                    s["points"] += 1
                else:
                    s["losses"] += 1

    table = pd.DataFrame(list(stats.values()))
    table = table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
    table["rank"] = range(1, len(table) + 1)
    return table


def simulate_all_groups(scored_df: pd.DataFrame, mode: str = "deterministic") -> Dict[str, pd.DataFrame]:
    """
    模拟全部12组小组赛，返回 {group: 积分榜DataFrame}
    """
    from utils.data_loader import get_teams_by_group
    groups_dict = get_teams_by_group(scored_df)
    results = {}
    for g, df in groups_dict.items():
        teams_list = df.to_dict("records")
        results[g] = simulate_group(teams_list, mode=mode)
    return results


def get_group_qualifiers(group_results: Dict[str, pd.DataFrame]) -> Tuple[List[dict], List[dict]]:
    """
    根据小组赛结果确定32强出线名单。
    规则：每组前2名直接出线，8支最好的第三名出线。

    返回: (直接出线队伍列表, 最佳第三名出线队伍列表)
    """
    top2 = []
    thirds = []

    for g, table in group_results.items():
        top2.append({"group": g, "pos": 1, **table.iloc[0].to_dict()})
        top2.append({"group": g, "pos": 2, **table.iloc[1].to_dict()})
        if len(table) >= 3:
            thirds.append({"group": g, "pos": 3, **table.iloc[2].to_dict()})

    # 最佳第三名：按积分→净胜球→进球数排序取前8
    thirds_sorted = sorted(thirds, key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True)
    best_thirds = thirds_sorted[:8]

    return top2, best_thirds


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.data_loader import load_teams
    from models.team_rating import compute_overall_scores

    teams = load_teams()
    scored = compute_overall_scores(teams)
    group_results = simulate_all_groups(scored)

    for g, table in sorted(group_results.items()):
        print(f"\n=== 小组 {g} ===")
        print(table[["rank", "flag", "name", "played", "wins", "draws", "losses",
                      "gf", "ga", "gd", "points"]].to_string(index=False))
