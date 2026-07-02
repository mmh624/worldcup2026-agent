"""
Layer 2 — 淘汰赛三阶段制胜模型
─────────────────────────────────────────────────────────
阶段1: 90分钟正常时间（泊松比分）
阶段2: 平局→加时赛（λ × EXTRA_TIME_FACTOR）
阶段3: 加时仍平→点球大战（强队55%胜率为基础）
─────────────────────────────────────────────────────────
"""
import math
import random
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    EXTRA_TIME_FACTOR, PENALTY_WIN_PROB_BASE, RANDOM_SEED, POISSON_K_MAX
)
from models.group_stage import compute_lambdas, predict_score, sample_score, _poisson_pmf


def penalty_win_prob(overall_a: float, overall_b: float) -> float:
    """
    点球大战：根据综合实力差距微调胜率。
    基础55%，差距每10分调整2%。
    """
    diff = overall_a - overall_b
    p = PENALTY_WIN_PROB_BASE + (diff / 10.0) * 0.02
    return max(0.35, min(0.65, p))   # 约束在[35%, 65%]


def simulate_knockout_match(
    team_a: dict,
    team_b: dict,
    mode: str = "deterministic",
    rng: Optional[random.Random] = None,
) -> dict:
    """
    模拟一场淘汰赛。

    mode:
      'deterministic' — 概率最高路径（预测比分）
      'sample'        — 蒙特卡洛随机采样

    返回dict:
        winner_id, loser_id, winner_name, winner_flag,
        score_a, score_b, decided_by ('regular'/'extra_time'/'penalty'),
        prob_a_win (全程胜率)
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    a_home = team_a.get("home_boost", False)
    b_home = team_b.get("home_boost", False)

    lam_a, lam_b = compute_lambdas(
        team_a["attack_lambda"], team_a["defense_factor"],
        team_b["attack_lambda"], team_b["defense_factor"],
        a_is_home=a_home, b_is_home=b_home,
    )

    # ── 阶段1: 90分钟 ────────────────────────────────
    if mode == "sample":
        ga, gb = sample_score(lam_a, lam_b, rng)
    else:
        ga, gb = predict_score(lam_a, lam_b)

    decided_by = "regular"

    if ga != gb:
        winner_id = team_a["id"] if ga > gb else team_b["id"]
    else:
        # ── 阶段2: 加时赛 ────────────────────────────
        et_lam_a = lam_a * EXTRA_TIME_FACTOR
        et_lam_b = lam_b * EXTRA_TIME_FACTOR

        if mode == "sample":
            et_a, et_b = sample_score(et_lam_a, et_lam_b, rng)
        else:
            et_a, et_b = predict_score(et_lam_a, et_lam_b)

        ga += et_a
        gb += et_b

        if ga != gb:
            winner_id = team_a["id"] if ga > gb else team_b["id"]
            decided_by = "extra_time"
        else:
            # ── 阶段3: 点球大战 ──────────────────────
            decided_by = "penalty"
            p_a = penalty_win_prob(
                team_a.get("overall_score", 50),
                team_b.get("overall_score", 50),
            )
            if mode == "sample":
                winner_id = team_a["id"] if rng.random() < p_a else team_b["id"]
            else:
                # deterministic: 综合实力强的胜
                winner_id = team_a["id"] if team_a.get("overall_score", 50) >= team_b.get("overall_score", 50) else team_b["id"]

    winner = team_a if winner_id == team_a["id"] else team_b
    loser  = team_b if winner_id == team_a["id"] else team_a

    return {
        "team_a":       team_a["id"],
        "team_a_name":  team_a["name"],
        "team_a_flag":  team_a["flag"],
        "team_b":       team_b["id"],
        "team_b_name":  team_b["name"],
        "team_b_flag":  team_b["flag"],
        "score_a":      ga,
        "score_b":      gb,
        "decided_by":   decided_by,
        "winner_id":    winner["id"],
        "winner_name":  winner["name"],
        "winner_flag":  winner["flag"],
        "loser_id":     loser["id"],
        "loser_name":   loser["name"],
    }


def build_bracket_r32(top2: List[dict], best_thirds: List[dict], scored_df: pd.DataFrame) -> List[Tuple[dict, dict]]:
    """
    构建32强对阵（简化规则：相邻组1st vs 相邻组2nd交叉）。
    实际2026赛制会更复杂，此处用简化对阵方便展示。

    返回: [(team_a, team_b), ...] 共16场对阵
    """
    from utils.data_loader import get_team_by_id
    all_qualifiers = top2 + best_thirds
    all_qualifiers_sorted = sorted(all_qualifiers, key=lambda x: (x["group"], x["pos"]))

    # 组织16组对阵：A1 vs B2, B1 vs A2, C1 vs D2, ...
    groups_order = list("ABCDEFGHIJKL")
    matchups = []
    id_to_row = {row["id"]: row for _, row in scored_df.iterrows()}

    # 直接出线32队：12组×2 = 24 + 8最佳第三
    group_firsts  = {q["group"]: q for q in top2 if q["pos"] == 1}
    group_seconds = {q["group"]: q for q in top2 if q["pos"] == 2}
    thirds_list   = list(best_thirds)

    # 简化对阵：按组顺序两两交叉
    pairs = [
        ("A", 1, "B", 2), ("B", 1, "A", 2),
        ("C", 1, "D", 2), ("D", 1, "C", 2),
        ("E", 1, "F", 2), ("F", 1, "E", 2),
        ("G", 1, "H", 2), ("H", 1, "G", 2),
        ("I", 1, "J", 2), ("J", 1, "I", 2),
        ("K", 1, "L", 2), ("L", 1, "K", 2),
    ]

    def _get_team(g, pos):
        if pos == 1:
            return group_firsts.get(g)
        else:
            return group_seconds.get(g)

    for g1, p1, g2, p2 in pairs:
        ta = _get_team(g1, p1)
        tb = _get_team(g2, p2)
        if ta and tb:
            ta_full = {**id_to_row.get(ta["id"], {}), **ta}
            tb_full = {**id_to_row.get(tb["id"], {}), **tb}
            matchups.append((ta_full, tb_full))

    # 补充最佳第三名的对阵（取剩余第三名两两对战）
    # 只补到16组
    while len(matchups) < 16 and len(thirds_list) >= 2:
        ta = thirds_list.pop(0)
        tb = thirds_list.pop(0)
        ta_full = {**id_to_row.get(ta["id"], {}), **ta}
        tb_full = {**id_to_row.get(tb["id"], {}), **tb}
        matchups.append((ta_full, tb_full))

    return matchups[:16]


def simulate_full_knockout(
    bracket: List[Tuple[dict, dict]],
    scored_df: pd.DataFrame,
    mode: str = "deterministic",
    rng: Optional[random.Random] = None,
) -> Dict:
    """
    完整模拟淘汰赛各轮：32强→16强→8强→4强→决赛→冠军。

    返回完整的淘汰赛结构dict:
    {
      "r32": [match_result, ...],
      "r16": [...],
      "qf":  [...],
      "sf":  [...],
      "final": match_result,
      "champion": team_dict,
      "bracket_tree": [...],   # 嵌套赛程树（前端可视化用）
    }
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    id_to_row = {row["id"]: row.to_dict() for _, row in scored_df.iterrows()}
    rounds_map = {16: "r32", 8: "r16", 4: "qf", 2: "sf", 1: "final"}
    all_rounds = {}
    current_matchups = bracket

    for round_size in [16, 8, 4, 2]:
        round_key = rounds_map[round_size]
        results = []
        next_round = []

        for ta, tb in current_matchups:
            # 补充评分字段（如未存在）
            for t in [ta, tb]:
                if "overall_score" not in t or pd.isna(t.get("overall_score")):
                    base = id_to_row.get(t["id"], {})
                    t.update({k: v for k, v in base.items() if k not in t})

            match_res = simulate_knockout_match(ta, tb, mode=mode, rng=rng)
            results.append(match_res)

            winner_id = match_res["winner_id"]
            winner_base = id_to_row.get(winner_id, {})
            winner_full  = {**winner_base, "winner_flag": match_res["winner_flag"],
                           "winner_name": match_res["winner_name"]}
            next_round.append(winner_full)

        all_rounds[round_key] = results

        # 下一轮配对（顺序两两）
        current_matchups = []
        for i in range(0, len(next_round) - 1, 2):
            current_matchups.append((next_round[i], next_round[i + 1]))

    # ── 决赛 ────────────────────────────────────────
    if current_matchups:
        ta, tb = current_matchups[0]
        for t in [ta, tb]:
            if "overall_score" not in t or (isinstance(t.get("overall_score"), float) and math.isnan(t["overall_score"])):
                base = id_to_row.get(t.get("id", ""), {})
                t.update({k: v for k, v in base.items() if k not in t})

        final_match = simulate_knockout_match(ta, tb, mode=mode, rng=rng)
        all_rounds["final"] = [final_match]
        champion_id = final_match["winner_id"]
        champion = id_to_row.get(champion_id, {"id": champion_id,
                                                "name": final_match["winner_name"],
                                                "flag": final_match["winner_flag"]})
        all_rounds["champion"] = champion

    return all_rounds
