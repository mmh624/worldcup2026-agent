"""
Layer 0 — 五维度球队综合评分模型
─────────────────────────────────────────────────────────
维度        权重    说明
历史底蕴    20%     世界杯冠军/亚军/四强次数加权
FIFA排名    30%     排名倒序归一化 + ELO加权
攻防效率    20%     (attack_score + defense_score) / 2 归一
球员班底    15%     player_strength 归一
近期状态    15%     recent_form 归一
─────────────────────────────────────────────────────────
最终输出: overall_score ∈ [0, 100]，以及各维度分数（白盒可解释）
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import WEIGHTS

# 历史战绩权重系数（越近期冠军权重越高）
HISTORY_WEIGHTS = {
    "wins":   10,   # 世界杯冠军次数
    "finals": 5,    # 世界杯亚军/决赛次数
    "semis":  2,    # 世界杯四强次数
}
MAX_HISTORY_SCORE = (5 * HISTORY_WEIGHTS["wins"] +
                     7 * HISTORY_WEIGHTS["finals"] +
                     13 * HISTORY_WEIGHTS["semis"])  # 德国是上限参考


def _normalize(series: pd.Series, vmin: float = None, vmax: float = None) -> pd.Series:
    """Min-Max归一化到 [0, 100]"""
    lo = series.min() if vmin is None else vmin
    hi = series.max() if vmax is None else vmax
    if hi == lo:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo) * 100


def compute_history_score(df: pd.DataFrame) -> pd.Series:
    """
    计算历史底蕴得分（0-100）
    公式: wins×10 + finals×5 + semis×2，再归一化
    """
    raw = (df["world_cup_wins"] * HISTORY_WEIGHTS["wins"] +
           df["world_cup_finals"] * HISTORY_WEIGHTS["finals"] +
           df["world_cup_semis"] * HISTORY_WEIGHTS["semis"])
    return _normalize(raw, vmin=0, vmax=MAX_HISTORY_SCORE)


def compute_fifa_score(df: pd.DataFrame) -> pd.Series:
    """
    FIFA排名得分（排名越小越好 → 倒序归一化）
    """
    # 排名越高（数字越大）得分越低
    inv_rank = 1.0 / df["fifa_rank"].astype(float)
    return _normalize(inv_rank)


def compute_attack_defense_score(df: pd.DataFrame) -> pd.Series:
    """
    攻防综合得分 = (attack_score + defense_score) / 2，归一化
    """
    raw = (df["attack_score"] + df["defense_score"]) / 2.0
    return _normalize(raw)


def compute_player_score(df: pd.DataFrame) -> pd.Series:
    """球员班底得分"""
    return _normalize(df["player_strength"].astype(float))


def compute_form_score(df: pd.DataFrame) -> pd.Series:
    """近期状态得分"""
    return _normalize(df["recent_form"].astype(float))


def compute_overall_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有球队的五维度得分 + 综合评分。

    返回原始df扩展版，新增列：
        score_history, score_fifa, score_ad, score_player, score_form,
        overall_score, attack_lambda（泊松参数用）, defense_factor
    """
    result = df.copy()

    result["score_history"] = compute_history_score(df)
    result["score_fifa"]    = compute_fifa_score(df)
    result["score_ad"]      = compute_attack_defense_score(df)
    result["score_player"]  = compute_player_score(df)
    result["score_form"]    = compute_form_score(df)

    w = WEIGHTS
    result["overall_score"] = (
        result["score_history"] * w["history"] +
        result["score_fifa"]    * w["fifa_rank"] +
        result["score_ad"]      * w["attack_defense"] +
        result["score_player"]  * w["player"] +
        result["score_form"]    * w["form"]
    ).round(2)

    # 攻击力/防御力归一化到 [0.5, 1.5]，供泊松λ使用
    attack_norm  = _normalize(df["attack_score"].astype(float))
    defense_norm = _normalize(df["defense_score"].astype(float))
    result["attack_lambda"]   = (attack_norm  / 100 + 0.5).round(4)   # [0.5, 1.5]
    result["defense_factor"]  = (defense_norm / 100 + 0.5).round(4)   # [0.5, 1.5]

    return result


def get_score_breakdown(row: pd.Series) -> dict:
    """
    返回单支球队的可解释评分明细，供前端推理面板展示。
    """
    return {
        "历史底蕴 (20%)":  round(row["score_history"], 1),
        "FIFA排名 (30%)":  round(row["score_fifa"], 1),
        "攻防效率 (20%)":  round(row["score_ad"], 1),
        "球员班底 (15%)":  round(row["score_player"], 1),
        "近期状态 (15%)":  round(row["score_form"], 1),
        "综合评分":        round(row["overall_score"], 1),
    }


if __name__ == "__main__":
    # 快速自测
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.data_loader import load_teams
    teams = load_teams()
    scored = compute_overall_scores(teams)
    print(scored[["name", "score_history", "score_fifa", "score_ad",
                  "score_player", "score_form", "overall_score"]]
          .sort_values("overall_score", ascending=False)
          .head(10).to_string(index=False))
