"""
数据加载工具 — 从JSON文件读取球队信息，构建Pandas DataFrame
"""
import json
import os
import pandas as pd
from typing import Dict, List

# 允许从模块任意位置导入
_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, "..", "data")


def load_teams() -> pd.DataFrame:
    """
    加载48支球队信息，返回DataFrame。

    字段说明：
        id, name, name_en, flag, fifa_rank, group, confederation,
        world_cup_wins, world_cup_finals, world_cup_semis,
        attack_score, defense_score, player_strength, recent_form, home_boost
    """
    path = os.path.join(DATA_DIR, "teams_2026.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["teams"])
    return df


def get_teams_by_group(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """按小组分组，返回 {group_label: DataFrame}"""
    return {g: sub.reset_index(drop=True) for g, sub in df.groupby("group")}


def get_team_by_id(df: pd.DataFrame, team_id: str) -> Dict:
    """根据id取单支球队信息，返回dict"""
    row = df[df["id"] == team_id]
    if row.empty:
        raise ValueError(f"Team not found: {team_id}")
    return row.iloc[0].to_dict()


def team_display(t: Dict) -> str:
    """返回 '🇦🇷 阿根廷' 格式的显示字符串"""
    return f"{t['flag']} {t['name']}"
