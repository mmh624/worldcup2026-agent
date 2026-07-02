"""
Qwen API 推理分析模块
─────────────────────────────────────────────────────────
调用通义千问（dashscope）生成：
  - 两队对战分析（200-300字）
  - 冠军推理报告（500字）
  - 小组形势分析
  
无API_KEY时自动降级为规则模板推理（保证离线可用）
─────────────────────────────────────────────────────────
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import QWEN_API_KEY, QWEN_MODEL, QWEN_MAX_TOKENS
from typing import Dict, Optional


def _call_qwen(prompt: str, system_msg: str = "你是专业足球分析师，擅长世界杯预测分析。") -> str:
    """
    调用Qwen API，失败时返回None。
    """
    if not QWEN_API_KEY:
        return None
    try:
        import dashscope
        from dashscope import Generation
        dashscope.api_key = QWEN_API_KEY
        response = Generation.call(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=QWEN_MAX_TOKENS,
            result_format="message",
        )
        if response.status_code == 200:
            return response.output.choices[0]["message"]["content"]
        return None
    except Exception as e:
        return None


def _fallback_match_analysis(team_a: dict, team_b: dict, match_result: dict) -> str:
    """
    离线降级：基于规则生成对战分析文本（无需API）。
    """
    sa = team_a.get("overall_score", 50)
    sb = team_b.get("overall_score", 50)
    diff = abs(sa - sb)
    stronger = team_a if sa >= sb else team_b
    weaker   = team_b if sa >= sb else team_a

    if diff > 20:
        advantage_desc = f"{stronger['name']}在综合实力上占据明显优势（评分差：{diff:.1f}分）"
    elif diff > 8:
        advantage_desc = f"{stronger['name']}略占上风（评分差：{diff:.1f}分）"
    else:
        advantage_desc = f"双方实力相当（评分差仅{diff:.1f}分），胜负难料"

    history_desc = ""
    if stronger.get("world_cup_wins", 0) > 0:
        history_desc = f"{stronger['name']}拥有{stronger['world_cup_wins']}次世界杯冠军经验，大赛心理优势明显。"

    decided_map = {"regular": "正常时间", "extra_time": "加时赛", "penalty": "点球大战"}
    decided_cn  = decided_map.get(match_result.get("decided_by", "regular"), "")

    winner = match_result.get("winner_name", "")
    score_a = match_result.get("score_a", 0)
    score_b = match_result.get("score_b", 0)

    attack_a = team_a.get("attack_score", 70)
    attack_b = team_b.get("attack_score", 70)
    def_a    = team_a.get("defense_score", 70)
    def_b    = team_b.get("defense_score", 70)

    analysis = f"""**{team_a['flag']} {team_a['name']} vs {team_b['flag']} {team_b['name']}**

**实力对比：** {advantage_desc}。{history_desc}

**进攻端：** {team_a['name']}进攻评分{attack_a}，{team_b['name']}进攻评分{attack_b}；
**防守端：** {team_a['name']}防守评分{def_a}，{team_b['name']}防守评分{def_b}。

**预测结果：** 本场比赛预测比分为 {score_a}-{score_b}，{winner}最终{decided_cn}获胜。

**模型推理：** 基于泊松分布预测，结合五维度球队评分（历史底蕴20%+FIFA排名30%+攻防效率20%+球员班底15%+近期状态15%），综合计算两队期望进球数λ，通过独立泊松分布联合概率得出最优预测比分。"""

    return analysis


def _fallback_champion_report(champion_info: dict) -> str:
    """
    离线降级：生成冠军预测报告模板。
    """
    c = champion_info
    path_lines = ""
    for p in c.get("path", []):
        path_lines += f"\n  - **{p['round']}**：击败 {p['opponent']} {p['score']}（{p['decided_by']}）"

    report = f"""# 🏆 2026世界杯冠军预测报告

## 预测结果：{c['champion_flag']} {c['champion_name']}

**夺冠概率：{c['champion_prob']:.1f}%**（Wilson 95% 置信区间：[{c['ci_lower']:.1f}%, {c['ci_upper']:.1f}%]）

**综合实力评分：{c['overall_score']:.1f}/100**

## 核心优势分析

{c['champion_name']}在本届2026年美加墨世界杯中被模型评定为夺冠热门，主要基于以下四点：

1. **历史底蕴**：{c['champion_name']}拥有丰富的世界杯大赛经验，历史战绩出色，在淘汰赛阶段心理优势显著。

2. **FIFA排名支撑**：当前FIFA世界排名领先，反映其近期持续高水平竞技状态。

3. **攻防均衡**：球队在攻击端和防守端均保持高水准，五维评分综合最优。

4. **球员阵容**：现役主力球员正值当打之年，阵容深度具备完整赛程竞争力。

## 淘汰赛推演路径

以下为模型确定性预测的夺冠路径：{path_lines}

## 不确定性说明

蒙特卡洛模拟（N=10,000次）显示，世界杯存在较高随机性，即使最热门球队夺冠概率通常也在15%-30%之间。本预测基于数据模型，实际结果受伤病、临场发挥等多重因素影响。

---
*本报告由四层可解释预测模型生成：五维球队评分 → 泊松比分预测 → 三阶段淘汰赛模拟 → 蒙特卡洛概率聚合*
"""
    return report


def analyze_match(team_a: dict, team_b: dict, match_result: dict) -> str:
    """
    生成单场比赛的Qwen推理分析（优先调用API，降级用模板）。
    """
    if QWEN_API_KEY:
        prompt = f"""请分析以下2026年FIFA世界杯比赛预测：

球队A：{team_a['flag']} {team_a['name']}（FIFA排名#{team_a.get('fifa_rank','?')}，综合评分{team_a.get('overall_score',50):.1f}）
球队B：{team_b['flag']} {team_b['name']}（FIFA排名#{team_b.get('fifa_rank','?')}，综合评分{team_b.get('overall_score',50):.1f}）

关键数据对比：
- {team_a['name']} 进攻={team_a.get('attack_score',70)} 防守={team_a.get('defense_score',70)} 世界杯冠军={team_a.get('world_cup_wins',0)}次
- {team_b['name']} 进攻={team_b.get('attack_score',70)} 防守={team_b.get('defense_score',70)} 世界杯冠军={team_b.get('world_cup_wins',0)}次

模型预测结果：{match_result.get('winner_name','?')}获胜，比分{match_result.get('score_a',0)}-{match_result.get('score_b',0)}，通过{match_result.get('decided_by','正常时间')}决出。

请用200-250字的中文，从双方技战术、历史经验、关键球员三个角度分析这场比赛的预测结果，语言要专业且易于理解。"""

        result = _call_qwen(prompt)
        if result:
            return result

    return _fallback_match_analysis(team_a, team_b, match_result)


def generate_champion_report(champion_info: dict, scored_df=None) -> str:
    """
    生成冠军预测完整报告（优先Qwen，降级模板）。
    """
    if QWEN_API_KEY:
        path_str = "、".join([f"{p['round']}击败{p['opponent']}" for p in champion_info.get("path", [])])
        top5 = champion_info.get("top10_probs", [])[:5]
        top5_str = "、".join([f"{t['flag']}{t['name']}({t['champion_prob']}%)" for t in top5])

        prompt = f"""请作为专业足球分析师，为2026年美加墨世界杯生成冠军预测报告。

预测结果：
- 冠军预测：{champion_info['champion_flag']} {champion_info['champion_name']}
- 夺冠概率：{champion_info['champion_prob']:.1f}%（10,000次蒙特卡洛模拟）
- 95%置信区间：[{champion_info['ci_lower']:.1f}%, {champion_info['ci_upper']:.1f}%]
- 综合实力评分：{champion_info['overall_score']:.1f}/100

夺冠路径：{path_str}

其他热门球队：{top5_str}

请生成一份400-500字的专业分析报告，包含：
1. 为何{champion_info['champion_name']}最有可能夺冠（3个核心论点）
2. 主要竞争对手威胁分析
3. 潜在风险与不确定因素
4. 总结展望

用中文，语言要专业、有数据支撑、适合答辩展示。"""

        result = _call_qwen(prompt)
        if result:
            return result

    return _fallback_champion_report(champion_info)


def analyze_group(group_label: str, teams_in_group: list) -> str:
    """
    生成小组形势分析。
    """
    team_lines = "\n".join([
        f"  - {t['flag']} {t['name']}（FIFA#{t.get('fifa_rank','?')}，综合评分{t.get('overall_score',50):.1f}）"
        for t in teams_in_group
    ])

    if QWEN_API_KEY:
        prompt = f"""分析2026年世界杯{group_label}组形势：

{group_label}组球队：
{team_lines}

请用150字分析该组的出线形势，指出最可能出线的两支球队及其理由，语言简洁专业。"""
        result = _call_qwen(prompt)
        if result:
            return result

    # 降级：按评分排序输出简单分析
    sorted_teams = sorted(teams_in_group, key=lambda x: x.get("overall_score", 0), reverse=True)
    if len(sorted_teams) >= 2:
        t1, t2 = sorted_teams[0], sorted_teams[1]
        return (f"{group_label}组预测：{t1['flag']}{t1['name']}（综合评分{t1.get('overall_score',50):.1f}）"
                f"和{t2['flag']}{t2['name']}（{t2.get('overall_score',50):.1f}）"
                f"最有可能出线，两队在FIFA排名和近期状态上均领先组内其他球队。")
    return f"{group_label}组形势分析待更新。"
