"""快速验证脚本 - 跳过蒙特卡洛"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from utils.data_loader import load_teams
from models.team_rating import compute_overall_scores
from models.group_stage import simulate_all_groups, get_group_qualifiers
from models.knockout import build_bracket_r32, simulate_full_knockout

print("=== 快速验证 (无蒙特卡洛) ===\n")

teams = load_teams()
print(f"球队: {len(teams)}, 小组: {teams['group'].nunique()}")

scored = compute_overall_scores(teams)
top5 = scored.sort_values("overall_score", ascending=False).head(5)
print("TOP5:", [(r['name'], round(r['overall_score'],1)) for _, r in top5.iterrows()])

groups = simulate_all_groups(scored)
top2, thirds = get_group_qualifiers(groups)
print(f"出线: {len(top2)} + 最佳第三: {len(thirds)}")

bracket = build_bracket_r32(top2, thirds, scored)
knockout = simulate_full_knockout(bracket, scored)
champion = knockout.get('champion', {})
final = knockout.get('final', [{}])[0]

print(f"决赛: {final.get('team_a_name')} {final.get('score_a')}-{final.get('score_b')} {final.get('team_b_name')}")
print(f"冠军: {champion.get('flag')} {champion.get('name')}")
print("OK!")
