"""快速验证脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from utils.data_loader import load_teams
from models.team_rating import compute_overall_scores
from models.champion import run_monte_carlo, get_champion_prediction

print("=== 2026世界杯冠军预测Agent - 快速验证 ===\n")

teams = load_teams()
print(f"已加载 {len(teams)} 支球队，{teams['group'].nunique()} 个小组")

scored = compute_overall_scores(teams)
print("\nTOP5综合评分:")
for _, r in scored.sort_values("overall_score", ascending=False).head(5).iterrows():
    print(f"  {r['flag']} {r['name']}: {r['overall_score']:.1f}")

print("\n运行蒙特卡洛模拟 (1000次)...")
mc = run_monte_carlo(scored, n_sims=1000)
probs = mc["probs_df"]

print("\nTOP10夺冠概率:")
for _, r in probs.head(10).iterrows():
    print(f"  {int(r['rank']):2d}. {r['flag']} {r['name']:<12} {r['champion_prob']:5.1f}%")

pred = get_champion_prediction(scored, mc)
print(f"\n预测冠军: {pred['champion_flag']} {pred['champion_name']} ({pred['champion_prob']:.1f}%)")
print("夺冠路径:")
for p in pred["path"]:
    print(f"  {p['round']}: 击败 {p['opponent']} {p['score']} ({p['decided_by']})")

print("\n验证通过！可以启动 Streamlit 了。")
