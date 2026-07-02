"""
FastAPI 后端服务 — 2026世界杯冠军预测API
启动：uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(
    title="2026世界杯冠军预测 API",
    description="四层可解释预测：五维评分→泊松→淘汰赛→蒙特卡洛",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局数据缓存
_cache = {}


def get_data():
    """懒加载预测数据"""
    global _cache
    if not _cache:
        from utils.data_loader import load_teams
        from models.team_rating import compute_overall_scores
        from models.group_stage import simulate_all_groups, get_group_qualifiers
        from models.knockout import build_bracket_r32, simulate_full_knockout
        from models.champion import run_monte_carlo, get_champion_prediction

        teams  = load_teams()
        scored = compute_overall_scores(teams)
        groups = simulate_all_groups(scored)
        top2, thirds = get_group_qualifiers(groups)
        bracket = build_bracket_r32(top2, thirds, scored)
        knockout = simulate_full_knockout(bracket, scored)
        mc = run_monte_carlo(scored, n_sims=5000)
        pred = get_champion_prediction(scored, mc)

        _cache = {
            "teams": teams,
            "scored": scored,
            "groups": groups,
            "knockout": knockout,
            "mc": mc,
            "pred": pred,
        }
    return _cache


@app.get("/")
def root():
    return {"message": "2026世界杯冠军预测 API", "docs": "/docs"}


@app.get("/api/teams")
def get_teams():
    """获取全部48支球队信息"""
    d = get_data()
    return d["scored"].to_dict("records")


@app.get("/api/groups")
def get_groups():
    """获取12组小组赛预测积分榜"""
    d = get_data()
    return {g: t.to_dict("records") for g, t in d["groups"].items()}


@app.get("/api/knockout")
def get_knockout():
    """获取完整淘汰赛对阵结果"""
    d = get_data()
    ko = d["knockout"]
    result = {}
    for k in ["r32", "r16", "qf", "sf", "final"]:
        matches = ko.get(k, [])
        result[k] = [
            {kk: v for kk, v in m.items() if not isinstance(v, float) or not __import__('math').isnan(v)}
            for m in matches
        ]
    result["champion"] = {
        k: v for k, v in ko.get("champion", {}).items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
    return result


@app.get("/api/champion")
def get_champion():
    """获取冠军预测 + 蒙特卡洛概率 + 推理路径"""
    d = get_data()
    pred = d["pred"].copy()
    # 清理嵌套对象
    pred.pop("knockout_full", None)
    pred.pop("group_results", None)
    return pred


@app.get("/api/probs")
def get_probs(top: int = 20):
    """获取全部球队夺冠概率排行"""
    d = get_data()
    return d["mc"]["probs_df"].head(top).to_dict("records")


@app.get("/api/match")
def predict_match(team_a: str, team_b: str):
    """
    预测两队对战结果
    参数: team_a, team_b (球队ID，如 ARG, FRA)
    """
    d = get_data()
    scored = d["scored"]
    try:
        from utils.data_loader import get_team_by_id
        from models.group_stage import compute_lambdas, predict_score, win_draw_loss_probs
        ta = get_team_by_id(scored, team_a.upper())
        tb = get_team_by_id(scored, team_b.upper())
        lam_a, lam_b = compute_lambdas(
            ta["attack_lambda"], ta["defense_factor"],
            tb["attack_lambda"], tb["defense_factor"],
        )
        ga, gb = predict_score(lam_a, lam_b)
        pw, pd, pl = win_draw_loss_probs(lam_a, lam_b)
        return {
            "team_a": ta["name"], "team_a_flag": ta["flag"],
            "team_b": tb["name"], "team_b_flag": tb["flag"],
            "predicted_score": f"{ga}-{gb}",
            "prob_a_win": round(pw, 3),
            "prob_draw": round(pd, 3),
            "prob_b_win": round(pl, 3),
            "lambda_a": lam_a, "lambda_b": lam_b,
        }
    except ValueError as e:
        return {"error": str(e)}


@app.get("/api/reasoning/{team_id}")
def get_reasoning(team_id: str):
    """获取某支球队的夺冠推理分析（Qwen生成）"""
    d = get_data()
    pred = d["pred"]
    if pred.get("champion_id") == team_id.upper():
        from backend.services.qwen_service import generate_champion_report
        report = generate_champion_report(pred)
        return {"team_id": team_id, "report": report}
    return {"team_id": team_id, "message": "该球队非预测冠军，请查看 /api/champion"}


if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
