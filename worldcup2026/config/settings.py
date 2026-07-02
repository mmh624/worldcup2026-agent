"""
全局配置文件 — 2026世界杯冠军预测Agent
"""
import os

# ── 路径配置 ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── 随机种子（固定保证可复现） ──────────────────────────
RANDOM_SEED = 42

# ── 蒙特卡洛模拟次数 ──────────────────────────────────
MONTE_CARLO_N = 10000

# ── 泊松分布截断值（计算P(X=k)时的最大k） ──────────────
POISSON_K_MAX = 10

# ── 全局联赛平均进球数（用于泊松λ基准） ─────────────────
LEAGUE_AVG_GOALS = 1.32

# ── 五维度评分权重 ────────────────────────────────────
WEIGHTS = {
    "history":       0.20,   # 历史底蕴（世界杯成绩）
    "fifa_rank":     0.30,   # FIFA排名实力
    "attack_defense":0.20,   # 攻防效率
    "player":        0.15,   # 球员班底
    "form":          0.15,   # 近期状态
}

# ── 东道主加成 ────────────────────────────────────────
HOME_BOOST_FACTOR = 1.12   # 东道主进攻λ × 1.12

# ── 淘汰赛加时赛进球衰减系数 ──────────────────────────
EXTRA_TIME_FACTOR = 0.33   # 加时赛进球概率 × 0.33

# ── 点球大战胜率（强队 vs 弱队会有微调） ─────────────────
PENALTY_WIN_PROB_BASE = 0.55   # 略强一方点球胜率

# ── Qwen API配置 ──────────────────────────────────────
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_MODEL   = "qwen-turbo"    # qwen-turbo / qwen-max
QWEN_MAX_TOKENS = 600

# ── 2026世界杯晋级规则 ────────────────────────────────
# 12组，每组4队，每组前2名 + 8支最佳第三名共32强进入淘汰赛
GROUPS = list("ABCDEFGHIJKL")
TEAMS_PER_GROUP = 4
QUALIFY_TOP_N = 2            # 每组直接出线
BEST_THIRD_COUNT = 8         # 最佳第三名出线数

# ── FastAPI服务配置 ───────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
