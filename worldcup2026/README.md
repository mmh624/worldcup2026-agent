# ⚽ 2026世界杯冠军预测 Agent

基于四层可解释预测架构的2026年美加墨世界杯冠军预测系统。

## 预测结果

**预测冠军：🇦🇷 阿根廷**  
决赛对阵：🇧🇷 巴西  
夺冠概率：蒙特卡洛10,000次模拟计算

---

## 快速启动

### 1. 安装依赖

```bash
cd worldcup2026
pip install -r requirements.txt
```

### 2. 启动可视化界面

```bash
python -m streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`

### 3. 启动 API 服务（可选）

```bash
python backend/main.py
```

API 文档：`http://localhost:8000/docs`

---

## 配置 Qwen API（可选）

设置环境变量后，推理分析将由通义千问生成（否则自动使用规则模板）：

```bash
# Windows
set DASHSCOPE_API_KEY=你的API_KEY

# 或在应用侧边栏 ⚙️ 设置中直接填入
```

API Key 申请：https://dashscope.aliyuncs.com

---

## 功能页面

| 页面 | 内容 |
|------|------|
| 🏠 首页总览 | 冠军预测速览、夺冠概率TOP10、架构说明 |
| 📊 球队实力榜 | 五维雷达图、攻防气泡图、48队完整评分表 |
| 🏟️ 小组赛预测 | 12组积分表、72场比分预测、胜平负概率 |
| 🏆 淘汰赛对阵 | Plotly赛程树、32强到决赛逐轮对阵 |
| 🥇 冠军预测 | 蒙特卡洛概率、仪表盘、Qwen推理报告下载 |
| 🔍 单场预测 | 任意两队对战预测、比分热力图、实时Qwen分析 |

---

## 预测模型架构

```
Layer 0: 五维度球队评分
  历史底蕴(20%) + FIFA排名(30%) + 攻防效率(20%) + 球员班底(15%) + 近期状态(15%)
        ↓
Layer 1: 泊松单场预测
  λ = 1.32 × 攻击力 × (1/防守力) × 东道主加成
  → 独立泊松分布计算胜平负概率 + 最优比分
        ↓
Layer 2: 赛程推演
  小组赛(12组×6场) → 积分榜 → 32强出线
  淘汰赛: 90分钟 → 加时赛(λ×0.33) → 点球(55%)
        ↓
Layer 3: 蒙特卡洛聚合
  N=10,000次模拟 → 夺冠概率 + Wilson 95% CI
  固定随机种子 42 保证结果可复现
```

---

## API 接口

```
GET /api/teams              # 48支球队信息
GET /api/groups             # 12组小组赛积分榜
GET /api/knockout           # 淘汰赛各轮对阵结果
GET /api/champion           # 冠军预测 + 推理路径
GET /api/probs?top=20       # 夺冠概率排行榜
GET /api/match?team_a=ARG&team_b=FRA  # 单场预测
```

---

## 项目结构

```
worldcup2026/
├── streamlit_app.py        # Streamlit 主入口
├── requirements.txt        # 依赖清单
├── verify_quick.py         # 快速验证（无蒙特卡洛）
├── verify.py               # 完整验证（含蒙特卡洛）
├── config/
│   └── settings.py         # 全局配置（权重、API Key等）
├── data/
│   └── teams_2026.json     # 2026世界杯真实分组数据（48队12组）
├── utils/
│   └── data_loader.py      # 数据加载工具
├── models/
│   ├── team_rating.py      # Layer 0: 五维度评分
│   ├── group_stage.py      # Layer 1+2: 泊松预测+小组赛
│   ├── knockout.py         # Layer 2: 淘汰赛三阶段
│   └── champion.py         # Layer 3: 蒙特卡洛聚合
├── backend/
│   ├── main.py             # FastAPI 服务
│   └── services/
│       └── qwen_service.py # Qwen API 推理（支持降级模板）
└── app/
    ├── home.py             # 首页
    ├── analysis.py         # 球队实力分析
    ├── groups.py           # 小组赛
    ├── bracket.py          # 淘汰赛对阵树
    ├── champion_page.py    # 冠军预测
    └── match_detail.py     # 单场预测
```

---

## 球队ID参考（常用）

| ID | 球队 | 小组 |
|----|------|------|
| ARG | 🇦🇷 阿根廷 | J |
| FRA | 🇫🇷 法国 | I |
| BRA | 🇧🇷 巴西 | C |
| ESP | 🇪🇸 西班牙 | H |
| ENG | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 英格兰 | L |
| GER | 🇩🇪 德国 | E |
| POR | 🇵🇹 葡萄牙 | K |
| NED | 🇳🇱 荷兰 | F |
| BEL | 🇧🇪 比利时 | G |
| ITA | 🇮🇹 意大利 | B |

---

## 技术栈

- **Python 3.10+** | Streamlit | FastAPI | Pandas | NumPy | Plotly
- **AI推理**：通义千问 Qwen API（dashscope）
- **算法**：泊松分布 · 蒙特卡洛模拟 · Wilson置信区间
