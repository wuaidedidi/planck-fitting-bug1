# Planck 黑体辐射光谱拟合与分析引擎

基于 Planck 辐射定律的黑体辐射光谱参数拟合与数据分析系统。

## 项目概述

本项目对三组不同温度的黑体辐射源（白炽灯丝 ~3000K、中温恒星 ~4500K、太阳表面 ~6000K）的光谱实验数据进行参数拟合，通过两阶段优化算法（差分进化 + L-BFGS-B）精确估计各源的有效温度 T 和发射率 ε，并基于 Pandas 执行深度数据分析与可视化。

## 物理模型

**Planck 辐射定律:**

```
B(λ, T) = ε × (2hc² / λ⁵) × 1 / (exp(hc/(λ·k_B·T)) - 1)
```

**Wien 位移定律:**  `λ_max × T = 2.898 × 10⁻³ m·K`

**Stefan-Boltzmann 定律:**  `j = ε × σ × T⁴`

## 技术栈

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 运行环境 |
| NumPy | 1.26.4 | 数值计算 |
| SciPy | 1.13.1 | 优化算法（DE + L-BFGS-B） |
| Matplotlib | 3.9.2 | 图表可视化 |
| Pydantic | 2.9.2 | 数据校验 |
| Pandas | 2.2.3 | 数据分析与统计 |
| pytest | 8.3.3 | 测试框架 |
| Docker | - | 容器化运行 |

## 项目结构

```
planck-fitting/
├── Dockerfile              # Docker 镜像构建
├── docker-compose.yml      # 容器编排
├── requirements.txt        # Python 依赖
├── README.md
├── src/
│   ├── main.py             # 主入口（全流程编排）
│   ├── data/
│   │   └── spectrum.py     # 实验数据管理 + Pydantic 校验
│   ├── models/
│   │   └── planck.py       # Planck 辐射模型
│   ├── fitting/
│   │   └── optimizer.py    # 两阶段参数优化器
│   ├── analysis/
│   │   └── analyzer.py     # Pandas 数据分析引擎
│   ├── visualization/
│   │   └── plotter.py      # Matplotlib 图表生成（8种）
│   ├── report/
│   │   └── generator.py    # 多格式报告（TXT/JSON/CSV）
│   └── utils/
│       └── logger.py       # 结构化日志系统
├── tests/
│   ├── test_spectrum_data.py   # 数据模块测试
│   ├── test_planck_model.py    # 模型测试
│   ├── test_optimizer.py       # 优化器测试
│   ├── test_analysis.py        # 分析模块测试
│   └── test_report.py          # 报告与图表测试
└── output/                 # 运行输出目录
    ├── figures/            # PNG 图表
    └── logs/               # 日志文件
```

## 快速开始

### Docker 运行（推荐）

```bash
# 运行主程序
docker compose up planck-fitting

# 运行测试
docker compose run --rm planck-test
```

### 本地运行

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python -m src.main

# 运行测试
python -m pytest tests/ -v
```

## 输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `fitting_report.txt` | TXT | 详细拟合报告 |
| `fitting_result.json` | JSON | 结构化拟合数据 |
| `fitting_data.csv` | CSV | 实验-计算对比数据表 |
| `analysis_summary.txt` | TXT | 数据分析摘要 |
| `analysis_report.json` | JSON | 分析结果结构化数据 |
| `analysis_master_data.csv` | CSV | 主数据表 |
| `figures/spectral_fitting.png` | PNG | 各源光谱拟合曲线 |
| `figures/multi_source_comparison.png` | PNG | 多源叠加对比 |
| `figures/wien_verification.png` | PNG | Wien 位移验证 |
| `figures/fitting_dashboard.png` | PNG | 综合拟合仪表板 |
| `figures/correlation_heatmap.png` | PNG | 相关性热力图 |
| `figures/parameter_sensitivity.png` | PNG | 参数灵敏度 |
| `figures/residual_distribution.png` | PNG | 残差分布 |
| `figures/analysis_dashboard.png` | PNG | 分析综合仪表板 |

## 核心功能

### 参数拟合
- **两阶段优化**: 差分进化（全局搜索） → L-BFGS-B（局部精化）
- **拟合参数**: 有效温度 T (1000-10000 K)、发射率 ε (0.01-1.0)
- **质量指标**: RMSD、AARD(%)、R²

### 数据分析 (Pandas)
- 描述性统计（均值/方差/CV/范围）
- Wien 位移定律验证
- Stefan-Boltzmann 定律验证
- 残差分析（正态性 Shapiro-Wilk 检验）
- 异常值检测（Z-score）
- 参数灵敏度分析
- 波长分组分析（UV-Vis / NIR / MIR）
- Pearson 相关性矩阵

### 可视化 (Matplotlib)
- 8 种专业图表
- 双仪表板（拟合 + 分析）

## 测试

```bash
# 全量测试（83 个测试用例）
python -m pytest tests/ -v

# 单模块测试
python -m pytest tests/test_planck_model.py -v
```

## 实验数据

| 辐射源 | 描述 | 数据点 | 波长范围 |
|--------|------|--------|----------|
| SRC-1 | 白炽灯丝 (~3000 K) | 15 | 0.50 ~ 10.0 μm |
| SRC-2 | 中温恒星 (~4500 K) | 15 | 0.30 ~ 6.0 μm |
| SRC-3 | 太阳表面 (~6000 K) | 15 | 0.20 ~ 4.0 μm |
