# AI预测分析师 - ML预测 + SHAP可解释 + LLM自然语言分析系统

简体中文 | [English](./README_EN.md)

## 项目简介

AI预测分析师是一个将传统机器学习与大型语言模型相结合的智能预测分析系统。该系统在XGBoost高精度预测的基础上，通过SHAP可解释性分析和LLM自然语言生成，为每个预测结果提供透明、可理解的解释。这是传统ML在AI时代的升级范式，特别适用于金融、能源、银行等行业对模型可解释性有严格要求的场景。

## 核心亮点

### 🎯 SHAP可解释性
- 树模型专用的TreeExplainer实现高效分析
- 单次预测：精确展示每个特征对预测结果的贡献度
- 全局洞察：揭示模型决策规律和特征重要性排序

### 💬 LLM自然语言解释
- DeepSeek大模型驱动的智能分析
- 将冰冷的数值预测转化为专业的业务语言
- 自动生成因果推断、趋势分析和业务建议

### 📊 自动洞察提取
- 趋势检测：自动识别负荷连续上升/下降模式
- 异常告警：及时发现实际值与预测的重大偏离
- 关联分析：量化温度、湿度等因素的影响强度

### 🔄 多维度对比分析
- 时间维度：与历史同期、昨日、前周同期对比
- 因素分解：量化各因素对差异的贡献度
- 归因分析：回答"为什么今天负荷变高了"

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Streamlit  │  │  FastAPI    │  │  Jupyter/CLI           │  │
│  │  Web界面   │  │  REST API   │  │  Python调用            │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼─────────────────┘
          │                │                     │
┌─────────▼────────────────▼─────────────────────▼─────────────────┐
│                      分析服务层 (Analyst)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  LLM Explainer  │  │  Insight Engine  │  │  Comparator     │  │
│  │  自然语言生成    │  │  自动洞察提取    │  │  时段对比分析    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼───────────────────┼────────────────────┼────────────┘
            │                   │                    │
┌───────────▼───────────────────▼────────────────────▼────────────┐
│                    可解释性层 (Explainer)                        │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │  SHAP Tree Explainer    │    │  Report Generator            │ │
│  │  特征贡献度分析          │    │  结构化分析报告生成          │ │
│  └───────────┬─────────────┘    └──────────────┬──────────────┘ │
└──────────────┼──────────────────────────────────┼───────────────┘
               │                                  │
┌──────────────▼──────────────────────────────────▼───────────────┐
│                       模型层 (Model)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  XGBoost    │  │  Predictor   │  │  Evaluator             │ │
│  │  负荷预测    │  │  批量/单次    │  │  MAE/RMSE/MAPE/R²     │ │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────────┘ │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
┌─────────▼────────────────▼──────────────────────────────────────┐
│                      数据层 (Data)                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Data Generator │  │  Preprocessor   │  │  合成负荷数据    │  │
│  │  合成数据生成    │  │  特征工程       │  │  2年历史数据    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **机器学习** | XGBoost | 高精度负荷预测模型 |
| **可解释性** | SHAP | 特征贡献度分析 |
| **大语言模型** | DeepSeek API | 自然语言解释生成 |
| **后端框架** | FastAPI | RESTful API服务 |
| **前端界面** | Streamlit | 交互式Web界面 |
| **数据处理** | Pandas, NumPy | 数据处理与分析 |
| **可视化** | Plotly, Matplotlib | 交互式图表 |
| **配置管理** | pydantic-settings | 配置集中管理 |
| **Python版本** | 3.11+ | 开发环境 |

## 快速启动

### 1. 环境准备

```bash
# 克隆项目（如果使用git）
git clone <repository-url>
cd 项目三_AI预测分析师

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 设置DeepSeek API密钥（必需）
export DEEPSEEK_API_KEY="your-api-key-here"

# 或者在首次运行时，系统会引导您输入
```

### 3. 启动应用

**方式一：Web界面（推荐新手）**
```bash
streamlit run main.py
```
访问 `http://localhost:8501`

**方式二：API服务**
```bash
uvicorn api:app --reload
```
访问 `http://localhost:8000/docs` 查看API文档

**方式三：Python脚本调用**
```python
from src.model.predictor import load_model, predict
from src.explainer.shap_analyzer import explain_prediction
from src.analyst.llm_explainer import generate_explanation

# 预测
prediction = predict(features)
# SHAP分析
shap_values = explain_prediction(prediction, features)
# LLM解释
explanation = generate_explanation(prediction, shap_values, features)
print(explanation)
```

## 项目结构

```
项目三_AI预测分析师/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── config.py                 # 配置管理
├── main.py                   # Streamlit Web界面入口
├── api.py                    # FastAPI REST API服务
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py     # 合成电力负荷数据生成器
│   │   └── preprocessor.py    # 数据预处理+特征工程
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py        # XGBoost模型训练
│   │   ├── predictor.py       # 模型预测服务
│   │   └── evaluator.py       # 模型评估指标
│   ├── explainer/
│   │   ├── __init__.py
│   │   ├── shap_analyzer.py   # SHAP特征贡献分析
│   │   └── report.py          # 分析报告生成
│   ├── analyst/
│   │   ├── __init__.py
│   │   ├── llm_explainer.py   # LLM自然语言解释生成
│   │   ├── insight.py         # 洞察提取（趋势/异常/关联）
│   │   └── comparator.py      # 多时段对比分析
│   └── utils/
│       ├── __init__.py
│       └── visualization.py   # 可视化工具
├── data/
│   ├── raw/                  # 原始合成数据（自动生成）
│   └── processed/            # 处理后数据
├── models/
│   └── xgboost_model.pkl     # 训练好的模型（首次运行自动生成）
└── tests/
    ├── test_predictor.py     # 预测器测试
    ├── test_shap.py          # SHAP分析测试
    ├── test_llm_explainer.py  # LLM解释器测试
    └── test_comparator.py    # 对比分析器测试
```

## 核心功能演示

### 预测分析

```python
# 单次预测
from src.model.predictor import load_model, predict

features = {
    'hour': 14,
    'day_of_week': 2,
    'temperature': 35.0,
    'humidity': 60,
    'is_holiday': False,
    'season': 'summer'
}
result = predict(features)
# 返回: {'prediction': 2850.5, 'lower_bound': 2780.0, 'upper_bound': 2920.0}
```

### SHAP可解释分析

```python
from src.explainer.shap_analyzer import explain_prediction

shap_result = explain_prediction(prediction, features)
# 返回: [
#   {'feature': 'temperature', 'shap_value': 45.2, 'feature_value': 35.0, 'direction': 'positive'},
#   {'feature': 'hour', 'shap_value': 23.1, 'feature_value': 14, 'direction': 'positive'},
#   ...
# ]
```

### LLM自然语言解释

```python
from src.analyst.llm_explainer import generate_explanation

explanation = generate_explanation(
    prediction=2850.5,
    shap_values=shap_result,
    features=features,
    detail_level='standard'  # 'brief' | 'standard' | 'detailed'
)
# 输出专业分析报告，包含因果推断和业务建议
```

## 与传统ML项目的对比

| 维度 | 传统ML项目 | 本项目 |
|------|-----------|--------|
| **模型可解释性** | ❌ 黑箱预测 | ✅ SHAP透明化 |
| **结果可理解性** | ❌ 需要数据科学家解读 | ✅ 业务人员可读 |
| **自动化程度** | ❌ 人工分析环节多 | ✅ 自动洞察提取 |
| **交互能力** | ❌ 静态报告 | ✅ 对话式分析 |
| **对比分析** | ❌ 手动对比 | ✅ 智能归因分析 |
| **AI应用契合度** | ❌ 偏传统 | ✅ ML+LLM融合 |

## 应用场景

- ⚡ **电力负荷预测**：电网调度、需求侧管理
- 🏦 **金融预测**：风险评估、交易预测
- 🏭 **工业预测**：产能预测、设备维护
- 🌡️ **能源管理**：暖通空调优化、能耗分析

## 开发指南

### 添加新的特征

在 `src/data/preprocessor.py` 中的 `create_features()` 方法添加新特征。

### 更换预测模型

在 `src/model/trainer.py` 中替换为LightGBM、CatBoost等其他树模型。

### 接入其他LLM

修改 `src/analyst/llm_explainer.py` 中的API调用，适配OpenAI、Claude等。
