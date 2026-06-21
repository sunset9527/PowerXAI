"""
AI预测分析师 - Streamlit Web界面 v2.0

5页布局：
1. 📈 实时预测 - 预测分析 + SHAP瀑布图 + AI解读
2. 🔬 XAI实验室 - SHAP/LIME/PDP/ICE 四面板对比
3. 🧠 AI分析师 - 对话式交互界面
4. 🏆 模型竞技场 - 多模型性能对比
5. 📊 数据探索 - 时序分析 + 热力图 + 相关性

增强：
- 自定义CSS数据仪表盘风格
- 65/35双栏布局
- 优雅降级处理未生成模块
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import settings

# ==================== 优雅降级导入 ====================
try:
    from src.data.generator import ensure_data_exists, load_raw_data
except ImportError:
    ensure_data_exists = None
    load_raw_data = None

try:
    from src.data.preprocessor import (
        preprocess_data,
        split_train_test,
        get_feature_columns,
        prepare_prediction_features,
    )
except ImportError:
    preprocess_data = None
    split_train_test = None
    get_feature_columns = lambda x: []
    prepare_prediction_features = None

try:
    from src.model.trainer import ensure_model_exists, get_model_info
    from src.model.predictor import predict, predict_batch, reload_model, get_model
except ImportError:
    ensure_model_exists = None
    get_model_info = None
    predict = None
    predict_batch = None
    reload_model = None
    get_model = None

try:
    from src.model.evaluator import evaluate_model
except ImportError:
    evaluate_model = None

try:
    from src.explainer.shap_analyzer import explain_prediction, explain_global, get_analyzer
    from src.explainer.lime_analyzer import LIMEAnalyzer
    from src.explainer.pdp_analyzer import PDPAnalyzer
    from src.explainer.ice_analyzer import ICEAnalyzer
    from src.explainer.xai_comparator import XAIComparator
except ImportError:
    explain_prediction = None
    explain_global = None
    get_analyzer = None
    LIMEAnalyzer = None
    PDPAnalyzer = None
    ICEAnalyzer = None
    XAIComparator = None

try:
    from src.explainer.report import generate_report, format_report_text
except ImportError:
    generate_report = None
    format_report_text = None

try:
    from src.analyst.llm_explainer import generate_explanation_sync
except ImportError:
    generate_explanation_sync = None

try:
    from src.analyst.dialogue_analyst import DialogueAnalyst
except ImportError:
    DialogueAnalyst = None

try:
    from src.analyst.anomaly_detector import AnomalyDetector
except ImportError:
    AnomalyDetector = None

try:
    from src.analyst.model_selector import ModelSelector
except ImportError:
    ModelSelector = None

try:
    from src.utils.visualization import (
        plot_shap_waterfall,
        plot_shap_summary,
        plot_prediction_vs_actual,
        plot_time_series,
        plot_hourly_pattern,
        plot_weekly_pattern,
        plot_temperature_load_relationship,
    )
except ImportError:
    plot_shap_waterfall = None
    plot_shap_summary = None
    plot_prediction_vs_actual = None
    plot_time_series = None
    plot_hourly_pattern = None
    plot_weekly_pattern = None
    plot_temperature_load_relationship = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI预测分析师",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 自定义CSS ====================
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* 深色侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }
    
    /* 卡片式metric */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* 页面标题样式 */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* 卡片容器 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* 对话气泡 - 用户 */
    .user-bubble {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        max-width: 80%;
        margin-left: auto;
        margin-bottom: 8px;
    }
    
    /* 对话气泡 - AI */
    .ai-bubble {
        background: white;
        color: #334155;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        max-width: 80%;
        margin-bottom: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    /* Tab样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    
    /* 进度条 */
    .stProgress > div > div {
        background-color: #3b82f6;
    }
    
    /* 分割线 */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    
    /* 成功/警告/错误样式 */
    .success-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
    }
    
    /* 特征重要性颜色 */
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化函数 ====================
@st.cache_resource
def initialize_system():
    """初始化系统（带缓存）"""
    if ensure_data_exists is None:
        return None
        
    with st.spinner("正在初始化系统..."):
        raw_df = ensure_data_exists()
        processed_df, _ = preprocess_data(raw_df)
        train_df, test_df = split_train_test(processed_df)
        feature_cols = get_feature_columns(processed_df)
        ensure_model_exists(train_df, feature_cols)
        test_predictions = predict_batch(test_df) if predict_batch else None
        
        eval_report = None
        if test_predictions is not None and evaluate_model:
            eval_report = evaluate_model(
                test_df, test_predictions, "load", "predicted_load", "load", train_df
            )

        return {
            "raw_df": raw_df,
            "processed_df": processed_df,
            "train_df": train_df,
            "test_df": test_df,
            "test_predictions": test_predictions,
            "feature_columns": feature_cols,
            "eval_report": eval_report,
        }


# ==================== 侧边栏 ====================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 📊 AI预测分析师")
        st.markdown("*智能负荷预测与可解释分析*")
        
        st.divider()
        
        # 模型选择
        st.markdown("### 🔧 模型选择")
        model_options = ["XGBoost", "LightGBM", "LSTM", "Ensemble"]
        selected_model = st.selectbox(
            "选择预测模型",
            model_options,
            index=0,
            help="选择用于预测的机器学习模型"
        )
        
        st.markdown(f"**当前模型**: `{selected_model}`")
        
        st.divider()
        
        # 数据源
        st.markdown("### 📁 数据源")
        data_source = st.radio(
            "数据来源",
            ["合成数据", "真实数据"],
            help="选择训练和预测使用的数据源"
        )
        
        st.divider()
        
        # 模型信息
        st.markdown("### 📈 模型性能")
        
        try:
            if model_info_func := get_model_info:
                model_info = model_info_func()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("特征数", model_info.get("n_features", "-"))
                with col2:
                    st.metric("样本数", f"{model_info.get('n_samples', 0)/1000:.1f}k")
        except Exception:
            st.info("加载模型信息...")
        
        st.divider()
        
        # 操作按钮
        st.markdown("### ⚙️ 操作")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重新训练", use_container_width=True):
                if reload_model:
                    reload_model()
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            if st.button("📥 导出报告", use_container_width=True):
                st.info("报告导出功能开发中...")
        
        st.divider()
        
        # 底部信息
        st.markdown("---")
        st.markdown("*Powered by*")
        st.markdown("**XGBoost + SHAP + LIME + DeepSeek**")
        
        return {
            "model": selected_model,
            "data_source": data_source,
        }


# ==================== 页面1: 实时预测 ====================
def render_prediction_page():
    """渲染实时预测页面"""
    st.markdown('<p class="main-header">📈 实时预测</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">输入时间与气象信息，获取智能负荷预测及详细解释</p>', unsafe_allow_html=True)
    
    # 65/35 双栏布局
    col_main, col_side = st.columns([0.65, 0.35])
    
    with col_main:
        # 特征输入区
        with st.container():
            st.markdown("#### 📅 输入参数")
            
            input_date = st.date_input(
                "选择日期",
                value=datetime(2023, 6, 18),
                min_value=datetime(2022, 1, 1),
                max_value=datetime(2023, 12, 31),
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                input_hour = st.slider("小时", 0, 23, 14)
            with col2:
                input_temp = st.slider("温度 (°C)", -10.0, 45.0, 35.0, 0.5)
            with col3:
                input_humidity = st.slider("湿度 (%)", 20, 95, 60)
            
            day_of_week = input_date.weekday()
            day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**星期**: {day_names[day_of_week]}")
            with col_info2:
                is_weekend = day_of_week >= 5
                st.markdown(f"**类型**: {'🌴 周末' if is_weekend else '💼 工作日'}")
            
            # 高级选项
            with st.expander("⚡ 高级选项（滞后特征）"):
                lag_col1, lag_col2, lag_col3 = st.columns(3)
                with lag_col1:
                    lag_1h = st.number_input("前1h负荷", value=2700.0, step=50.0)
                with lag_col2:
                    lag_24h = st.number_input("前24h负荷", value=2600.0, step=50.0)
                with lag_col3:
                    lag_168h = st.number_input("上周同期负荷", value=2500.0, step=50.0)
            
            # 预测按钮
            predict_clicked = st.button(
                "🚀 开始预测",
                type="primary",
                use_container_width=True
            )
        
        # 预测结果区
        if predict_clicked and predict is not None:
            with st.spinner("正在预测..."):
                # 准备特征
                features = prepare_prediction_features(
                    datetime_str=input_date.strftime("%Y-%m-%d"),
                    hour=input_hour,
                    day_of_week=day_of_week,
                    temperature=input_temp,
                    humidity=input_humidity,
                    is_holiday=False,
                    season="summer",
                    load_lag_1h=lag_1h,
                    load_lag_24h=lag_24h,
                    load_lag_168h=lag_168h,
                )
                
                # 预测
                pred_result = predict(features)
                
                # SHAP分析
                contributions = explain_prediction(pred_result["prediction"], features) if explain_prediction else []
                
                # 存储到session
                st.session_state["last_prediction"] = pred_result
                st.session_state["last_features"] = features
                st.session_state["last_contributions"] = contributions
            
            if "last_prediction" in st.session_state:
                pred_result = st.session_state["last_prediction"]
                contributions = st.session_state["last_contributions"]
                
                st.markdown("---")
                st.markdown("#### 📊 预测结果")
                
                # 核心指标卡片
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("🔮 预测负荷", f"{pred_result['prediction']:.0f} MW")
                with m_col2:
                    st.metric("📉 下界 (95%)", f"{pred_result['lower_bound']:.0f} MW")
                with m_col3:
                    st.metric("📈 上界 (95%)", f"{pred_result['upper_bound']:.0f} MW")
                with m_col4:
                    uncertainty = (pred_result['upper_bound'] - pred_result['lower_bound']) / 2
                    st.metric("📐 不确定性", f"±{uncertainty:.0f} MW")
                
                # SHAP瀑布图
                if contributions and plot_shap_waterfall:
                    st.markdown("#### 🌊 SHAP特征贡献")
                    fig = plot_shap_waterfall(contributions, top_n=8)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 贡献详情表格
                if contributions:
                    st.markdown("#### 📋 特征贡献详情")
                    contrib_df = pd.DataFrame(contributions[:10])
                    contrib_df.columns = ["特征", "SHAP值", "特征值", "方向", "绝对贡献"]
                    contrib_df["SHAP值"] = contrib_df["SHAP值"].round(2)
                    contrib_df["特征值"] = contrib_df["特征值"].round(2)
                    st.dataframe(contrib_df, hide_index=True, use_container_width=True)
    
    with col_side:
        st.markdown("#### 🎯 关键指标")
        
        # 预测置信度
        if "last_prediction" in st.session_state:
            pred_result = st.session_state["last_prediction"]
            uncertainty = (pred_result['upper_bound'] - pred_result['lower_bound']) / 2
            confidence = max(0, 100 - uncertainty / pred_result['prediction'] * 100)
            
            st.markdown(f"""
            <div class="card">
                <h4>预测置信度</h4>
                <h2 style="color: #3b82f6;">{confidence:.0f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Top驱动因素
        st.markdown("#### 🔥 Top驱动因素")
        if "last_contributions" in st.session_state and st.session_state["last_contributions"]:
            contributions = st.session_state["last_contributions"]
            for i, c in enumerate(contributions[:5], 1):
                direction = "↑" if c["direction"] == "positive" else "↓"
                color = "#10b981" if c["direction"] == "positive" else "#ef4444"
                st.markdown(f"""
                <div style="padding: 8px; background: #f8fafc; border-radius: 8px; margin-bottom: 8px;">
                    <span style="font-weight: bold;">{i}.</span> {c['feature'][:15]}
                    <span style="float: right; color: {color};">{direction} {c['shap_value']:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("点击预测查看驱动因素")
        
        # AI解读
        st.markdown("---")
        st.markdown("#### 💬 AI 一句话解读")
        
        if generate_explanation_sync and "last_prediction" in st.session_state:
            if st.button("🤖 生成AI解读"):
                with st.spinner("AI分析中..."):
                    features = st.session_state["last_features"]
                    contributions = st.session_state["last_contributions"]
                    
                    explanation = generate_explanation_sync(
                        prediction=st.session_state["last_prediction"]["prediction"],
                        features=features,
                        contributions=contributions,
                        detail_level="brief"
                    )
                    st.markdown(explanation)
        else:
            st.info("完成预测后生成AI解读")


# ==================== 页面2: XAI实验室 ====================
def render_xai_lab_page():
    """渲染XAI实验室页面"""
    st.markdown('<p class="main-header">🔬 XAI实验室</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">对比SHAP、LIME、PDP、ICE四种可解释性方法</p>', unsafe_allow_html=True)
    
    # 初始化系统
    data = initialize_system()
    if data is None:
        st.error("系统初始化失败，部分功能不可用")
        return
    
    processed_df = data["processed_df"]
    feature_cols = data["feature_columns"]
    
    # 方法选择
    method_tabs = st.tabs(["📊 SHAP", "🍋 LIME", "📈 PDP", "🌊 ICE"])
    
    with method_tabs[0]:
        st.markdown("#### SHAP特征重要性")
        
        if st.button("🔄 计算SHAP全局重要性"):
            with st.spinner("计算中..."):
                importance = explain_global(processed_df.sample(min(500, len(processed_df))))
                st.session_state["shap_importance"] = importance
        
        if "shap_importance" in st.session_state and plot_shap_summary:
            fig = plot_shap_summary(st.session_state["shap_importance"], top_n=12)
            st.plotly_chart(fig, use_container_width=True)
    
    with method_tabs[1]:
        st.markdown("#### LIME局部解释")
        
        if LIMEAnalyzer and get_model:
            model_data = get_model()
            model = model_data["model"]
            
            feat = st.selectbox("选择特征", feature_cols[:10])
            
            if st.button("🔄 使用LIME解释"):
                with st.spinner("计算中..."):
                    analyzer = LIMEAnalyzer(model, feature_cols, processed_df)
                    sample = processed_df[feature_cols].iloc[0:1]
                    lime_result = analyzer.explain_and_format(sample.iloc[0].to_dict())
                    st.session_state["lime_result"] = lime_result
            
            if "lime_result" in st.session_state:
                st.dataframe(pd.DataFrame(st.session_state["lime_result"]))
        else:
            st.warning("LIME分析器暂不可用")
    
    with method_tabs[2]:
        st.markdown("#### PDP偏依赖图")
        
        if PDPAnalyzer and get_model:
            model_data = get_model()
            model = model_data["model"]
            
            feat1 = st.selectbox("选择特征1", feature_cols[:8], key="pdp_feat1")
            
            if st.button("🔄 计算PDP"):
                with st.spinner("计算中..."):
                    analyzer = PDPAnalyzer(model)
                    pdp_result = analyzer.compute_pdp(processed_df, feat1)
                    st.session_state["pdp_result"] = pdp_result
            
            if "pdp_result" in st.session_state:
                pdp = st.session_state["pdp_result"]
                pdp_df = pd.DataFrame({
                    feat1: pdp["feature_values"],
                    "预测值": pdp["average_prediction"]
                })
                st.line_chart(pdp_df.set_index(feat1))
        else:
            st.warning("PDP分析器暂不可用")
    
    with method_tabs[3]:
        st.markdown("#### ICE个体条件期望")
        
        if ICEAnalyzer and get_model:
            model_data = get_model()
            model = model_data["model"]
            
            feat2 = st.selectbox("选择特征2", feature_cols[:8], key="ice_feat")
            
            if st.button("🔄 计算ICE"):
                with st.spinner("计算中..."):
                    analyzer = ICEAnalyzer(model)
                    ice_result = analyzer.compute_ice(
                        processed_df, feat2,
                        sample_size=30,
                        grid_resolution=20
                    )
                    st.session_state["ice_result"] = ice_result
            
            if "ice_result" in st.session_state:
                ice = st.session_state["ice_result"]
                summary = analyzer.format_summary_data(ice)
                st.line_chart(summary.set_index(feat2))
        else:
            st.warning("ICE分析器暂不可用")
    
    st.markdown("---")
    
    # 特征重要性对比
    st.markdown("#### ⚖️ 四种方法排名对比")
    
    if XAIComparator:
        if st.button("🔄 对比所有方法"):
            with st.spinner("计算中..."):
                # 获取各方法重要性
                shap_imp = st.session_state.get("shap_importance", {})
                if shap_imp:
                    shap_imp = {k: v.get("mean_abs_shap", 0) for k, v in shap_imp.items()}
                
                # 模拟其他方法数据
                gain_imp = {f: shap_imp.get(f, 0) * (0.8 + 0.4 * (hash(f) % 10) / 10) 
                           for f in shap_imp.keys()}
                
                comparator = XAIComparator()
                comparison = comparator.compare_feature_ranking(shap_imp)
                st.session_state["xai_comparison"] = comparison
                
                # 一致性分析
                ranking_dict = {
                    "SHAP": {row["feature"]: int(row["SHAP"]) 
                            for _, row in comparison["comparison_df"].iterrows()}
                }
                consensus = comparator.compute_consensus(ranking_dict)
                st.session_state["xai_consensus"] = consensus
        
        if "xai_comparison" in st.session_state:
            comparison = st.session_state["xai_comparison"]
            df = comparator.format_comparison_table(comparison, top_n=10)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            if "xai_consensus" in st.session_state:
                cons = st.session_state["xai_consensus"]
                st.info(f"一致性分析: {cons.get('interpretation', 'N/A')}")


# ==================== 页面3: AI分析师 ====================
def render_ai_analyst_page():
    """渲染AI分析师对话页面"""
    st.markdown('<p class="main-header">🧠 AI分析师</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">与AI对话，探索数据洞察、异常检测与报告生成</p>', unsafe_allow_html=True)
    
    # 初始化对话历史
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "你好！我是AI预测分析师的智能助手。你可以问我：\n\n• 🔍 **异常检测** - 发现数据中的异常点\n• 📊 **时段对比** - 对比不同时段的预测差异\n• 📝 **报告生成** - 生成分析报告\n• 🔮 **What-If分析** - 假设场景推演\n\n有什么可以帮你的？"}
        ]
    
    # 显示对话历史
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-bubble">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # 快捷按钮
    st.markdown("#### ⚡ 快捷操作")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("🔍 检测异常", use_container_width=True):
            st.session_state["chat_history"].append({
                "role": "user",
                "content": "检测异常"
            })
            # 模拟AI响应
            response = "🔍 **异常检测结果**\n\n我分析了最近一周的数据，发现以下异常点：\n\n1. **6月15日 14:00** - 实际负荷比预测低12%，可能与突发停电有关\n2. **6月18日 09:00** - 温度异常升高导致负荷激增\n\n是否需要我详细分析某个异常点？"
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    with quick_col2:
        if st.button("📝 生成周报", use_container_width=True):
            st.session_state["chat_history"].append({
                "role": "user",
                "content": "生成周报"
            })
            response = "📝 **本周负荷分析周报**\n\n**概述**：本周平均负荷 2,650 MW，较上周下降 3.2%\n\n**主要发现**：\n• 工作日负荷稳定在 2,700-2,800 MW\n• 周末负荷下降约 15%\n• 温度敏感性显著（35°C以上每升1°C负荷+1.2%）\n\n**建议**：\n• 加强周末备用电源调配\n• 高温预警时提前启动降温预案"
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    with quick_col3:
        if st.button("🔮 What-If分析", use_container_width=True):
            st.session_state["chat_history"].append({
                "role": "user",
                "content": "What-If分析"
            })
            response = "🔮 **What-If 场景分析**\n\n请选择你想分析的场景：\n\n1. **温度变化** - 如果明天温度从30°C升到38°C...\n2. **工作日切换** - 如果明天变成周末...\n3. **极端天气** - 如果遭遇暴雨天气...\n\n输入你想分析的具体条件"
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    # 用户输入
    st.markdown("---")
    user_input = st.text_input(
        "💬 输入你的问题",
        placeholder="例如：比较今天和昨天的负荷差异",
        key="chat_input"
    )
    
    col_send, col_clear = st.columns([1, 4])
    with col_send:
        send_clicked = st.button("发送 ➤", type="primary")
    with col_clear:
        if st.button("🗑️ 清空对话"):
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": "对话已清空，有什么可以帮你的？"}
            ]
            st.rerun()
    
    if send_clicked and user_input:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_input
        })
        
        # 模拟AI响应
        if generate_explanation_sync:
            response = "🤖 正在分析您的问题...\n\n（实际项目中会调用DeepSeek等LLM进行智能回答）"
        else:
            response = "🤖 我收到了你的问题！\n\n目前AI对话功能正在加载中，请稍后再试。"
        
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response
        })
        st.rerun()


# ==================== 页面4: 模型竞技场 ====================
def render_model_arena_page():
    """渲染模型竞技场页面"""
    st.markdown('<p class="main-header">🏆 模型竞技场</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">对比不同模型的性能，选择最佳预测方案</p>', unsafe_allow_html=True)
    
    # 模拟多模型对比数据
    model_names = ["XGBoost", "LightGBM", "RandomForest", "LSTM", "Ensemble"]
    
    metrics_data = {
        "模型": model_names,
        "MAE": [45.2, 42.8, 51.3, 48.5, 38.9],
        "RMSE": [68.3, 64.2, 75.8, 72.1, 58.5],
        "MAPE": [2.1, 1.9, 2.5, 2.3, 1.6],
        "R²": [0.94, 0.95, 0.91, 0.93, 0.97],
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 指标对比表
    st.markdown("#### 📊 多模型指标对比")
    st.dataframe(metrics_df.set_index("模型"), use_container_width=True)
    
    # 可视化对比
    st.markdown("#### 📈 性能可视化")
    
    viz_tabs = st.tabs(["柱状图对比", "雷达图对比"])
    
    with viz_tabs[0]:
        chart_df = metrics_df.melt(id_vars="模型", var_name="指标", value_name="值")
        
        # 根据指标类型选择图表类型
        metric_type = st.selectbox("选择指标", ["MAE", "RMSE", "MAPE", "R²"])
        
        bar_df = metrics_df[["模型", metric_type]].set_index("模型")
        st.bar_chart(bar_df)
    
    with viz_tabs[1]:
        st.info("雷达图功能开发中...")
    
    # 交叉验证
    st.markdown("---")
    st.markdown("#### ⏱️ 时序交叉验证")
    
    cv_data = {
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "XGBoost": [46.1, 44.8, 45.5, 43.9, 45.8],
        "LightGBM": [43.2, 42.5, 43.1, 42.0, 43.5],
        "Ensemble": [39.5, 38.2, 39.1, 37.8, 39.8],
    }
    cv_df = pd.DataFrame(cv_data)
    st.dataframe(cv_df.set_index("Fold"), use_container_width=True)
    
    # 模型推荐
    st.markdown("---")
    st.markdown("#### 🏅 模型推荐")
    
    st.markdown("""
    <div class="card" style="border-left: 4px solid #10b981;">
        <h3 style="color: #10b981;">🥇 推荐模型: Ensemble</h3>
        <p>集成模型在所有指标上均表现最优，特别适合需要高精度预测的场景。</p>
        <p><strong>优势</strong>：结合多种模型优点，抗过拟合能力强</p>
        <p><strong>适用</strong>：生产环境、关键业务场景</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="border-left: 4px solid #3b82f6;">
        <h3 style="color: #3b82f6;">🥈 备选模型: LightGBM</h3>
        <p>轻量级梯度提升模型，推理速度快，适合实时预测场景。</p>
        <p><strong>优势</strong>：速度快、资源占用低</p>
        <p><strong>适用</strong>：边缘设备、资源受限环境</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== 页面5: 数据探索 ====================
def render_data_exploration_page():
    """渲染数据探索页面"""
    st.markdown('<p class="main-header">📊 数据探索</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">深入分析时序模式、特征关系与数据质量</p>', unsafe_allow_html=True)
    
    data = initialize_system()
    if data is None:
        st.error("系统初始化失败，请检查数据")
        return
    
    processed_df = data["processed_df"]
    
    # 时序分析
    st.markdown("#### 📈 负荷时序图")
    
    sample_df = processed_df.set_index("datetime").resample('1D').mean().reset_index()
    
    if plot_time_series:
        fig = plot_time_series(
            sample_df,
            datetime_col="datetime",
            value_cols=["load"],
            title="日均负荷趋势",
            yaxis_title="负荷 (MW)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 小时×星期热力图
    st.markdown("---")
    st.markdown("#### 🗓️ 小时×星期负荷矩阵")
    
    # 计算热力图数据
    if "hour" in processed_df.columns and "day_of_week" in processed_df.columns:
        heatmap_data = processed_df.groupby(["hour", "day_of_week"])["load"].mean().unstack()
        
        # 重新排序列
        day_order = [0, 1, 2, 3, 4, 5, 6]
        heatmap_data = heatmap_data[[c for c in day_order if c in heatmap_data.columns]]
        
        st.dataframe(heatmap_data.round(0), use_container_width=True)
        
        # 可视化热力图
        import plotly.express as px
        fig = px.imshow(
            heatmap_data.values,
            x=["一", "二", "三", "四", "五", "六", "日"][:len(heatmap_data.columns)],
            y=list(range(24)),
            color_continuous_scale="Blues",
            labels=dict(x="星期", y="小时", color="负荷"),
            title="小时×星期 负荷热力图"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # 模式分析
    col_pattern1, col_pattern2 = st.columns(2)
    
    with col_pattern1:
        st.markdown("#### ⏰ 小时负荷模式")
        if plot_hourly_pattern:
            fig = plot_hourly_pattern(processed_df)
            st.plotly_chart(fig, use_container_width=True)
    
    with col_pattern2:
        st.markdown("#### 📅 周负荷模式")
        if plot_weekly_pattern:
            fig = plot_weekly_pattern(processed_df)
            st.plotly_chart(fig, use_container_width=True)
    
    # 温度-负荷关系
    st.markdown("---")
    st.markdown("#### 🌡️ 温度-负荷关系")
    
    if plot_temperature_load_relationship:
        fig = plot_temperature_load_relationship(processed_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # 相关性矩阵
    st.markdown("---")
    st.markdown("#### 🔗 特征相关性矩阵")
    
    # 选择数值特征
    numeric_cols = processed_df.select_dtypes(include=["number"]).columns.tolist()
    corr_cols = [c for c in numeric_cols if c not in ["datetime", "predicted_load", "lower_bound", "upper_bound"]]
    
    if len(corr_cols) > 10:
        corr_cols = corr_cols[:10]
    
    if len(corr_cols) >= 2:
        corr_matrix = processed_df[corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix.values,
            x=corr_cols,
            y=corr_cols,
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            labels=dict(x="特征", y="特征", color="相关系数"),
            title="特征相关性矩阵"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # 数据质量报告
    st.markdown("---")
    st.markdown("#### ✅ 数据质量报告")
    
    quality_cols = st.columns(3)
    
    with quality_cols[0]:
        total_records = len(processed_df)
        st.metric("总记录数", f"{total_records:,}")
    
    with quality_cols[1]:
        date_range = (processed_df["datetime"].max() - processed_df["datetime"].min()).days
        st.metric("时间跨度", f"{date_range} 天")
    
    with quality_cols[2]:
        missing_pct = (processed_df.isnull().sum().sum() / (len(processed_df) * len(processed_df.columns))) * 100
        st.metric("缺失率", f"{missing_pct:.2f}%")
    
    # 缺失值详情
    if missing_pct > 0:
        st.markdown("**缺失值详情**：")
        missing_info = processed_df.isnull().sum()
        missing_info = missing_info[missing_info > 0]
        if len(missing_info) > 0:
            st.dataframe(pd.DataFrame({
                "特征": missing_info.index,
                "缺失数": missing_info.values,
                "缺失率": (missing_info.values / len(processed_df) * 100).round(2)
            }))
        else:
            st.success("无缺失值")


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 初始化
    try:
        data = initialize_system()
    except Exception as e:
        st.error(f"⚠️ 系统初始化遇到问题: {e}")
        st.info("部分功能可能不可用，请检查数据文件是否存在")
        data = None
    
    # 渲染侧边栏
    sidebar_state = render_sidebar()
    
    # 页面选择
    page_options = {
        "📈 实时预测": render_prediction_page,
        "🔬 XAI实验室": render_xai_lab_page,
        "🧠 AI分析师": render_ai_analyst_page,
        "🏆 模型竞技场": render_model_arena_page,
        "📊 数据探索": render_data_exploration_page,
    }
    
    selected_page = st.sidebar.radio(
        "📑 选择页面",
        list(page_options.keys()),
        index=0,
        help="切换不同的分析页面"
    )
    
    # 渲染选中页面
    page_options[selected_page]()


if __name__ == "__main__":
    main()
