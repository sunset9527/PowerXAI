"""
可视化工具

功能：
- SHAP waterfall图（单次预测）
- SHAP summary图（全局特征重要性）
- 预测 vs 实际曲线（时间序列）
- 特征重要性柱状图
- 使用Plotly做交互式图表
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_shap_waterfall(
    contributions: List[Dict],
    base_value: float = None,
    prediction: float = None,
    top_n: int = 10,
    title: str = "SHAP特征贡献分析"
) -> go.Figure:
    """
    绘制SHAP waterfall图

    Args:
        contributions: 特征贡献列表
        base_value: 基础值（期望值）
        prediction: 预测值
        top_n: 显示前N个特征
        title: 图表标题

    Returns:
        Plotly Figure对象
    """
    # 取top特征
    top_contrib = contributions[:top_n]

    # 特征名（中文映射）
    feature_names = [c["feature"] for c in top_contrib]
    shap_values = [c["shap_value"] for c in top_contrib]

    # 颜色映射
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in shap_values]

    # 创建水平条形图
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=feature_names,
        x=shap_values,
        orientation='h',
        marker_color=colors,
        text=[f"+{v:.1f}" if v > 0 else f"{v:.1f}" for v in shap_values],
        textposition='outside',
    ))

    # 添加基准线
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="SHAP值 (MW)",
        yaxis_title="特征",
        height=400,
        margin=dict(l=150),
        template="plotly_white",
    )

    return fig


def plot_shap_summary(
    importance: Dict,
    top_n: int = 15,
    title: str = "全局特征重要性 (SHAP)"
) -> go.Figure:
    """
    绘制SHAP summary图

    Args:
        importance: 全局重要性字典
        top_n: 显示前N个特征
        title: 图表标题

    Returns:
        Plotly Figure对象
    """
    # 排序并取top
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1]["mean_abs_shap"], reverse=True)
    )

    items = list(sorted_importance.items())[:top_n]
    feature_names = [item[0] for item in items]
    mean_values = [item[1]["mean_abs_shap"] for item in items]

    # 颜色映射
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_names)))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=feature_names,
        x=mean_values,
        orientation='h',
        marker=dict(
            color=mean_values,
            colorscale='RdYlGn',
            reversescale=True,
        ),
        text=[f"{v:.2f}" for v in mean_values],
        textposition='outside',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="平均 |SHAP值| (MW)",
        yaxis_title="特征",
        height=max(400, len(feature_names) * 30),
        margin=dict(l=150),
        template="plotly_white",
    )

    return fig


def plot_prediction_vs_actual(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    actual_col: str = "load",
    predicted_col: str = "predicted_load",
    lower_bound_col: str = "lower_bound",
    upper_bound_col: str = "upper_bound",
    title: str = "预测值 vs 实际值",
    sample_freq: Optional[str] = None
) -> go.Figure:
    """
    绘制预测值与实际值对比图

    Args:
        df: 数据DataFrame
        datetime_col: 时间列名
        actual_col: 实际值列名
        predicted_col: 预测值列名
        lower_bound_col: 下界列名（可选）
        upper_bound_col: 上界列名（可选）
        title: 图表标题
        sample_freq: 采样频率（如"1H", "1D"）

    Returns:
        Plotly Figure对象
    """
    df = df.copy()

    # 确保datetime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # 采样（如果数据量太大）
    if sample_freq and len(df) > 1000:
        df = df.set_index(datetime_col).resample(sample_freq).mean().reset_index()

    fig = go.Figure()

    # 添加实际值线
    fig.add_trace(go.Scatter(
        x=df[datetime_col],
        y=df[actual_col],
        mode='lines',
        name='实际值',
        line=dict(color='#3498db', width=1.5),
    ))

    # 添加预测值线
    fig.add_trace(go.Scatter(
        x=df[datetime_col],
        y=df[predicted_col],
        mode='lines',
        name='预测值',
        line=dict(color='#e74c3c', width=1.5),
    ))

    # 添加置信区间
    if lower_bound_col in df.columns and upper_bound_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[datetime_col],
            y=df[upper_bound_col],
            mode='lines',
            name='上界',
            line=dict(color='#e74c3c', width=0, dash='dash'),
            showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=df[datetime_col],
            y=df[lower_bound_col],
            mode='lines',
            name='下界',
            line=dict(color='#e74c3c', width=0, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.1)',
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="负荷 (MW)",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="x unified",
    )

    return fig


def plot_feature_importance(
    importance: Dict,
    top_n: int = 15,
    title: str = "特征重要性排序"
) -> go.Figure:
    """
    绘制特征重要性柱状图

    Args:
        importance: 重要性字典
        top_n: 显示前N个特征
        title: 图表标题

    Returns:
        Plotly Figure对象
    """
    # 排序
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    items = list(sorted_importance.items())[:top_n]
    feature_names = [item[0] for item in items]
    importance_values = [item[1] for item in items]

    # 归一化到0-1
    max_val = max(importance_values) if importance_values else 1
    normalized = [v / max_val for v in importance_values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=feature_names,
        x=normalized,
        orientation='h',
        marker=dict(
            color=normalized,
            colorscale='Viridis',
        ),
        text=[f"{v:.4f}" for v in importance_values],
        textposition='outside',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="重要性 (归一化)",
        yaxis_title="特征",
        height=max(400, len(feature_names) * 30),
        margin=dict(l=150),
        template="plotly_white",
    )

    return fig


def plot_time_series(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    value_cols: List[str] = None,
    title: str = "时序图",
    yaxis_title: str = "值"
) -> go.Figure:
    """
    绘制时序图

    Args:
        df: 数据DataFrame
        datetime_col: 时间列名
        value_cols: 要绘制的值列名列表
        title: 图表标题
        yaxis_title: Y轴标题

    Returns:
        Plotly Figure对象
    """
    df = df.copy()

    # 确保datetime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    fig = go.Figure()

    # 默认绘制第一列数值
    if value_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if datetime_col in numeric_cols:
            numeric_cols.remove(datetime_col)
        value_cols = numeric_cols[:3]  # 最多3列

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    for i, col in enumerate(value_cols):
        fig.add_trace(go.Scatter(
            x=df[datetime_col],
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_hourly_pattern(
    df: pd.DataFrame,
    hour_col: str = "hour",
    load_col: str = "load",
    title: str = "小时负荷模式"
) -> go.Figure:
    """
    绘制小时负荷模式图

    Args:
        df: 数据DataFrame
        hour_col: 小时列名
        load_col: 负荷列名
        title: 图表标题

    Returns:
        Plotly Figure对象
    """
    # 按小时聚合
    hourly_avg = df.groupby(hour_col)[load_col].agg(['mean', 'std']).reset_index()

    fig = go.Figure()

    # 平均值
    fig.add_trace(go.Scatter(
        x=hourly_avg[hour_col],
        y=hourly_avg['mean'],
        mode='lines+markers',
        name='平均负荷',
        line=dict(color='#3498db', width=2),
        error_y=dict(
            type='data',
            array=hourly_avg['std'],
            visible=True,
            color='#3498db',
            thickness=0.5,
            width=3,
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="小时",
        yaxis_title="负荷 (MW)",
        template="plotly_white",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
        ),
    )

    return fig


def plot_weekly_pattern(
    df: pd.DataFrame,
    dow_col: str = "day_of_week",
    load_col: str = "load",
    title: str = "周负荷模式"
) -> go.Figure:
    """
    绘制周负荷模式图

    Args:
        df: 数据DataFrame
        dow_col: 星期列名
        load_col: 负荷列名
        title: 图表标题

    Returns:
        Plotly Figure对象
    """
    # 按星期聚合
    weekly_avg = df.groupby(dow_col)[load_col].agg(['mean', 'std']).reset_index()

    # 星期名称映射
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekly_avg['day_name'] = weekly_avg[dow_col].map(lambda x: day_names[x])

    fig = go.Figure()

    colors = ['#3498db'] * 5 + ['#2ecc71'] * 2  # 工作日蓝色，周末绿色

    fig.add_trace(go.Bar(
        x=weekly_avg['day_name'],
        y=weekly_avg['mean'],
        marker_color=colors,
        text=[f"{v:.0f}" for v in weekly_avg['mean']],
        textposition='outside',
        error_y=dict(
            type='data',
            array=weekly_avg['std'],
            visible=True,
            color='gray',
            thickness=1,
            width=3,
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="星期",
        yaxis_title="平均负荷 (MW)",
        template="plotly_white",
    )

    return fig


def plot_temperature_load_relationship(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    load_col: str = "load",
    title: str = "温度-负荷关系"
) -> go.Figure:
    """
    绘制温度-负荷关系散点图

    Args:
        df: 数据DataFrame
        temp_col: 温度列名
        load_col: 负荷列名
        title: 图表标题

    Returns:
        Plotly Figure对象
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[temp_col],
        y=df[load_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df[load_col],
            colorscale='Viridis',
            opacity=0.6,
        ),
        text=df.index,
        hovertemplate='温度: %{x:.1f}°C<br>负荷: %{y:.0f} MW',
    ))

    # 添加趋势线
    z = np.polyfit(df[temp_col], df[load_col], 2)
    p = np.poly1d(z)
    x_line = np.linspace(df[temp_col].min(), df[temp_col].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name='趋势线',
        line=dict(color='red', width=2, dash='dash'),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="温度 (°C)",
        yaxis_title="负荷 (MW)",
        template="plotly_white",
    )

    return fig


def create_dashboard(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    actual_col: str = "load",
    predicted_col: str = "predicted_load",
    title: str = "负荷预测分析仪表板"
) -> go.Figure:
    """
    创建综合仪表板

    Args:
        df: 数据DataFrame
        datetime_col: 时间列名
        actual_col: 实际值列名
        predicted_col: 预测值列名
        title: 仪表板标题

    Returns:
        Plotly Figure对象（子图）
    """
    # 创建2x2子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "预测 vs 实际",
            "小时负荷模式",
            "周负荷模式",
            "温度-负荷关系"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
    )

    # 1. 预测 vs 实际
    df_sample = df.set_index(datetime_col).resample('1D').mean().reset_index()

    fig.add_trace(
        go.Scatter(
            x=df_sample[datetime_col],
            y=df_sample[actual_col],
            mode='lines',
            name='实际值',
            line=dict(color='#3498db')
        ),
        row=1, col=1
    )

    if predicted_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df_sample[datetime_col],
                y=df_sample[predicted_col],
                mode='lines',
                name='预测值',
                line=dict(color='#e74c3c')
            ),
            row=1, col=1
        )

    # 2. 小时模式
    hourly = df.groupby('hour')[actual_col].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=hourly['hour'],
            y=hourly[actual_col],
            mode='lines+markers',
            name='小时模式',
            line=dict(color='#2ecc71')
        ),
        row=1, col=2
    )

    # 3. 周模式
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekly = df.groupby('day_of_week')[actual_col].mean().reset_index()
    weekly['day_name'] = weekly['day_of_week'].map(lambda x: day_names[x])
    colors = ['#3498db'] * 5 + ['#2ecc71'] * 2

    fig.add_trace(
        go.Bar(
            x=weekly['day_name'],
            y=weekly[actual_col],
            marker_color=colors,
            name='周模式'
        ),
        row=2, col=1
    )

    # 4. 温度-负荷关系
    fig.add_trace(
        go.Scatter(
            x=df['temperature'],
            y=df[actual_col],
            mode='markers',
            marker=dict(size=4, color='#9b59b6', opacity=0.5),
            name='温度关系'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=True,
        height=800,
    )

    return fig


if __name__ == "__main__":
    # 演示可视化
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    np.random.seed(42)
    n = 100

    data = {
        "datetime": pd.date_range("2023-06-01", periods=n, freq="H"),
        "load": 2000 + np.random.randn(n) * 100,
        "predicted_load": 2000 + np.random.randn(n) * 100,
        "temperature": 25 + np.sin(np.arange(n) * np.pi / 12) * 10,
        "hour": pd.date_range("2023-06-01", periods=n, freq="H").hour,
        "day_of_week": pd.date_range("2023-06-01", periods=n, freq="H").dayofweek,
    }

    df = pd.DataFrame(data)

    # 示例SHAP贡献
    contributions = [
        {"feature": "temperature", "shap_value": 45.2, "feature_value": 35.0, "direction": "positive"},
        {"feature": "load_lag_1h", "shap_value": 23.1, "feature_value": 2700.0, "direction": "positive"},
        {"feature": "humidity", "shap_value": -12.5, "feature_value": 60.0, "direction": "negative"},
        {"feature": "is_workday", "shap_value": 18.3, "feature_value": 1, "direction": "positive"},
        {"feature": "hour", "shap_value": 8.7, "feature_value": 14, "direction": "positive"},
    ]

    # 示例重要性
    importance = {
        "temperature": {"mean_abs_shap": 45.2, "std_shap": 10.5},
        "load_lag_1h": {"mean_abs_shap": 38.1, "std_shap": 8.2},
        "humidity": {"mean_abs_shap": 22.3, "std_shap": 5.6},
        "is_workday": {"mean_abs_shap": 18.5, "std_shap": 4.1},
        "hour": {"mean_abs_shap": 15.2, "std_shap": 3.8},
    }

    # 生成图表
    fig1 = plot_shap_waterfall(contributions)
    fig2 = plot_shap_summary(importance)
    fig3 = plot_prediction_vs_actual(df)

    print("图表已创建")
    print("  - fig1: SHAP Waterfall")
    print("  - fig2: SHAP Summary")
    print("  - fig3: Prediction vs Actual")
