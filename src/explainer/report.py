"""
分析报告生成

功能：
- 基于SHAP分析生成结构化报告
- 包含top-5影响特征
- 正/负向贡献分解
- 异常特征识别
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .shap_analyzer import (
    SHAPAnalyzer,
    get_feature_contribution,
    get_analyzer
)

logger = logging.getLogger(__name__)


def get_top_contributors(
    contributions: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    获取贡献最大的特征

    Args:
        contributions: 特征贡献列表
        top_n: 返回数量

    Returns:
        Top-N特征贡献列表
    """
    return contributions[:top_n]


def identify_anomalies(
    contributions: List[Dict],
    features: Dict,
    threshold_multiplier: float = 2.0
) -> List[Dict]:
    """
    识别异常特征

    基于特征值与常规范围的偏离程度识别异常

    Args:
        contributions: 特征贡献列表
        features: 特征值字典
        threshold_multiplier: 阈值乘数

    Returns:
        异常特征列表
    """
    anomalies = []

    # 定义常规范围
    normal_ranges = {
        "temperature": (5, 40),
        "humidity": (20, 95),
        "hour": (0, 23),
        "load_lag_1h": (500, 4000),
        "load_lag_24h": (500, 4000),
        "load_lag_168h": (500, 4000),
    }

    for contrib in contributions:
        feature = contrib["feature"]
        value = contrib["feature_value"]

        # 检查是否在常规范围外
        if feature in normal_ranges:
            low, high = normal_ranges[feature]
            if value < low or value > high:
                anomalies.append({
                    "feature": feature,
                    "value": value,
                    "normal_range": (low, high),
                    "deviation": "low" if value < low else "high",
                    "deviation_amount": abs(value - low) if value < low else abs(value - high),
                })

        # 检查极端SHAP贡献
        if contrib["abs_contribution"] > 200:  # 超过200MW的特征贡献
            anomalies.append({
                "feature": feature,
                "value": value,
                "shap_value": contrib["shap_value"],
                "deviation": "extreme_contribution",
                "deviation_amount": contrib["abs_contribution"],
            })

    return anomalies


def analyze_seasonal_patterns(
    features: Dict,
    contributions: List[Dict]
) -> Dict:
    """
    分析季节性模式

    Args:
        features: 特征字典
        contributions: 贡献列表

    Returns:
        季节性分析结果
    """
    season = features.get("season", "unknown")
    month = features.get("month", 6)
    temperature = features.get("temperature", 20)

    patterns = []

    # 夏季模式
    if season == "summer":
        patterns.append("夏季高温模式")
        if temperature > 30:
            patterns.append(f"高温预警：{temperature}°C，制冷需求显著")

    # 冬季模式
    elif season == "winter":
        patterns.append("冬季低温模式")
        if temperature < 10:
            patterns.append(f"低温预警：{temperature}°C，制热需求显著")

    # 工作日/周末模式
    if features.get("is_workday", 0) == 1:
        patterns.append("工作日负荷特征")
    elif features.get("is_weekend", False):
        patterns.append("周末负荷特征")
    elif features.get("is_holiday", False):
        patterns.append("节假日负荷特征")

    # 时段模式
    hour = features.get("hour", 12)
    if 7 <= hour <= 9:
        patterns.append("早高峰时段")
    elif 17 <= hour <= 21:
        patterns.append("晚高峰时段")
    elif 0 <= hour <= 5:
        patterns.append("深夜低谷时段")

    return {
        "patterns": patterns,
        "season": season,
        "is_peak_hour": hour in range(7, 10) or hour in range(17, 22),
        "is_weekend": features.get("is_weekend", False),
        "is_holiday": features.get("is_holiday", False),
    }


def generate_report(
    prediction: float,
    features: Dict,
    contributions: List[Dict],
    detail_level: str = "standard"
) -> Dict:
    """
    生成结构化分析报告

    Args:
        prediction: 预测负荷值
        features: 特征字典
        contributions: SHAP贡献列表
        detail_level: 详细程度 ('brief', 'standard', 'detailed')

    Returns:
        结构化报告字典
    """
    # 获取主要贡献
    summary = get_feature_contribution(contributions)

    # 识别异常
    anomalies = identify_anomalies(contributions, features)

    # 季节性分析
    seasonal = analyze_seasonal_patterns(features, contributions)

    # 构建报告
    report = {
        "prediction": prediction,
        "unit": "MW",
        "summary": summary,
        "top_features": get_top_contributors(contributions, top_n=5),
        "positive_contributors": [
            c for c in contributions if c["direction"] == "positive"
        ][:5],
        "negative_contributors": [
            c for c in contributions if c["direction"] == "negative"
        ][:5],
        "anomalies": anomalies,
        "seasonal_patterns": seasonal,
        "detail_level": detail_level,
    }

    # 添加时间上下文
    report["time_context"] = {
        "hour": features.get("hour", 12),
        "day_of_week": features.get("day_of_week", 0),
        "month": features.get("month", 1),
        "season": features.get("season", "unknown"),
        "is_workday": features.get("is_workday", 0) == 1,
        "is_weekend": features.get("is_weekend", False),
        "is_holiday": features.get("is_holiday", False),
    }

    # 添加温度上下文
    report["temperature_context"] = {
        "temperature": features.get("temperature", 20),
        "humidity": features.get("humidity", 50),
        "apparent_temperature": features.get("apparent_temperature", 20),
        "thermal_comfort_index": features.get("thermal_comfort_index", 0),
        "heating_cooling_index": features.get("heating_cooling_index", 0),
    }

    # 添加滞后特征上下文
    report["lag_context"] = {
        "load_lag_1h": features.get("load_lag_1h", None),
        "load_lag_24h": features.get("load_lag_24h", None),
        "load_lag_168h": features.get("load_lag_168h", None),
    }

    return report


def format_report_text(report: Dict) -> str:
    """
    将报告格式化为可读文本

    Args:
        report: 结构化报告字典

    Returns:
        格式化的文本报告
    """
    lines = []

    # 标题
    lines.append("=" * 60)
    lines.append("负荷预测分析报告")
    lines.append("=" * 60)

    # 预测结果
    lines.append(f"\n📊 预测负荷: {report['prediction']:.2f} MW")

    # 时间上下文
    ctx = report.get("time_context", {})
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    day = day_names[ctx.get("day_of_week", 0)]

    time_desc = f"{ctx.get('hour', 12)}:00"
    date_type = "工作日" if ctx.get("is_workday") else (
        "周末" if ctx.get("is_weekend") else "节假日"
    )
    lines.append(f"⏰ 时间: {day} {time_desc} ({date_type})")

    # 温度上下文
    temp_ctx = report.get("temperature_context", {})
    lines.append(
        f"🌡️ 天气: 气温 {temp_ctx.get('temperature', 20):.1f}°C, "
        f"湿度 {temp_ctx.get('humidity', 50):.0f}%"
    )

    # 主要贡献因素
    lines.append("\n📈 主要贡献因素 (Top 5):")
    for i, contrib in enumerate(report.get("top_features", [])[:5], 1):
        direction = "↑" if contrib["direction"] == "positive" else "↓"
        lines.append(
            f"  {i}. {contrib['feature']}: {direction} {contrib['shap_value']:.2f} MW "
            f"(特征值: {contrib['feature_value']:.2f})"
        )

    # 正向贡献
    pos = report.get("positive_contributors", [])
    if pos:
        total_pos = sum(c["shap_value"] for c in pos)
        lines.append(f"\n⬆️ 正向贡献总量: +{total_pos:.2f} MW")

    # 负向贡献
    neg = report.get("negative_contributors", [])
    if neg:
        total_neg = sum(c["shap_value"] for c in neg)
        lines.append(f"⬇️ 负向贡献总量: {total_neg:.2f} MW")

    # 季节性模式
    patterns = report.get("seasonal_patterns", {}).get("patterns", [])
    if patterns:
        lines.append(f"\n🔄 季节性模式: {', '.join(patterns)}")

    # 异常检测
    anomalies = report.get("anomalies", [])
    if anomalies:
        lines.append("\n⚠️ 异常特征:")
        for a in anomalies[:3]:
            if a.get("deviation") in ["low", "high"]:
                lines.append(
                    f"  - {a['feature']}: {a['value']:.2f} "
                    f"(常规范围: {a['normal_range']})"
                )
            else:
                lines.append(
                    f"  - {a['feature']}: 极端贡献 {a.get('shap_value', 0):.2f} MW"
                )

    lines.append("=" * 60)

    return "\n".join(lines)


def compare_reports(
    report1: Dict,
    report2: Dict
) -> Dict:
    """
    比较两个报告的差异

    Args:
        report1: 第一个报告
        report2: 第二个报告

    Returns:
        差异分析结果
    """
    # 预测差异
    pred_diff = report2["prediction"] - report1["prediction"]

    # 获取两个报告的特征贡献
    contrib1 = {c["feature"]: c for c in report1.get("top_features", [])}
    contrib2 = {c["feature"]: c for c in report2.get("top_features", [])}

    # 计算特征贡献变化
    all_features = set(contrib1.keys()) | set(contrib2.keys())
    feature_changes = []

    for feat in all_features:
        shap1 = contrib1.get(feat, {}).get("shap_value", 0)
        shap2 = contrib2.get(feat, {}).get("shap_value", 0)
        change = shap2 - shap1

        if abs(change) > 10:  # 只显示变化超过10MW的特征
            feature_changes.append({
                "feature": feat,
                "shap1": shap1,
                "shap2": shap2,
                "change": change,
                "change_direction": "increase" if change > 0 else "decrease",
            })

    # 排序
    feature_changes.sort(key=lambda x: abs(x["change"]), reverse=True)

    return {
        "prediction_diff": round(pred_diff, 2),
        "prediction_diff_pct": round(pred_diff / report1["prediction"] * 100, 2),
        "feature_changes": feature_changes,
    }


if __name__ == "__main__":
    # 演示报告生成
    logging.basicConfig(level=logging.INFO)

    from .shap_analyzer import explain_prediction

    # 示例特征
    features = {
        "hour": 14,
        "day_of_week": 2,
        "temperature": 35.0,
        "humidity": 60,
        "is_workday": 1,
        "is_weekend": False,
        "is_holiday": False,
        "season": "summer",
        "month": 6,
        "temperature_squared": 1225.0,
        "load_lag_1h": 2700.0,
        "load_lag_24h": 2600.0,
        "load_lag_168h": 2500.0,
        "thermal_comfort_index": 10.0,
        "heating_cooling_index": 19.5,
        "apparent_temperature": 38.5,
        "hour_sin": 0.975,
        "hour_cos": -0.222,
        "day_of_week_sin": 0.974,
        "day_of_week_cos": -0.225,
        "month_sin": 0.5,
        "month_cos": 0.866,
        "day_of_month": 18,
        "day_of_month_sin": 0.485,
        "day_of_month_cos": 0.875,
        "temp_workday_interaction": 35.0,
        "temp_humidity_interaction": 21.0,
        "heat_humidity_index": 35.5,
        "load_same_hour_last_week": 2500.0,
        "load_change_1h": 0.02,
        "load_change_24h": 0.05,
        "load_ma_3h": 2680.0,
        "load_ma_24h": 2650.0,
    }

    # SHAP分析
    contributions = explain_prediction(None, features)

    # 生成报告
    report = generate_report(
        prediction=2850.5,
        features=features,
        contributions=contributions,
        detail_level="standard"
    )

    # 输出报告
    print(format_report_text(report))
