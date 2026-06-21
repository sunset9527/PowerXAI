"""
模型预测服务

功能：
- 加载训练好的模型
- 单次预测
- 批量预测
- 返回预测值+置信区间
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import settings
from .trainer import load_model, ensure_model_exists

logger = logging.getLogger(__name__)


# 全局模型缓存
_model_cache: Optional[Dict] = None


def get_model() -> Dict:
    """
    获取模型（带缓存）

    Returns:
        模型数据字典
    """
    global _model_cache

    if _model_cache is None:
        _model_cache = load_model()

    return _model_cache


def predict(
    features: Union[Dict, pd.DataFrame],
    return_confidence: bool = True
) -> Dict:
    """
    单次预测

    Args:
        features: 特征字典或单行DataFrame
        return_confidence: 是否返回置信区间

    Returns:
        预测结果字典
    """
    model_data = get_model()
    model = model_data["model"]
    confidence_model = model_data["confidence_model"]
    feature_columns = model_data["training_info"]["feature_columns"]

    # 转换输入格式
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features

    # 确保特征列顺序正确
    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"特征 {col} 不在输入中，使用默认值0")
            df[col] = 0

    X = df[feature_columns].values

    # 预测
    prediction = model.predict(X)[0]

    result = {
        "prediction": round(float(prediction), 2),
        "unit": "MW"
    }

    # 置信区间
    if return_confidence:
        # 使用置信区间模型估计不确定性
        uncertainty = confidence_model.predict(X)[0]

        # 计算置信区间（95%）
        lower_bound = prediction - 1.96 * uncertainty
        upper_bound = prediction + 1.96 * uncertainty

        result["lower_bound"] = round(float(max(0, lower_bound)), 2)
        result["upper_bound"] = round(float(upper_bound), 2)
        result["confidence_interval"] = round(float(uncertainty * 1.96), 2)

    return result


def predict_batch(
    df: pd.DataFrame,
    return_confidence: bool = True
) -> pd.DataFrame:
    """
    批量预测

    Args:
        df: 包含特征的DataFrame
        return_confidence: 是否返回置信区间

    Returns:
        添加了预测结果的DataFrame
    """
    model_data = get_model()
    model = model_data["model"]
    confidence_model = model_data["confidence_model"]
    feature_columns = model_data["training_info"]["feature_columns"]

    # 确保特征列存在
    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"特征 {col} 不在输入中，使用默认值0")
            df[col] = 0

    X = df[feature_columns].values

    # 批量预测
    predictions = model.predict(X)

    # 置信区间
    if return_confidence:
        uncertainties = confidence_model.predict(X)
        lower_bounds = predictions - 1.96 * uncertainties
        upper_bounds = predictions + 1.96 * uncertainties

    # 构建结果
    result_df = df.copy()
    result_df["predicted_load"] = np.round(predictions, 2)

    if return_confidence:
        result_df["lower_bound"] = np.round(np.maximum(0, lower_bounds), 2)
        result_df["upper_bound"] = np.round(upper_bounds, 2)

    return result_df


def predict_with_history(
    current_features: Dict,
    historical_df: pd.DataFrame,
    datetime_col: str = "datetime"
) -> Dict:
    """
    使用历史数据增强的预测

    自动从历史数据中提取滞后特征

    Args:
        current_features: 当前时刻的特征字典
        historical_df: 历史数据DataFrame（已排序）
        datetime_col: 日期时间列名

    Returns:
        预测结果字典
    """
    from datetime import datetime, timedelta

    model_data = get_model()
    feature_columns = model_data["training_info"]["feature_columns"]

    # 确保历史数据已排序
    hist_df = historical_df.sort_values(datetime_col).copy()

    # 解析当前时间
    if isinstance(current_features.get("datetime"), str):
        current_dt = datetime.strptime(current_features["datetime"], "%Y-%m-%d %H:%M:%S")
    else:
        current_dt = current_features.get("datetime", datetime.now())

    # 提取滞后特征
    lag_1h = hist_df[hist_df[datetime_col] == current_dt - timedelta(hours=1)]
    lag_24h = hist_df[hist_df[datetime_col] == current_dt - timedelta(days=1)]
    lag_168h = hist_df[hist_df[datetime_col] == current_dt - timedelta(days=7)]

    # 填充滞后特征
    if not lag_1h.empty:
        current_features["load_lag_1h"] = lag_1h["load"].values[0]
    if not lag_24h.empty:
        current_features["load_lag_24h"] = lag_24h["load"].values[0]
    if not lag_168h.empty:
        current_features["load_lag_168h"] = lag_168h["load"].values[0]
        current_features["load_same_hour_last_week"] = lag_168h["load"].values[0]

    # 计算变化率
    if (
        "load_lag_1h" in current_features
        and "load_lag_24h" in current_features
        and lag_1h is not None
        and not lag_1h.empty
    ):
        prev_1h = hist_df[hist_df[datetime_col] == current_dt - timedelta(hours=2)]
        if not prev_1h.empty:
            current_features["load_change_1h"] = (
                current_features["load_lag_1h"] - prev_1h["load"].values[0]
            ) / prev_1h["load"].values[0]

        current_features["load_change_24h"] = (
            current_features["load_lag_1h"] - current_features["load_lag_24h"]
        ) / current_features["load_lag_24h"]

    # 计算移动平均
    recent_df = hist_df[
        (hist_df[datetime_col] >= current_dt - timedelta(hours=3))
        & (hist_df[datetime_col] < current_dt)
    ]
    if not recent_df.empty:
        current_features["load_ma_3h"] = recent_df["load"].mean()
    else:
        current_features["load_ma_3h"] = current_features.get("load_lag_1h", 1500)

    recent_24h = hist_df[
        (hist_df[datetime_col] >= current_dt - timedelta(hours=24))
        & (hist_df[datetime_col] < current_dt)
    ]
    if not recent_24h.empty:
        current_features["load_ma_24h"] = recent_24h["load"].mean()
    else:
        current_features["load_ma_24h"] = current_features.get("load_lag_24h", 1500)

    # 执行预测
    return predict(current_features)


def reload_model():
    """
    重新加载模型（清除缓存）
    """
    global _model_cache
    _model_cache = None
    get_model()
    logger.info("模型已重新加载")


if __name__ == "__main__":
    # 演示预测
    logging.basicConfig(level=logging.INFO)

    # 示例特征
    features = {
        "hour": 14,
        "day_of_week": 2,
        "day_of_week_sin": 0.974,
        "day_of_week_cos": -0.225,
        "is_weekend": 0,
        "is_holiday": 0,
        "is_workday": 1,
        "month": 6,
        "month_sin": 0.5,
        "month_cos": 0.866,
        "day_of_month": 18,
        "day_of_month_sin": 0.485,
        "day_of_month_cos": 0.875,
        "hour_sin": 0.975,
        "hour_cos": -0.222,
        "season": "summer",
        "temperature": 35.0,
        "humidity": 60,
        "apparent_temperature": 38.5,
        "thermal_comfort_index": 10.0,
        "heating_cooling_index": 19.5,
        "temperature_squared": 1225.0,
        "temp_workday_interaction": 35.0,
        "temp_humidity_interaction": 21.0,
        "heat_humidity_index": 35.5,
        "load_lag_1h": 2700.0,
        "load_lag_24h": 2600.0,
        "load_lag_168h": 2500.0,
        "load_same_hour_last_week": 2500.0,
        "load_change_1h": 0.02,
        "load_change_24h": 0.05,
        "load_ma_3h": 2680.0,
        "load_ma_24h": 2650.0,
    }

    result = predict(features)
    print("\n预测结果:")
    print(f"  预测负荷: {result['prediction']} MW")
    print(f"  置信区间: [{result['lower_bound']}, {result['upper_bound']}] MW")
