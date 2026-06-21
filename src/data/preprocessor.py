"""
数据预处理与特征工程模块

功能：
- 时间特征提取
- 滞后特征创建（前1小时、前24小时、前168小时）
- 温度衍生特征（体感温度、冷热度指数）
- 数据标准化
- 训练/测试集划分
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import settings

logger = logging.getLogger(__name__)


def calculate_apparent_temperature(temperature: float, humidity: float) -> float:
    """
    计算体感温度（简化版）

    使用简化的热量指数公式计算体感温度。

    Args:
        temperature: 实际温度（摄氏度）
        humidity: 相对湿度（百分比）

    Returns:
        体感温度（摄氏度）
    """
    # 简化的体感温度公式（适用于高温）
    if temperature >= 20:
        # 高温体感（类似热指数）
        T = temperature
        R = humidity
        apparent = (
            -8.78469475556
            + 1.61139411 * T
            + 2.33854883889 * R
            - 0.14611605 * T * R
            - 0.012308094 * T**2
            - 0.0164248277778 * R**2
        )
    else:
        # 低温体感（风寒指数简化版，这里假设风速固定）
        T = temperature
        R = humidity
        # 湿度越低，体感越冷
        apparent = T - (50 - R) * 0.05 if R < 50 else T

    return round(apparent, 1)


def calculate_thermal_comfort_index(temperature: float) -> float:
    """
    计算热舒适指数

    Args:
        temperature: 温度

    Returns:
        热舒适指数（偏离舒适区25°C的程度）
    """
    comfortable_temp = 25.0  # 人体舒适温度
    return temperature - comfortable_temp


def calculate_heating_cooling_index(temperature: float) -> float:
    """
    计算制热制冷需求指数

    Args:
        temperature: 温度

    Returns:
        正值表示制冷需求，负值表示制热需求
    """
    if temperature > 22:
        # 制冷需求
        return (temperature - 22) * 1.5
    elif temperature < 18:
        # 制热需求
        return (18 - temperature) * 1.2
    else:
        return 0.0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建特征工程

    Args:
        df: 原始数据DataFrame

    Returns:
        添加了特征的DataFrame
    """
    logger.info("开始特征工程...")

    df = df.copy()

    # 确保datetime列存在
    if "datetime" not in df.columns:
        raise ValueError("数据中缺少datetime列")

    # ==================== 时间特征 ====================
    # 已经是datetime类型则直接使用，否则转换
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # 小时特征（周期性编码 - sin/cos）
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 星期几特征（周期性编码）
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 月份特征（周期性编码）
    df["month"] = df["datetime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 日特征（周期性编码）
    df["day_of_month"] = df["datetime"].dt.day
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    # 是否工作日（排除周末和节假日）
    df["is_workday"] = (~df["is_weekend"]) & (~df["is_holiday"])
    df["is_workday"] = df["is_workday"].astype(int)

    # ==================== 滞后特征 ====================
    # 按时间排序
    df = df.sort_values("datetime").reset_index(drop=True)

    # 前1小时负荷（lag_1）
    df["load_lag_1h"] = df["load"].shift(1)

    # 前24小时负荷（lag_24）
    df["load_lag_24h"] = df["load"].shift(24)

    # 前168小时负荷（一周前的同一小时）
    df["load_lag_168h"] = df["load"].shift(168)

    # 同期平均负荷（上周同小时平均，可用于平滑）
    df["load_same_hour_last_week"] = df["load_lag_168h"]

    # 负荷变化率（前1小时）
    df["load_change_1h"] = df["load"].pct_change(periods=1).replace([np.inf, -np.inf], 0)

    # 负荷变化率（前24小时）
    df["load_change_24h"] = df["load"].pct_change(periods=24).replace([np.inf, -np.inf], 0)

    # 移动平均（3小时）
    df["load_ma_3h"] = df["load"].rolling(window=3, min_periods=1).mean()

    # 移动平均（24小时）
    df["load_ma_24h"] = df["load"].rolling(window=24, min_periods=1).mean()

    # ==================== 温度衍生特征 ====================
    # 体感温度
    df["apparent_temperature"] = df.apply(
        lambda row: calculate_apparent_temperature(
            row["temperature"], row["humidity"]
        ),
        axis=1
    )

    # 热舒适指数
    df["thermal_comfort_index"] = df["temperature"].apply(calculate_thermal_comfort_index)

    # 制热制冷指数
    df["heating_cooling_index"] = df["temperature"].apply(calculate_heating_cooling_index)

    # 温度平方（捕捉非线性关系）
    df["temperature_squared"] = df["temperature"] ** 2

    # 温度类别（离散化）
    def categorize_temperature(temp: float) -> str:
        if temp < 10:
            return "cold"
        elif temp < 20:
            return "cool"
        elif temp < 25:
            return "comfortable"
        elif temp < 30:
            return "warm"
        else:
            return "hot"

    df["temperature_category"] = df["temperature"].apply(categorize_temperature)

    # 湿度类别
    def categorize_humidity(hum: float) -> str:
        if hum < 40:
            return "dry"
        elif hum < 60:
            return "moderate"
        elif hum < 80:
            return "humid"
        else:
            return "very_humid"

    df["humidity_category"] = df["humidity"].apply(categorize_humidity)

    # ==================== 交互特征 ====================
    # 温度 × 工作日
    df["temp_workday_interaction"] = df["temperature"] * df["is_workday"]

    # 温度 × 湿度
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100

    # 高温高湿指数（夏季闷热程度）
    df["heat_humidity_index"] = (
        df["temperature"] * 0.7 + df["humidity"] * 0.3 * 0.3
    )

    # ==================== 时段特征 ====================
    def get_time_period(hour: int) -> str:
        if 0 <= hour < 6:
            return "night_late"
        elif 6 <= hour < 9:
            return "morning_peak"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "noon"
        elif 14 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening_peak"
        else:
            return "night_early"

    df["time_period"] = df["hour"].apply(get_time_period)

    # ==================== 处理缺失值 ====================
    # 滞后特征可能产生NaN（前几个小时）
    initial_rows = df["load_lag_1h"].isna().sum()
    if initial_rows > 0:
        logger.info(f"因滞后特征产生 {initial_rows} 个缺失值，将使用前向填充")
        df = df.fillna(method="ffill")

    # 最终检查并删除任何剩余的缺失值
    remaining_na = df.isna().sum().sum()
    if remaining_na > 0:
        logger.warning(f"仍有 {remaining_na} 个缺失值，将删除这些行")
        df = df.dropna()

    logger.info(f"特征工程完成，共 {len(df.columns)} 个特征")

    return df


def preprocess_data(df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    数据预处理主函数

    Args:
        df: 原始数据DataFrame，如果为None则从文件加载

    Returns:
        (处理后的DataFrame, 标准化器)
    """
    if df is None:
        from .generator import load_raw_data
        df = load_raw_data()

    # 特征工程
    df_processed = create_features(df)

    # 保存处理后的数据
    output_path = settings.PROCESSED_DATA_DIR / "processed_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"处理后数据已保存至: {output_path}")

    # 创建并拟合标准化器（仅对连续特征）
    continuous_features = [
        "temperature", "humidity", "apparent_temperature",
        "thermal_comfort_index", "heating_cooling_index",
        "temperature_squared", "temp_humidity_interaction",
        "heat_humidity_index"
    ]

    scaler = StandardScaler()

    # 检查哪些特征存在
    available_features = [f for f in continuous_features if f in df_processed.columns]

    if available_features:
        scaler.fit(df_processed[available_features])

    return df_processed, scaler


def split_train_test(
    df: pd.DataFrame,
    train_ratio: Optional[float] = None,
    test_days: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    划分训练集和测试集

    默认使用最后1个月（约30天）作为测试集

    Args:
        df: 处理后的DataFrame
        train_ratio: 训练集比例（与test_days互斥）
        test_days: 测试集天数（优先使用）

    Returns:
        (训练集, 测试集)
    """
    train_ratio = train_ratio or settings.TRAIN_RATIO
    df = df.sort_values("datetime").reset_index(drop=True)

    if test_days:
        # 按天数划分
        total_hours = len(df)
        test_hours = test_days * 24
        train_hours = total_hours - test_hours

        train_df = df.iloc[:train_hours]
        test_df = df.iloc[train_hours:]
    else:
        # 按比例划分
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    logger.info(f"训练集: {len(train_df)} 条 ({train_df['datetime'].min()} 至 {train_df['datetime'].max()})")
    logger.info(f"测试集: {len(test_df)} 条 ({test_df['datetime'].min()} 至 {test_df['datetime'].max()})")

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    获取用于模型的特征列

    Args:
        df: 处理后的DataFrame

    Returns:
        特征列名列表
    """
    # 排除列
    exclude_columns = [
        "datetime", "date", "load",  # 时间相关和目标变量
        "time_period", "temperature_category", "humidity_category"  # 字符串分类特征
    ]

    # 获取所有数值列
    feature_columns = [
        col for col in df.columns
        if col not in exclude_columns
        and df[col].dtype in ["int64", "float64", "int32", "float32"]
    ]

    return feature_columns


def prepare_prediction_features(
    datetime_str: str,
    hour: int,
    day_of_week: int,
    temperature: float,
    humidity: float,
    is_holiday: bool,
    season: str,
    load_lag_1h: Optional[float] = None,
    load_lag_24h: Optional[float] = None,
    load_lag_168h: Optional[float] = None,
    historical_mean_load: Optional[float] = None
) -> Dict:
    """
    为单次预测准备特征

    Args:
        datetime_str: 日期时间字符串
        hour: 小时
        day_of_week: 星期几 (0=周一, 6=周日)
        temperature: 温度
        humidity: 湿度
        is_holiday: 是否节假日
        season: 季节
        load_lag_1h: 前1小时负荷（可选）
        load_lag_24h: 前24小时负荷（可选）
        load_lag_168h: 前168小时负荷（可选）
        historical_mean_load: 历史同期平均负荷（可选）

    Returns:
        特征字典
    """
    from datetime import datetime

    # 解析日期
    if isinstance(datetime_str, str):
        dt = datetime.strptime(datetime_str, "%Y-%m-%d")
    else:
        dt = datetime_str

    is_weekend = day_of_week >= 5

    # 周期性编码
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month = dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_month = dt.day
    dom_sin = np.sin(2 * np.pi * day_of_month / 31)
    dom_cos = np.cos(2 * np.pi * day_of_month / 31)

    # 工作日
    is_workday = 1 if (not is_weekend and not is_holiday) else 0

    # 滞后特征（使用提供的值或默认值）
    default_lag = 1500.0  # 默认负荷值
    lag_1 = load_lag_1h if load_lag_1h is not None else default_lag
    lag_24 = load_lag_24h if load_lag_24h is not None else default_lag
    lag_168 = load_lag_168h if load_lag_168h is not None else default_lag

    # 移动平均（简化）
    ma_3h = (lag_1 + default_lag + default_lag) / 3
    ma_24h = (lag_24 + default_lag * 23) / 24

    # 变化率（简化）
    change_1h = 0.0
    change_24h = 0.0

    # 温度衍生特征
    apparent_temp = calculate_apparent_temperature(temperature, humidity)
    thermal_comfort = calculate_thermal_comfort_index(temperature)
    heating_cooling = calculate_heating_cooling_index(temperature)
    temp_squared = temperature ** 2
    temp_humidity = temperature * humidity / 100
    heat_humidity = temperature * 0.7 + humidity * 0.3 * 0.3

    # 交互特征
    temp_workday = temperature * is_workday

    features = {
        "hour": hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": day_of_week,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
        "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday),
        "is_workday": is_workday,
        "month": month,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_month": day_of_month,
        "day_of_month_sin": dom_sin,
        "day_of_month_cos": dom_cos,
        "season": season,
        "temperature": temperature,
        "humidity": humidity,
        "apparent_temperature": apparent_temp,
        "thermal_comfort_index": thermal_comfort,
        "heating_cooling_index": heating_cooling,
        "temperature_squared": temp_squared,
        "temp_workday_interaction": temp_workday,
        "temp_humidity_interaction": temp_humidity,
        "heat_humidity_index": heat_humidity,
        "load_lag_1h": lag_1,
        "load_lag_24h": lag_24,
        "load_lag_168h": lag_168,
        "load_same_hour_last_week": lag_168,
        "load_change_1h": change_1h,
        "load_change_24h": change_24h,
        "load_ma_3h": ma_3h,
        "load_ma_24h": ma_24h,
    }

    return features


if __name__ == "__main__":
    # 演示数据预处理
    logging.basicConfig(level=logging.INFO)

    from .generator import ensure_data_exists

    # 加载/生成数据
    raw_df = ensure_data_exists()

    # 预处理
    processed_df, scaler = preprocess_data(raw_df)

    # 划分数据集
    train_df, test_df = split_train_test(processed_df)

    # 获取特征列
    feature_cols = get_feature_columns(processed_df)
    print(f"\n特征列 ({len(feature_cols)}):")
    print(feature_cols)

    print(f"\n训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
