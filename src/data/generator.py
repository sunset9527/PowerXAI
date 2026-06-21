"""
电力负荷数据生成器
生成具有真实规律的合成电力负荷数据

负荷规律：
- 夏季高温负荷高（空调制冷需求）
- 冬季低温负荷高（取暖需求）
- 工作日负荷 > 周末负荷
- 早晚双峰特征（早高峰7-9点，晚高峰17-21点）
- 节假日负荷降低
- 温度敏感性强（极端温度时更明显）
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

# 配置日志
logger = logging.getLogger(__name__)


def get_season(date: datetime) -> str:
    """
    根据日期判断季节

    Args:
        date: 日期对象

    Returns:
        季节名称：spring, summer, autumn, winter
    """
    month = date.month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return "winter"


def is_chinese_holiday(date: datetime) -> bool:
    """
    判断是否为中国的法定节假日
    简化版本，包含主要节假日

    Args:
        date: 日期对象

    Returns:
        是否为节假日
    """
    month = date.month
    day = date.day
    weekday = date.weekday()

    # 周末
    if weekday >= 5:
        return True

    # 元旦 (1月1日)
    if month == 1 and day == 1:
        return True

    # 春节（简化：除夕到初六，大约1月21日-27日范围内）
    if month == 1 and 21 <= day <= 27:
        return True
    if month == 2 and 1 <= day <= 6:
        return True

    # 清明节（简化：4月4日-6日）
    if month == 4 and 4 <= day <= 6:
        return True

    # 劳动节（简化：5月1日-3日）
    if month == 5 and 1 <= day <= 3:
        return True

    # 国庆节（简化：10月1日-7日）
    if month == 10 and 1 <= day <= 7:
        return True

    return False


def calculate_base_load(
    date: datetime,
    hour: int,
    temperature: float,
    is_holiday: bool,
    season: str
) -> float:
    """
    计算基础负荷值

    Args:
        date: 日期对象
        hour: 小时 (0-23)
        temperature: 温度（摄氏度）
        is_holiday: 是否节假日
        season: 季节

    Returns:
        基础负荷值（MW）
    """
    # 基准负荷
    base_load = 1500.0

    # 时间段系数（早晚双峰特征）
    if 7 <= hour <= 9:  # 早高峰
        hour_factor = 1.15
    elif 10 <= hour <= 16:  # 午间时段
        hour_factor = 1.05
    elif 17 <= hour <= 21:  # 晚高峰
        hour_factor = 1.25
    elif 22 <= hour <= 23:  # 夜间下降
        hour_factor = 0.95
    else:  # 深夜低谷
        hour_factor = 0.85

    # 节假日系数（节假日负荷降低）
    holiday_factor = 0.85 if is_holiday else 1.0

    # 周末系数（周末负荷略低）
    weekday = date.weekday()
    weekend_factor = 0.92 if weekday >= 5 else 1.0

    # 季节系数
    if season == "summer":
        # 夏季：高温时负荷增加（空调制冷）
        if temperature > 25:
            season_factor = 1.0 + (temperature - 25) * 0.03
        else:
            season_factor = 1.0 - (25 - temperature) * 0.01
    elif season == "winter":
        # 冬季：低温时负荷增加（取暖）
        if temperature < 15:
            season_factor = 1.0 + (15 - temperature) * 0.025
        else:
            season_factor = 1.0 - (temperature - 15) * 0.01
    elif season == "spring" or season == "autumn":
        # 春秋季：温度适宜，负荷较低
        if 15 <= temperature <= 25:
            season_factor = 0.95
        else:
            season_factor = 1.0
    else:
        season_factor = 1.0

    # 温度敏感系数（极端温度时敏感性增加）
    temp_sensitivity = 1.0
    if temperature > 35 or temperature < 5:
        temp_sensitivity = 1.2
    elif temperature > 30 or temperature < 10:
        temp_sensitivity = 1.1

    # 计算综合负荷
    load = base_load * hour_factor * holiday_factor * weekend_factor * season_factor

    # 应用温度敏感性
    if season == "summer" and temperature > 25:
        load += (temperature - 25) * 15 * temp_sensitivity
    elif season == "winter" and temperature < 15:
        load += (15 - temperature) * 12 * temp_sensitivity

    return max(load, 500)  # 确保负荷不低于最小值


def generate_temperature(date: datetime, hour: int, season: str) -> float:
    """
    生成合成温度数据

    Args:
        date: 日期对象
        hour: 小时
        season: 季节

    Returns:
        温度值（摄氏度）
    """
    # 基础温度（根据季节）
    if season == "summer":
        base_temp = 28.0
        temp_range = 10.0
    elif season == "winter":
        base_temp = 10.0
        temp_range = 12.0
    elif season == "spring":
        base_temp = 18.0
        temp_range = 8.0
    else:  # autumn
        base_temp = 16.0
        temp_range = 8.0

    # 每日变化（正弦曲线，峰值在下午2点）
    hour_angle = (hour - 14) * (np.pi / 12)
    daily_variation = np.sin(hour_angle) * 5

    # 随机噪声
    np.random.seed(hash(f"{date.year}{date.month}{date.day}{hour}") % (2**31))
    noise = np.random.normal(0, 2)

    # 月份内的渐变（简化）
    day_of_year = date.timetuple().tm_yday
    monthly_trend = np.sin((day_of_year - 80) * (2 * np.pi / 365)) * 3

    temperature = base_temp + daily_variation + noise + monthly_trend

    return round(temperature, 1)


def generate_humidity(temperature: float, season: str) -> float:
    """
    生成合成湿度数据（与温度负相关）

    Args:
        temperature: 温度
        season: 季节

    Returns:
        湿度百分比（0-100）
    """
    # 基础湿度
    if season == "summer":
        base_humidity = 65.0
    elif season == "winter":
        base_humidity = 45.0
    else:
        base_humidity = 55.0

    # 与温度负相关
    temp_effect = (25 - temperature) * 0.8

    # 随机噪声
    noise = np.random.normal(0, 8)

    humidity = base_humidity + temp_effect + noise

    return max(20, min(95, round(humidity, 0)))


def generate_load_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    noise_level: Optional[float] = None
) -> pd.DataFrame:
    """
    生成合成电力负荷数据

    Args:
        start_date: 开始日期，格式YYYY-MM-DD
        end_date: 结束日期，格式YYYY-MM-DD
        noise_level: 噪声水平，相对于基础负荷的比例

    Returns:
        包含负荷数据的DataFrame
    """
    # 使用默认配置或指定值
    start_date = start_date or settings.DATA_START_DATE
    end_date = end_date or settings.DATA_END_DATE
    noise_level = noise_level or settings.NOISE_LEVEL

    logger.info(f"开始生成负荷数据: {start_date} 至 {end_date}")

    # 解析日期
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成小时级时间序列
    records = []
    current_date = start

    while current_date <= end:
        for hour in range(24):
            # 创建完整的时间戳
            timestamp = current_date + timedelta(hours=hour)

            # 判断季节
            season = get_season(timestamp)

            # 判断节假日
            is_holiday = is_chinese_holiday(timestamp)

            # 生成温度
            temperature = generate_temperature(timestamp, hour, season)

            # 生成湿度
            humidity = generate_humidity(temperature, season)

            # 计算基础负荷
            base_load = calculate_base_load(
                timestamp, hour, temperature, is_holiday, season
            )

            # 添加噪声
            noise = np.random.normal(0, base_load * noise_level)
            load = base_load + noise

            records.append({
                "datetime": timestamp,
                "date": timestamp.date(),
                "hour": hour,
                "day_of_week": timestamp.weekday(),
                "is_weekend": timestamp.weekday() >= 5,
                "is_holiday": is_holiday,
                "season": season,
                "temperature": temperature,
                "humidity": humidity,
                "load": round(load, 2)
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(records)

    logger.info(f"数据生成完成，共 {len(df)} 条记录")

    return df


def save_raw_data(df: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """
    保存原始数据到文件

    Args:
        df: 数据DataFrame
        filepath: 文件路径，默认为 data/raw/load_data.csv

    Returns:
        保存的文件路径
    """
    filepath = filepath or settings.RAW_DATA_DIR / "load_data.csv"

    # 确保目录存在
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 保存数据
    df.to_csv(filepath, index=False, encoding="utf-8")

    logger.info(f"原始数据已保存至: {filepath}")

    return filepath


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    从文件加载原始数据

    Args:
        filepath: 文件路径，默认为 data/raw/load_data.csv

    Returns:
        数据DataFrame
    """
    filepath = filepath or settings.RAW_DATA_DIR / "load_data.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    df = pd.read_csv(filepath)

    # 转换datetime列
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    logger.info(f"已加载数据 {len(df)} 条记录")

    return df


def ensure_data_exists() -> pd.DataFrame:
    """
    确保原始数据存在，如不存在则生成并保存

    Returns:
        原始数据DataFrame
    """
    filepath = settings.RAW_DATA_DIR / "load_data.csv"

    if filepath.exists():
        logger.info("检测到已有数据文件，直接加载")
        return load_raw_data(filepath)
    else:
        logger.info("未检测到数据文件，开始生成新数据")
        df = generate_load_data()
        save_raw_data(df)
        return df


if __name__ == "__main__":
    # 演示数据生成
    logging.basicConfig(level=logging.INFO)

    df = ensure_data_exists()

    print("\n数据概览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n统计信息:")
    print(df.describe())
