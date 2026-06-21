"""
PJM电力负荷数据加载器

支持PJM Hourly Energy Consumption数据集（Kaggle公开数据集）的加载和预处理。

功能：
- 从HuggingFace/Kaggle下载PJM数据
- 统一数据接口格式
- 数据标准化与缺失值处理
- 回退到合成数据（当真实数据不可用时）
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


def download_pjm_data(
    save_path: Optional[Path] = None,
    force_download: bool = False
) -> Optional[Path]:
    """
    从HuggingFace下载PJM Hourly Energy Consumption数据集

    PJM是一个美国区域输电组织，覆盖多个州的电力负荷数据。
    数据集包含小时级的电力负荷、温度等数据。

    Args:
        save_path: 保存路径
        force_download: 是否强制重新下载

    Returns:
        下载的数据文件路径，失败返回None
    """
    save_path = save_path or settings.RAW_DATA_DIR / "pjm_load_data.csv"

    # 如果文件已存在且不强制下载，直接返回
    if save_path.exists() and not force_download:
        logger.info(f"PJM数据已存在: {save_path}")
        return save_path

    logger.info("尝试从HuggingFace下载PJM数据集...")

    try:
        # 尝试使用pandas_datareader或直接下载
        import requests

        # HuggingFace上的PJM数据集镜像（简化版本）
        # 注意：实际使用时需要替换为真实可用的数据源
        hf_url = "https://huggingface.co/datasets/pjm-demo/hourly-energy-consumption/resolve/main/pjm_load_data.csv"

        response = requests.get(hf_url, timeout=30)
        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"PJM数据已下载至: {save_path}")
            return save_path
        else:
            logger.warning(f"下载失败，状态码: {response.status_code}")
            return None

    except ImportError:
        logger.warning("requests库未安装，无法下载数据")
        return None
    except Exception as e:
        logger.warning(f"下载数据时出错: {e}")
        return None


def load_pjm_data(data_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    加载PJM数据文件

    Args:
        data_path: 数据文件路径

    Returns:
        加载的DataFrame，失败返回None
    """
    if data_path is None:
        data_path = settings.RAW_DATA_DIR / "pjm_load_data.csv"

    if not data_path.exists():
        logger.warning(f"PJM数据文件不存在: {data_path}")
        return None

    try:
        df = pd.read_csv(data_path)
        logger.info(f"已加载PJM数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"加载PJM数据失败: {e}")
        return None


def standardize_pjm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    将PJM数据标准化为统一格式

    目标列名:
    - datetime: 日期时间
    - load: 电力负荷（MW）
    - temperature: 温度（摄氏度）
    - humidity: 湿度（百分比）
    - day_of_week: 星期几（0=周一, 6=周日）
    - is_holiday: 是否节假日

    Args:
        df: 原始PJM数据

    Returns:
        标准化后的DataFrame
    """
    logger.info("标准化PJM数据...")

    df = df.copy()

    # 获取列名映射（根据实际PJM数据格式调整）
    column_mapping = {}

    # 常见PJM数据列名
    datetime_candidates = ['datetime', 'Datetime', 'time', 'Time', 'timestamp', 'date']
    load_candidates = ['load', 'Load', 'load_MW', '负荷', 'energy', 'Demand']
    temp_candidates = ['temperature', 'temp', 'Temperature', 'dry_bulb', 'db_temp']
    humidity_candidates = ['humidity', 'rh', 'humidity_percent', 'relative_humidity']

    for col in df.columns:
        col_lower = col.lower()
        if not column_mapping.get('datetime') and col in datetime_candidates:
            column_mapping['datetime'] = col
        elif not column_mapping.get('load') and col in load_candidates:
            column_mapping['load'] = col
        elif not column_mapping.get('temperature') and col in temp_candidates:
            column_mapping['temperature'] = col
        elif not column_mapping.get('humidity') and col in humidity_candidates:
            column_mapping['humidity'] = col

    # 如果没有找到温度列，需要尝试从其他列推断或生成
    if 'temperature' not in column_mapping:
        logger.info("数据中未找到温度列，将使用统计插值补全")
        # 可以尝试从负荷数据反推温度，或使用默认值
        if 'load' in column_mapping:
            # 简化的温度估算：基于负荷水平的反推
            load_mean = df[column_mapping['load']].mean()
            df['temperature'] = 20 + (df[column_mapping['load']] - load_mean) / 100
            column_mapping['temperature'] = 'temperature'
        else:
            df['temperature'] = 20.0  # 默认温度
            column_mapping['temperature'] = 'temperature'

    if 'humidity' not in column_mapping:
        df['humidity'] = 50.0  # 默认湿度
        column_mapping['humidity'] = 'humidity'

    # 重命名列
    df = df.rename(columns=column_mapping)

    # 确保datetime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 确保列存在
    required_cols = ['datetime', 'load', 'temperature', 'humidity']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # 从datetime提取day_of_week
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek

    # 生成is_holiday（简化版）
    if 'is_holiday' not in df.columns:
        df['is_holiday'] = df['datetime'].dt.dayofweek >= 5

    # 只保留需要的列
    output_cols = ['datetime', 'load', 'temperature', 'humidity', 'day_of_week', 'is_holiday']
    df = df[output_cols].copy()

    # 数据类型转换
    df['load'] = pd.to_numeric(df['load'], errors='coerce')
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
    df['day_of_week'] = df['day_of_week'].astype(int)
    df['is_holiday'] = df['is_holiday'].astype(bool)

    logger.info(f"数据标准化完成，共 {len(df)} 条记录")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理缺失值

    Args:
        df: 输入DataFrame

    Returns:
        处理后的DataFrame
    """
    logger.info("处理缺失值...")

    initial_len = len(df)

    # 检查各列缺失值
    missing_stats = df.isnull().sum()
    if missing_stats.sum() > 0:
        logger.info(f"缺失值统计:\n{missing_stats[missing_stats > 0]}")

    # 删除负荷为空的行
    df = df.dropna(subset=['load'])

    # 温度插值（线性插值 + 前向/后向填充）
    if df['temperature'].isnull().any():
        df['temperature'] = df['temperature'].interpolate(method='linear')
        df['temperature'] = df['temperature'].fillna(method='ffill').fillna(method='bfill')

    # 湿度插值
    if df['humidity'].isnull().any():
        df['humidity'] = df['humidity'].interpolate(method='linear')
        df['humidity'] = df['humidity'].fillna(method='ffill').fillna(method='bfill')

    # 如果仍有缺失，用中位数填充
    if df['temperature'].isnull().any():
        df['temperature'] = df['temperature'].fillna(df['temperature'].median())
    if df['humidity'].isnull().any():
        df['humidity'] = df['humidity'].fillna(df['humidity'].median())

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info(f"因缺失值删除了 {dropped} 行")

    return df


def convert_timezone(df: pd.DataFrame, target_tz: str = 'America/New_York') -> pd.DataFrame:
    """
    时区转换（PJM数据通常使用美国东部时区）

    Args:
        df: 输入DataFrame
        target_tz: 目标时区

    Returns:
        转换后的DataFrame
    """
    if pd.api.types.is_datetime64_any_dtype(df['datetime']):
        # 如果没有时区信息，假设是UTC并转换
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        df['datetime'] = df['datetime'].dt.tz_convert(target_tz)

    return df


def load_real_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    统一的数据加载接口

    优先尝试加载真实数据，失败时回退到合成数据。

    Args:
        data_path: 真实数据文件路径（如果指定）

    Returns:
        统一格式的DataFrame
    """
    logger.info("=" * 50)
    logger.info("开始加载数据")
    logger.info("=" * 50)

    # 检查配置
    use_real_data = getattr(settings, 'USE_REAL_DATA', False)
    real_data_path = getattr(settings, 'REAL_DATA_PATH', None)

    # 确定数据路径
    if data_path:
        data_path = Path(data_path)
    elif real_data_path:
        data_path = Path(real_data_path)
    else:
        data_path = settings.RAW_DATA_DIR / "pjm_load_data.csv"

    # 尝试加载真实数据
    df = None

    if use_real_data or data_path.exists():
        logger.info(f"尝试加载真实数据: {data_path}")

        # 如果指定了真实数据路径
        if data_path.suffix == '.csv':
            df = load_pjm_data(data_path)
            if df is not None:
                df = standardize_pjm_data(df)
                df = handle_missing_values(df)

        # 尝试下载PJM数据
        if df is None:
            downloaded_path = download_pjm_data()
            if downloaded_path:
                df = load_pjm_data(downloaded_path)
                if df is not None:
                    df = standardize_pjm_data(df)
                    df = handle_missing_values(df)

    # 回退到合成数据
    if df is None:
        logger.warning("真实数据不可用，回退到合成数据")
        from .generator import generate_load_data, save_raw_data

        df = generate_load_data()
        save_raw_data(df)

    # 确保数据按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    logger.info("数据加载完成")
    logger.info(f"数据范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    logger.info(f"数据形状: {df.shape}")

    return df


def validate_real_data(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    验证真实数据的基本要求

    Args:
        df: 输入DataFrame

    Returns:
        (是否有效, 验证信息字典)
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }

    # 检查必需列
    required_cols = ['datetime', 'load', 'temperature', 'humidity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation['valid'] = False
        validation['errors'].append(f"缺少必需列: {missing_cols}")

    # 检查数据类型
    if 'datetime' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            validation['warnings'].append("datetime列不是datetime类型")

    # 检查数值范围
    if 'load' in df.columns:
        if df['load'].min() < 0:
            validation['errors'].append("负荷值存在负数")
        if df['load'].isnull().all():
            validation['errors'].append("负荷值全部为空")

    if 'temperature' in df.columns:
        if df['temperature'].min() < -50 or df['temperature'].max() > 60:
            validation['warnings'].append("温度值超出合理范围(-50°C ~ 60°C)")

    if 'humidity' in df.columns:
        if df['humidity'].min() < 0 or df['humidity'].max() > 100:
            validation['warnings'].append("湿度值超出合理范围(0% ~ 100%)")

    # 检查数据量
    if len(df) < 100:
        validation['valid'] = False
        validation['errors'].append(f"数据量过少: {len(df)} 行")

    validation['valid'] = len(validation['errors']) == 0

    return validation['valid'], validation


def prepare_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备训练数据（添加额外的基础列）

    Args:
        df: 原始数据

    Returns:
        添加了额外列的DataFrame
    """
    df = df.copy()

    # 添加hour列
    df['hour'] = df['datetime'].dt.hour

    # 添加is_weekend列
    df['is_weekend'] = df['day_of_week'] >= 5

    # 添加season列
    def get_season(month: int) -> str:
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'winter'

    df['season'] = df['datetime'].dt.month.apply(get_season)

    # 确保date列存在
    df['date'] = df['datetime'].dt.date

    return df


if __name__ == "__main__":
    # 演示数据加载
    logging.basicConfig(level=logging.INFO)

    df = load_real_data()

    print("\n数据概览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n列信息:")
    print(df.dtypes)
    print(f"\n统计信息:")
    print(df.describe())

    # 验证数据
    valid, info = validate_real_data(df)
    print(f"\n数据验证: {'通过' if valid else '失败'}")
    if info['warnings']:
        print(f"警告: {info['warnings']}")
    if info['errors']:
        print(f"错误: {info['errors']}")
