"""
天气数据补全器

当真实数据缺少温度/湿度时，使用统计插值或API补全。

功能：
- 历史均值插值（同小时历史均值）
- open-meteo API免费接口补全
- 线性插值 + 边界填充
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeatherComplementer:
    """
    天气数据补全器

    用于补全缺失的温度和湿度数据。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        latitude: float = 40.7128,
        longitude: float = -74.0060
    ):
        """
        初始化天气补全器

        Args:
            df: 输入数据DataFrame
            datetime_col: 日期时间列名
            latitude: 纬度（用于API查询）
            longitude: 经度（用于API查询）
        """
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.latitude = latitude
        self.longitude = longitude

        # 确保datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_col]):
            self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])

        self._complemented = False

    def _needs_complementation(self) -> Tuple[bool, bool]:
        """
        检查是否需要补全

        Returns:
            (需要补全温度, 需要补全湿度)
        """
        needs_temp = (
            'temperature' not in self.df.columns
            or self.df['temperature'].isnull().any()
        )
        needs_humidity = (
            'humidity' not in self.df.columns
            or self.df['humidity'].isnull().any()
        )

        return needs_temp, needs_humidity

    def _get_historical_hourly_mean(
        self,
        column: str,
        hour_col: str = 'hour'
    ) -> pd.Series:
        """
        计算历史每小时均值

        Args:
            column: 要计算的列名
            hour_col: 小时列名

        Returns:
            每小时均值Series
        """
        if hour_col not in self.df.columns:
            self.df[hour_col] = self.df[self.datetime_col].dt.hour

        # 计算非空数据的每小时均值
        hourly_means = (
            self.df.dropna(subset=[column])
            .groupby(hour_col)[column]
            .mean()
        )

        return hourly_means

    def _get_historical_monthly_mean(
        self,
        column: str,
        month_col: str = 'month'
    ) -> pd.Series:
        """
        计算历史每月均值

        Args:
            column: 要计算的列名
            month_col: 月份列名

        Returns:
            每月均值Series
        """
        if month_col not in self.df.columns:
            self.df[month_col] = self.df[self.datetime_col].dt.month

        # 计算非空数据的每月均值
        monthly_means = (
            self.df.dropna(subset=[column])
            .groupby(month_col)[column]
            .mean()
        )

        return monthly_means

    def complement_with_historical_mean(self) -> 'WeatherComplementer':
        """
        使用历史均值补全天气数据

        结合同小时均值和同月均值进行补全。

        Returns:
            self（链式调用）
        """
        logger.info("使用历史均值补全天气数据...")

        needs_temp, needs_humidity = self._needs_complementation()

        # 添加辅助列
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df[self.datetime_col].dt.hour
        if 'month' not in self.df.columns:
            self.df['month'] = self.df[self.datetime_col].dt.month

        # 补全温度
        if needs_temp:
            if 'temperature' not in self.df.columns:
                self.df['temperature'] = np.nan

            temp_null_mask = self.df['temperature'].isnull()

            if temp_null_mask.any():
                # 获取每小时和每月的温度均值
                hourly_temp = self._get_historical_hourly_mean('temperature')
                monthly_temp = self._get_historical_monthly_mean('temperature')

                # 综合两者进行补全
                for idx in self.df[temp_null_mask].index:
                    hour = self.df.loc[idx, 'hour']
                    month = self.df.loc[idx, 'month']

                    hourly_val = hourly_temp.get(hour, 20.0)
                    monthly_val = monthly_temp.get(month, 20.0)

                    # 取加权平均（更重视季节性）
                    self.df.loc[idx, 'temperature'] = monthly_val * 0.6 + hourly_val * 0.4

                n_temp_filled = temp_null_mask.sum()
                logger.info(f"温度数据已补全: {n_temp_filled} 个值")

        # 补全湿度
        if needs_humidity:
            if 'humidity' not in self.df.columns:
                self.df['humidity'] = np.nan

            humidity_null_mask = self.df['humidity'].isnull()

            if humidity_null_mask.any():
                # 获取每小时和每月的湿度均值
                hourly_humidity = self._get_historical_hourly_mean('humidity')
                monthly_humidity = self._get_historical_monthly_mean('humidity')

                # 综合两者进行补全
                for idx in self.df[humidity_null_mask].index:
                    hour = self.df.loc[idx, 'hour']
                    month = self.df.loc[idx, 'month']

                    hourly_val = hourly_humidity.get(hour, 50.0)
                    monthly_val = monthly_humidity.get(month, 50.0)

                    # 取加权平均
                    self.df.loc[idx, 'humidity'] = monthly_val * 0.6 + hourly_val * 0.4

                n_humidity_filled = humidity_null_mask.sum()
                logger.info(f"湿度数据已补全: {n_humidity_filled} 个值")

        self._complemented = True

        return self

    def complement_with_interpolation(self) -> 'WeatherComplementer':
        """
        使用线性插值补全天气数据

        适用于数据中有少量缺失的情况。

        Returns:
            self（链式调用）
        """
        logger.info("使用线性插值补全天气数据...")

        needs_temp, needs_humidity = self._needs_complementation()

        # 按时间排序确保插值正确
        self.df = self.df.sort_values(self.datetime_col).reset_index(drop=True)

        # 线性插值
        if needs_temp and 'temperature' in self.df.columns:
            n_before = self.df['temperature'].isnull().sum()
            self.df['temperature'] = self.df['temperature'].interpolate(method='linear')
            # 边界填充
            self.df['temperature'] = (
                self.df['temperature']
                .fillna(method='ffill')
                .fillna(method='bfill')
            )
            n_after = self.df['temperature'].isnull().sum()
            logger.info(f"温度插值: 补全了 {n_before - n_after} 个值")

        if needs_humidity and 'humidity' in self.df.columns:
            n_before = self.df['humidity'].isnull().sum()
            self.df['humidity'] = self.df['humidity'].interpolate(method='linear')
            # 边界填充
            self.df['humidity'] = (
                self.df['humidity']
                .fillna(method='ffill')
                .fillna(method='bfill')
            )
            n_after = self.df['humidity'].isnull().sum()
            logger.info(f"湿度插值: 补全了 {n_before - n_after} 个值")

        self._complemented = True

        return self

    def complement_with_api(
        self,
        api_cache: Optional[Dict] = None
    ) -> 'WeatherComplementer':
        """
        使用open-meteo API补全天气数据

        Args:
            api_cache: API响应缓存（用于测试或离线场景）

        Returns:
            self（链式调用）
        """
        logger.info("尝试使用open-meteo API补全天气数据...")

        needs_temp, needs_humidity = self._needs_complementation()

        if not needs_temp and not needs_humidity:
            logger.info("无需补全天气数据")
            return self

        try:
            import requests

            # 获取需要补全的时间范围
            df_sorted = self.df.sort_values(self.datetime_col)
            start_date = df_sorted[self.datetime_col].min()
            end_date = df_sorted[self.datetime_col].max()

            # 构建API URL
            api_url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={self.latitude}"
                f"&longitude={self.longitude}"
                f"&start_date={start_date.strftime('%Y-%m-%d')}"
                f"&end_date={end_date.strftime('%Y-%m-%d')}"
                f"&hourly=temperature_2m,relative_humidity_2m"
                f"&timezone=auto"
            )

            logger.info(f"请求天气API: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

            response = requests.get(api_url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if 'hourly' in data:
                    api_temps = data['hourly'].get('temperature_2m', [])
                    api_humidity = data['hourly'].get('relative_humidity_2m', [])

                    if len(api_temps) == len(self.df):
                        if needs_temp:
                            self.df['temperature'] = api_temps
                            logger.info(f"API补全温度: {len(api_temps)} 个值")

                        if needs_humidity:
                            self.df['humidity'] = api_humidity
                            logger.info(f"API补全湿度: {len(api_humidity)} 个值")

                        self._complemented = True
                else:
                    logger.warning("API返回数据格式异常")
            else:
                logger.warning(f"API请求失败: {response.status_code}")

        except ImportError:
            logger.warning("requests库未安装，无法使用API补全")
        except Exception as e:
            logger.warning(f"API补全失败: {e}")

        return self

    def complement(
        self,
        method: str = 'auto',
        fallback_interpolation: bool = True
    ) -> pd.DataFrame:
        """
        补全天气数据

        Args:
            method: 补全方法 ('auto', 'historical', 'interpolation', 'api')
            fallback_interpolation: 方法失败时是否回退到插值

        Returns:
            补全后的DataFrame
        """
        logger.info(f"开始补全天气数据，方法: {method}")

        needs_temp, needs_humidity = self._needs_complementation()

        if not needs_temp and not needs_humidity:
            logger.info("无需补全天气数据")
            return self.df

        if method == 'auto':
            # 优先尝试历史均值
            self.complement_with_historical_mean()
            # 回退到插值
            if fallback_interpolation:
                self.complement_with_interpolation()

        elif method == 'historical':
            self.complement_with_historical_mean()

        elif method == 'interpolation':
            self.complement_with_interpolation()

        elif method == 'api':
            self.complement_with_api()
            # API失败时回退
            if fallback_interpolation:
                self.complement_with_interpolation()

        # 验证补全结果
        remaining_temp_null = (
            self.df['temperature'].isnull().sum()
            if 'temperature' in self.df.columns else 0
        )
        remaining_humidity_null = (
            self.df['humidity'].isnull().sum()
            if 'humidity' in self.df.columns else 0
        )

        logger.info(
            f"补全完成: 温度剩余缺失 {remaining_temp_null}, "
            f"湿度剩余缺失 {remaining_humidity_null}"
        )

        return self.df

    def get_complementation_report(self) -> Dict:
        """
        获取补全报告

        Returns:
            补全报告字典
        """
        return {
            'was_complemented': self._complemented,
            'current_nulls': {
                'temperature': int(self.df['temperature'].isnull().sum())
                    if 'temperature' in self.df.columns else None,
                'humidity': int(self.df['humidity'].isnull().sum())
                    if 'humidity' in self.df.columns else None,
            }
        }


def complement_weather(
    df: pd.DataFrame,
    method: str = 'auto',
    datetime_col: str = "datetime",
    **kwargs
) -> pd.DataFrame:
    """
    补全天气数据的便捷函数

    Args:
        df: 输入DataFrame
        method: 补全方法
        datetime_col: 日期时间列名
        **kwargs: 传递给WeatherComplementer的额外参数

    Returns:
        补全后的DataFrame
    """
    complementer = WeatherComplementer(df, datetime_col, **kwargs)
    return complementer.complement(method=method)


if __name__ == "__main__":
    # 演示天气补全
    logging.basicConfig(level=logging.INFO)

    from .generator import ensure_data_exists

    # 加载数据
    df = ensure_data_exists()

    # 模拟缺失部分数据
    df_missing = df.copy()
    df_missing.loc[100:150, 'temperature'] = np.nan
    df_missing.loc[200:250, 'humidity'] = np.nan

    print(f"缺失数据前 - 温度缺失: {df_missing['temperature'].isnull().sum()}")
    print(f"缺失数据前 - 湿度缺失: {df_missing['humidity'].isnull().sum()}")

    # 补全
    df_complemented = complement_weather(df_missing)

    print(f"补全后 - 温度缺失: {df_complemented['temperature'].isnull().sum()}")
    print(f"补全后 - 湿度缺失: {df_complemented['humidity'].isnull().sum()}")

    print("\n补全后数据样本:")
    print(df_complemented[['datetime', 'temperature', 'humidity']].head(10))
