"""
数据探索器

提供数据质量检测和统计摘要功能。

功能：
- 时序数据统计分析
- 缺失值检测
- 异常值检测
- 数据分布摘要
- 生成结构化数据画像
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataExplorer:
    """
    数据探索器

    用于分析数据质量、统计特征和分布情况。
    """

    def __init__(self, df: pd.DataFrame, datetime_col: str = "datetime"):
        """
        初始化数据探索器

        Args:
            df: 输入数据DataFrame
            datetime_col: 日期时间列名
        """
        self.df = df.copy()
        self.datetime_col = datetime_col
        self._ensure_datetime()

    def _ensure_datetime(self):
        """确保datetime列是datetime类型"""
        if self.datetime_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.datetime_col]):
                self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])

    def get_basic_info(self) -> Dict:
        """
        获取数据基本信息

        Returns:
            基本信息字典
        """
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'datetime_range': {
                'start': str(self.df[self.datetime_col].min()) if self.datetime_col in self.df.columns else None,
                'end': str(self.df[self.datetime_col].max()) if self.datetime_col in self.df.columns else None,
            },
            'date_range_days': (
                (self.df[self.datetime_col].max() - self.df[self.datetime_col].min()).days
                if self.datetime_col in self.df.columns else None
            )
        }

    def detect_missing_values(self) -> Dict:
        """
        检测缺失值

        Returns:
            缺失值统计字典
        """
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df) * 100).round(2)

        missing_info = {}
        for col in self.df.columns:
            if missing_counts[col] > 0:
                missing_info[col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percent[col]),
                    'dtype': str(self.df[col].dtype)
                }

        return {
            'total_missing': int(missing_counts.sum()),
            'missing_columns': missing_info,
            'is_complete': missing_counts.sum() == 0
        }

    def detect_duplicates(self) -> Dict:
        """
        检测重复行

        Returns:
            重复数据统计
        """
        n_duplicates = self.df.duplicated().sum()

        # 检查时间列重复
        datetime_duplicates = 0
        if self.datetime_col in self.df.columns:
            datetime_duplicates = self.df[self.datetime_col].duplicated().sum()

        return {
            'duplicate_rows': int(n_duplicates),
            'duplicate_datetimes': int(datetime_duplicates),
            'has_duplicates': n_duplicates > 0
        }

    def detect_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict:
        """
        检测异常值

        Args:
            columns: 要检查的列（默认所有数值列）
            method: 检测方法 ('iqr' 或 'zscore')
            threshold: 阈值 (IQR倍数或Z-score)

        Returns:
            异常值统计字典
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            data = self.df[col].dropna()

            if len(data) == 0:
                continue

            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            else:  # zscore
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((data - mean) / std)
                outliers = data[z_scores > threshold]

            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': int(len(outliers)),
                    'percentage': round(len(outliers) / len(data) * 100, 2),
                    'min_outlier': float(outliers.min()) if len(outliers) > 0 else None,
                    'max_outlier': float(outliers.max()) if len(outliers) > 0 else None,
                    'method': method
                }

        return {
            'outlier_columns': outlier_info,
            'total_outliers': sum(info['count'] for info in outlier_info.values()),
            'has_outliers': len(outlier_info) > 0
        }

    def check_time_continuity(
        self,
        freq: str = 'H',
        max_gap: Optional[int] = None
    ) -> Dict:
        """
        检查时间连续性

        Args:
            freq: 预期频率 ('H' = 小时, 'D' = 天)
            max_gap: 最大允许间隔（小时数）

        Returns:
            时间连续性统计
        """
        if self.datetime_col not in self.df.columns:
            return {'error': 'datetime列不存在'}

        df_sorted = self.df.sort_values(self.datetime_col).copy()
        time_diffs = df_sorted[self.datetime_col].diff()

        if freq == 'H':
            expected_diff = pd.Timedelta(hours=1)
            unit = 'hours'
        else:
            expected_diff = pd.Timedelta(days=1)
            unit = 'days'

        # 计算与预期频率的偏差
        diffs_hours = time_diffs.dt.total_seconds() / 3600
        diffs_hours = diffs_hours.dropna()

        if len(diffs_hours) == 0:
            return {'is_continuous': True, 'gaps': []}

        # 识别时间跳跃
        expected_hourly = 1 if freq == 'H' else 24
        gaps = diffs_hours[diffs_hours > (max_gap or expected_hourly * 2)]

        gap_info = []
        for idx, gap_hours in gaps.items():
            gap_start = df_sorted.loc[idx - 1, self.datetime_col] if idx > 0 else None
            gap_end = df_sorted.loc[idx, self.datetime_col]
            gap_info.append({
                'start': str(gap_start) if gap_start else None,
                'end': str(gap_end),
                'hours_gap': float(gap_hours)
            })

        return {
            'is_continuous': len(gaps) == 0,
            'expected_frequency': freq,
            'n_gaps': len(gaps),
            'total_missing_hours': float(diffs_hours.sum() - len(diffs_hours)),
            'gaps': gap_info[:10]  # 最多显示10个
        }

    def get_distribution_summary(self, columns: Optional[List[str]] = None) -> Dict:
        """
        获取数据分布摘要

        Args:
            columns: 要分析的列（默认所有数值列）

        Returns:
            分布统计字典
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        summary = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            data = self.df[col].dropna()

            if len(data) == 0:
                continue

            summary[col] = {
                'count': int(len(data)),
                'mean': round(float(data.mean()), 4),
                'std': round(float(data.std()), 4),
                'min': round(float(data.min()), 4),
                'q25': round(float(data.quantile(0.25)), 4),
                'median': round(float(data.median()), 4),
                'q75': round(float(data.quantile(0.75)), 4),
                'max': round(float(data.max()), 4),
                'skewness': round(float(data.skew()), 4),
                'kurtosis': round(float(data.kurtosis()), 4)
            }

        return summary

    def get_temporal_patterns(self, target_col: str = 'load') -> Dict:
        """
        获取时序模式分析

        Args:
            target_col: 目标变量列名

        Returns:
            时序模式字典
        """
        if self.datetime_col not in self.df.columns or target_col not in self.df.columns:
            return {'error': '必需的列不存在'}

        df_sorted = self.df.sort_values(self.datetime_col)

        patterns = {
            'hourly_pattern': {},
            'daily_pattern': {},
            'monthly_pattern': {},
            'weekly_pattern': {}
        }

        # 小时模式
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df[self.datetime_col].dt.hour

        hourly = df_sorted.groupby('hour')[target_col].agg(['mean', 'std', 'count'])
        patterns['hourly_pattern'] = {
            int(idx): {'mean': round(row['mean'], 2), 'std': round(row['std'], 2)}
            for idx, row in hourly.iterrows()
        }

        # 星期模式
        if 'day_of_week' not in self.df.columns:
            self.df['day_of_week'] = self.df[self.datetime_col].dt.dayofweek

        weekly = df_sorted.groupby('day_of_week')[target_col].agg(['mean', 'std'])
        patterns['weekly_pattern'] = {
            int(idx): {'mean': round(row['mean'], 2), 'std': round(row['std'], 2)}
            for idx, row in weekly.iterrows()
        }

        # 月份模式
        if 'month' not in self.df.columns:
            self.df['month'] = self.df[self.datetime_col].dt.month

        monthly = df_sorted.groupby('month')[target_col].agg(['mean', 'std'])
        patterns['monthly_pattern'] = {
            int(idx): {'mean': round(row['mean'], 2), 'std': round(row['std'], 2)}
            for idx, row in monthly.iterrows()
        }

        return patterns

    def explore(self) -> Dict:
        """
        执行完整的数据探索

        Returns:
            完整的探索报告
        """
        logger.info("开始数据探索...")

        report = {
            'basic_info': self.get_basic_info(),
            'missing_values': self.detect_missing_values(),
            'duplicates': self.detect_duplicates(),
            'outliers': self.detect_outliers(),
            'distribution': self.get_distribution_summary(),
        }

        # 可选：时间连续性检查
        try:
            report['time_continuity'] = self.check_time_continuity()
        except Exception as e:
            logger.warning(f"时间连续性检查失败: {e}")
            report['time_continuity'] = {'error': str(e)}

        # 可选：时序模式分析
        if 'load' in self.df.columns:
            try:
                report['temporal_patterns'] = self.get_temporal_patterns()
            except Exception as e:
                logger.warning(f"时序模式分析失败: {e}")

        logger.info("数据探索完成")

        return report


def generate_data_profile(df: pd.DataFrame, datetime_col: str = "datetime") -> Dict:
    """
    生成结构化数据画像

    Args:
        df: 输入数据DataFrame
        datetime_col: 日期时间列名

    Returns:
        数据画像字典
    """
    explorer = DataExplorer(df, datetime_col)
    return explorer.explore()


def print_data_profile(profile: Dict) -> str:
    """
    格式化输出数据画像

    Args:
        profile: 数据画像字典

    Returns:
        格式化的字符串
    """
    lines = ["=" * 60, "数据画像报告", "=" * 60]

    # 基本信息
    if 'basic_info' in profile:
        info = profile['basic_info']
        lines.extend([
            f"\n【基本信息】",
            f"  记录数: {info['n_rows']:,}",
            f"  特征数: {info['n_columns']}",
            f"  内存占用: {info['memory_usage_mb']} MB",
            f"  时间范围: {info['datetime_range']['start']} 至 {info['datetime_range']['end']}",
            f"  跨度天数: {info['date_range_days']} 天"
        ])

    # 缺失值
    if 'missing_values' in profile:
        missing = profile['missing_values']
        lines.append(f"\n【缺失值】")
        if missing['is_complete']:
            lines.append("  ✓ 数据完整，无缺失值")
        else:
            lines.append(f"  总缺失数: {missing['total_missing']}")
            for col, info in missing['missing_columns'].items():
                lines.append(f"    - {col}: {info['count']} ({info['percentage']}%)")

    # 重复
    if 'duplicates' in profile:
        dup = profile['duplicates']
        lines.append(f"\n【重复数据】")
        lines.append(f"  重复行数: {dup['duplicate_rows']}")
        lines.append(f"  时间重复: {dup['duplicate_datetimes']}")

    # 异常值
    if 'outliers' in profile:
        out = profile['outliers']
        lines.append(f"\n【异常值检测】")
        if out['has_outliers']:
            lines.append(f"  总异常数: {out['total_outliers']}")
            for col, info in list(out['outlier_columns'].items())[:5]:
                lines.append(f"    - {col}: {info['count']} ({info['percentage']}%)")
        else:
            lines.append("  ✓ 未检测到显著异常值")

    # 分布摘要
    if 'distribution' in profile and 'load' in profile['distribution']:
        dist = profile['distribution']['load']
        lines.extend([
            f"\n【负荷分布】",
            f"  均值: {dist['mean']:.2f}",
            f"  标准差: {dist['std']:.2f}",
            f"  最小值: {dist['min']:.2f}",
            f"  中位数: {dist['median']:.2f}",
            f"  最大值: {dist['max']:.2f}",
            f"  偏度: {dist['skewness']:.4f}",
            f"  峰度: {dist['kurtosis']:.4f}"
        ])

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # 演示数据探索
    logging.basicConfig(level=logging.INFO)

    from .generator import ensure_data_exists

    # 加载数据
    df = ensure_data_exists()

    # 探索数据
    profile = generate_data_profile(df)

    # 打印报告
    print(print_data_profile(profile))
