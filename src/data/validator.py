"""
数据验证器

提供数据质量验证功能。

功能：
- 列名验证
- 数据类型验证
- 数据范围验证
- 缺失值验证
- 重复值验证
- 时间连续性验证
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据验证器

    验证数据的完整性、一致性和有效性。
    """

    def __init__(self, df: pd.DataFrame, datetime_col: str = "datetime"):
        """
        初始化数据验证器

        Args:
            df: 要验证的DataFrame
            datetime_col: 日期时间列名
        """
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }

    def validate_columns(
        self,
        required_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None
    ) -> 'DataValidator':
        """
        验证列名

        Args:
            required_columns: 必需列列表
            optional_columns: 可选列列表

        Returns:
            self（链式调用）
        """
        if required_columns is None:
            required_columns = ['datetime', 'load', 'temperature', 'humidity']

        missing_required = [col for col in required_columns if col not in self.df.columns]

        if missing_required:
            self.validation_results['errors'].append(
                f"缺少必需列: {missing_required}"
            )
        else:
            self.validation_results['passed'].append(
                f"所有必需列存在: {required_columns}"
            )

        if optional_columns:
            missing_optional = [col for col in optional_columns if col not in self.df.columns]
            if missing_optional:
                self.validation_results['warnings'].append(
                    f"缺少可选列: {missing_optional}"
                )

        return self

    def validate_dtypes(
        self,
        dtype_rules: Optional[Dict[str, Union[type, str, List[str]]]] = None
    ) -> 'DataValidator':
        """
        验证数据类型

        Args:
            dtype_rules: 数据类型规则 {'列名': 期望类型或类型列表}

        Returns:
            self（链式调用）
        """
        if dtype_rules is None:
            dtype_rules = {
                'datetime': ['datetime64[ns]', 'datetime64'],
                'load': ['float64', 'float32', 'int64', 'int32'],
                'temperature': ['float64', 'float32', 'int64', 'int32'],
                'humidity': ['float64', 'float32', 'int64', 'int32'],
            }

        for col, expected_types in dtype_rules.items():
            if col not in self.df.columns:
                continue

            actual_dtype = str(self.df[col].dtype)

            if isinstance(expected_types, str):
                expected_types = [expected_types]

            if actual_dtype not in expected_types:
                self.validation_results['warnings'].append(
                    f"列 '{col}' 类型为 {actual_dtype}，期望 {expected_types}"
                )
            else:
                self.validation_results['passed'].append(
                    f"列 '{col}' 类型正确: {actual_dtype}"
                )

        return self

    def validate_ranges(
        self,
        range_rules: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> 'DataValidator':
        """
        验证数据范围

        Args:
            range_rules: 范围规则 {'列名': (最小值, 最大值)}

        Returns:
            self（链式调用）
        """
        if range_rules is None:
            range_rules = {
                'load': (0, 100000),  # 电力负荷通常在0-100GW
                'temperature': (-50, 60),  # 温度范围
                'humidity': (0, 100),  # 湿度百分比
                'hour': (0, 23),  # 小时
                'day_of_week': (0, 6),  # 星期几
            }

        for col, (min_val, max_val) in range_rules.items():
            if col not in self.df.columns:
                continue

            data = self.df[col].dropna()

            if len(data) == 0:
                self.validation_results['warnings'].append(
                    f"列 '{col}' 无有效数据"
                )
                continue

            out_of_range = data[(data < min_val) | (data > max_val)]

            if len(out_of_range) > 0:
                self.validation_results['errors'].append(
                    f"列 '{col}' 有 {len(out_of_range)} 个值超出范围 [{min_val}, {max_val}]"
                )
            else:
                self.validation_results['passed'].append(
                    f"列 '{col}' 范围检查通过: [{data.min()}, {data.max()}]"
                )

        return self

    def validate_missing_values(
        self,
        max_missing_percent: float = 10.0
    ) -> 'DataValidator':
        """
        验证缺失值

        Args:
            max_missing_percent: 允许的最大缺失百分比

        Returns:
            self（链式调用）
        """
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percent = missing_count / len(self.df) * 100

            if missing_count > 0:
                if missing_percent > max_missing_percent:
                    self.validation_results['errors'].append(
                        f"列 '{col}' 缺失值过多: {missing_count} ({missing_percent:.2f}%)"
                    )
                else:
                    self.validation_results['warnings'].append(
                        f"列 '{col}' 存在缺失值: {missing_count} ({missing_percent:.2f}%)"
                    )
            else:
                self.validation_results['passed'].append(
                    f"列 '{col}' 无缺失值"
                )

        return self

    def validate_duplicates(
        self,
        subset: Optional[List[str]] = None
    ) -> 'DataValidator':
        """
        验证重复值

        Args:
            subset: 用于检查重复的列列表

        Returns:
            self（链式调用）
        """
        if subset is None:
            # 默认检查时间列重复
            subset = [self.datetime_col] if self.datetime_col in self.df.columns else None

        if subset:
            duplicates = self.df.duplicated(subset=subset).sum()

            if duplicates > 0:
                self.validation_results['errors'].append(
                    f"存在 {duplicates} 条重复记录 (基于: {subset})"
                )
            else:
                self.validation_results['passed'].append(
                    f"无重复记录 (基于: {subset})"
                )

        # 检查完全重复
        full_duplicates = self.df.duplicated().sum()
        if full_duplicates > 0:
            self.validation_results['warnings'].append(
                f"存在 {full_duplicates} 条完全重复的行"
            )

        return self

    def validate_time_continuity(
        self,
        expected_freq_hours: int = 1,
        max_gap_hours: Optional[int] = None
    ) -> 'DataValidator':
        """
        验证时间连续性

        Args:
            expected_freq_hours: 预期时间间隔（小时）
            max_gap_hours: 最大允许间隔

        Returns:
            self（链式调用）
        """
        if self.datetime_col not in self.df.columns:
            self.validation_results['warnings'].append(
                "缺少datetime列，无法检查时间连续性"
            )
            return self

        # 确保datetime类型
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.datetime_col]):
            self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])

        # 按时间排序
        df_sorted = self.df.sort_values(self.datetime_col)
        time_diffs = df_sorted[self.datetime_col].diff()

        # 计算期望间隔
        expected_delta = timedelta(hours=expected_freq_hours)

        # 找出时间跳跃
        gaps = time_diffs[time_diffs > timedelta(hours=max_gap_hours or expected_freq_hours * 2)]

        if len(gaps) > 0:
            self.validation_results['warnings'].append(
                f"检测到 {len(gaps)} 个时间跳跃，最大间隔: {gaps.max()}"
            )
        else:
            self.validation_results['passed'].append(
                "时间连续性检查通过"
            )

        return self

    def validate_consistency(self) -> 'DataValidator':
        """
        验证数据一致性

        - 工作日与周末不应同时为True
        - holiday列与周末逻辑一致性
        - hour与datetime小时一致性

        Returns:
            self（链式调用）
        """
        # 检查is_weekend和is_holiday一致性
        if 'is_weekend' in self.df.columns and 'is_holiday' in self.df.columns:
            # 周末可能是节假日，但不应该是非节假日
            weekend_holiday_mismatch = (
                (self.df['is_weekend']) & (~self.df['is_holiday'])
            ).sum()

            if weekend_holiday_mismatch == 0:
                self.validation_results['passed'].append(
                    "周末与节假日逻辑一致"
                )
            else:
                self.validation_results['warnings'].append(
                    f"{weekend_holiday_mismatch} 条周末记录标记为非节假日"
                )

        # 检查hour与datetime一致性
        if 'hour' in self.df.columns and self.datetime_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.datetime_col]):
                self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])

            hour_from_datetime = self.df[self.datetime_col].dt.hour
            hour_mismatch = (self.df['hour'] != hour_from_datetime).sum()

            if hour_mismatch > 0:
                self.validation_results['warnings'].append(
                    f"{hour_mismatch} 条记录的hour列与datetime不一致"
                )
            else:
                self.validation_results['passed'].append(
                    "hour列与datetime一致"
                )

        # 检查day_of_week与datetime一致性
        if 'day_of_week' in self.df.columns and self.datetime_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.datetime_col]):
                self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])

            dow_from_datetime = self.df[self.datetime_col].dt.dayofweek
            dow_mismatch = (self.df['day_of_week'] != dow_from_datetime).sum()

            if dow_mismatch > 0:
                self.validation_results['warnings'].append(
                    f"{dow_mismatch} 条记录的day_of_week与datetime不一致"
                )
            else:
                self.validation_results['passed'].append(
                    "day_of_week与datetime一致"
                )

        return self

    def validate_all(
        self,
        required_columns: Optional[List[str]] = None,
        check_continuity: bool = True,
        check_consistency: bool = True
    ) -> Dict:
        """
        执行所有验证

        Args:
            required_columns: 必需列
            check_continuity: 是否检查时间连续性
            check_consistency: 是否检查数据一致性

        Returns:
            验证结果字典
        """
        # 重置结果
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }

        # 执行各项验证
        self.validate_columns(required_columns)
        self.validate_dtypes()
        self.validate_ranges()
        self.validate_missing_values()
        self.validate_duplicates()

        if check_continuity:
            self.validate_time_continuity()

        if check_consistency:
            self.validate_consistency()

        # 汇总结果
        passed_count = len(self.validation_results['passed'])
        warning_count = len(self.validation_results['warnings'])
        error_count = len(self.validation_results['errors'])

        summary = {
            'passed': passed_count,
            'warnings': warning_count,
            'errors': error_count,
            'is_valid': error_count == 0,
            'details': self.validation_results
        }

        return summary

    def get_report(self) -> str:
        """
        获取格式化的验证报告

        Returns:
            报告字符串
        """
        summary = {
            'passed': len(self.validation_results['passed']),
            'warnings': len(self.validation_results['warnings']),
            'errors': len(self.validation_results['errors']),
            'is_valid': len(self.validation_results['errors']) == 0
        }

        lines = [
            "=" * 60,
            "数据验证报告",
            "=" * 60,
            f"\n验证结果: {'✓ 通过' if summary['is_valid'] else '✗ 失败'}",
            f"通过项: {summary['passed']}",
            f"警告项: {summary['warnings']}",
            f"错误项: {summary['errors']}",
        ]

        if self.validation_results['errors']:
            lines.append("\n【错误】")
            for error in self.validation_results['errors']:
                lines.append(f"  ✗ {error}")

        if self.validation_results['warnings']:
            lines.append("\n【警告】")
            for warning in self.validation_results['warnings']:
                lines.append(f"  ⚠ {warning}")

        if self.validation_results['passed'] and summary['warnings'] == 0 and summary['errors'] == 0:
            lines.append("\n【通过项】")
            for passed in self.validation_results['passed'][:5]:
                lines.append(f"  ✓ {passed}")

        lines.append("=" * 60)

        return "\n".join(lines)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    datetime_col: str = "datetime"
) -> Dict:
    """
    验证DataFrame

    Args:
        df: 要验证的DataFrame
        required_columns: 必需列列表
        datetime_col: 日期时间列名

    Returns:
        验证结果字典
    """
    validator = DataValidator(df, datetime_col)
    return validator.validate_all(required_columns)


def validate_and_report(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    datetime_col: str = "datetime"
) -> Tuple[Dict, str]:
    """
    验证DataFrame并生成报告

    Args:
        df: 要验证的DataFrame
        required_columns: 必需列列表
        datetime_col: 日期时间列名

    Returns:
        (验证结果, 报告字符串)
    """
    validator = DataValidator(df, datetime_col)
    result = validator.validate_all(required_columns)
    report = validator.get_report()
    return result, report


if __name__ == "__main__":
    # 演示数据验证
    logging.basicConfig(level=logging.INFO)

    from .generator import ensure_data_exists

    # 加载数据
    df = ensure_data_exists()

    # 验证数据
    result, report = validate_and_report(df)

    print(report)

    print(f"\n最终结果: {'通过' if result['is_valid'] else '失败'}")
