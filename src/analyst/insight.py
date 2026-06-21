"""
洞察提取引擎

功能：
- 自动检测趋势：连续N天负荷上升/下降
- 异常告警：实际值偏离预测超过阈值
- 关联分析：温度变化与负荷变化的相关性
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrendInsight:
    """趋势洞察"""

    trend_type: str  # "increasing", "decreasing", "stable"
    duration: int  # 持续时间（小时）
    start_value: float
    end_value: float
    change_rate: float  # 变化率
    change_absolute: float  # 绝对变化量
    confidence: float  # 置信度


@dataclass
class AnomalyInsight:
    """异常洞察"""

    anomaly_type: str  # "high_error", "low_error", "extreme_value"
    timestamp: str
    actual_value: float
    predicted_value: float
    error: float
    error_percentage: float
    severity: str  # "low", "medium", "high"
    possible_causes: List[str]


@dataclass
class CorrelationInsight:
    """关联洞察"""

    factor1: str
    factor2: str
    correlation: float
    lag: int  # 滞后时间
    description: str


class InsightEngine:
    """
    洞察提取引擎
    """

    def __init__(
        self,
        trend_threshold: float = 0.02,
        anomaly_threshold: float = 0.15,
        min_trend_duration: int = 6
    ):
        """
        初始化洞察引擎

        Args:
            trend_threshold: 趋势判定阈值（变化率）
            anomaly_threshold: 异常判定阈值（误差百分比）
            min_trend_duration: 最小趋势持续时间（小时）
        """
        self.trend_threshold = trend_threshold
        self.anomaly_threshold = anomaly_threshold
        self.min_trend_duration = min_trend_duration

    def detect_trends(
        self,
        df: pd.DataFrame,
        value_col: str = "load",
        datetime_col: str = "datetime",
        direction: Optional[str] = None
    ) -> List[TrendInsight]:
        """
        检测趋势

        Args:
            df: 数据DataFrame
            value_col: 数值列名
            datetime_col: 时间列名
            direction: 筛选方向 ("increasing", "decreasing", None表示全部)

        Returns:
            趋势洞察列表
        """
        df = df.sort_values(datetime_col).copy()

        # 计算变化率
        df["change"] = df[value_col].diff()
        df["change_rate"] = df[value_col].pct_change()

        # 识别趋势段
        trends = []
        current_trend = []
        current_direction = None

        for idx, row in df.iterrows():
            if pd.isna(row["change_rate"]):
                continue

            # 判断趋势方向
            if row["change_rate"] > self.trend_threshold:
                trend_dir = "increasing"
            elif row["change_rate"] < -self.trend_threshold:
                trend_dir = "decreasing"
            else:
                trend_dir = "stable"

            if trend_dir == current_direction:
                current_trend.append(row)
            else:
                # 保存之前的趋势
                if len(current_trend) >= self.min_trend_duration:
                    trend_insight = self._create_trend_insight(
                        current_trend, current_direction
                    )
                    if trend_insight:
                        trends.append(trend_insight)

                # 开始新趋势
                current_trend = [row]
                current_direction = trend_dir

        # 处理最后一个趋势
        if len(current_trend) >= self.min_trend_duration:
            trend_insight = self._create_trend_insight(
                current_trend, current_direction
            )
            if trend_insight:
                trends.append(trend_insight)

        # 筛选方向
        if direction:
            trends = [t for t in trends if t.trend_type == direction]

        logger.info(f"检测到 {len(trends)} 个趋势")

        return trends

    def _create_trend_insight(
        self,
        rows: List[pd.Series],
        direction: str
    ) -> Optional[TrendInsight]:
        """
        创建趋势洞察

        Args:
            rows: 趋势段的数据行
            direction: 趋势方向

        Returns:
            TrendInsight对象
        """
        if not rows:
            return None

        start_value = rows[0]["load"]
        end_value = rows[-1]["load"]
        change_absolute = end_value - start_value
        change_rate = change_absolute / start_value if start_value != 0 else 0

        # 计算置信度（基于持续时间和变化幅度）
        duration = len(rows)
        confidence = min(1.0, duration / 24 * 0.5 + abs(change_rate) * 5)

        return TrendInsight(
            trend_type=direction,
            duration=duration,
            start_value=float(start_value),
            end_value=float(end_value),
            change_rate=float(change_rate),
            change_absolute=float(change_absolute),
            confidence=float(confidence),
        )

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        actual_col: str = "load",
        predicted_col: str = "predicted_load",
        datetime_col: str = "datetime"
    ) -> List[AnomalyInsight]:
        """
        检测异常

        Args:
            df: 数据DataFrame（需要包含actual和predicted列）
            actual_col: 实际值列名
            predicted_col: 预测值列名
            datetime_col: 时间列名

        Returns:
            异常洞察列表
        """
        if predicted_col not in df.columns:
            logger.warning(f"预测列 {predicted_col} 不存在，跳过异常检测")
            return []

        anomalies = []

        for idx, row in df.iterrows():
            actual = row[actual_col]
            predicted = row[predicted_col]

            if predicted == 0:
                continue

            error = actual - predicted
            error_pct = abs(error) / predicted

            # 判断是否为异常
            if error_pct > self.anomaly_threshold:
                # 判断严重程度
                if error_pct > 0.3:
                    severity = "high"
                elif error_pct > 0.2:
                    severity = "medium"
                else:
                    severity = "low"

                # 分析可能原因
                causes = self._analyze_anomaly_causes(row, error, error_pct)

                anomalies.append(AnomalyInsight(
                    anomaly_type="high_error" if error > 0 else "low_error",
                    timestamp=str(row[datetime_col]),
                    actual_value=float(actual),
                    predicted_value=float(predicted),
                    error=float(error),
                    error_percentage=float(error_pct * 100),
                    severity=severity,
                    possible_causes=causes
                ))

        # 按严重程度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda x: severity_order[x.severity])

        logger.info(f"检测到 {len(anomalies)} 个异常")

        return anomalies

    def _analyze_anomaly_causes(
        self,
        row: pd.Series,
        error: float,
        error_pct: float
    ) -> List[str]:
        """
        分析异常可能原因

        Args:
            row: 数据行
            error: 误差
            error_pct: 误差百分比

        Returns:
            可能原因列表
        """
        causes = []

        # 温度影响
        if "temperature" in row:
            temp = row["temperature"]
            if temp > 35:
                causes.append("极端高温可能导致空调负荷超预期")
            elif temp < 5:
                causes.append("极端低温可能导致取暖负荷超预期")

        # 滞后特征缺失
        if "load_lag_1h" in row and pd.isna(row.get("load_lag_1h")):
            causes.append("历史负荷数据缺失，影响预测准确性")

        # 工作日/节假日
        if row.get("is_holiday"):
            causes.append("节假日活动模式与常规工作日不同")
        elif row.get("is_weekend"):
            causes.append("周末负荷模式与工作日存在差异")

        # 季节性事件
        if "month" in row:
            month = row["month"]
            if month in [6, 7, 8] and error > 0:
                causes.append("夏季制冷需求持续增加")
            elif month in [12, 1, 2] and error > 0:
                causes.append("冬季取暖需求持续增加")

        if not causes:
            causes.append("未知原因，建议检查数据质量")

        return causes

    def find_correlations(
        self,
        df: pd.DataFrame,
        factor1_col: str,
        factor2_col: str,
        max_lag: int = 24
    ) -> List[CorrelationInsight]:
        """
        寻找两个因素之间的相关性

        Args:
            df: 数据DataFrame
            factor1_col: 因素1列名
            factor2_col: 因素2列名
            max_lag: 最大滞后时间

        Returns:
            关联洞察列表
        """
        if factor1_col not in df.columns or factor2_col not in df.columns:
            logger.warning(f"指定的因素列不存在")
            return []

        correlations = []

        # 尝试不同的滞后时间
        for lag in range(0, max_lag + 1):
            if lag == 0:
                corr = df[factor1_col].corr(df[factor2_col])
            else:
                corr = df[factor1_col].corr(df[factor2_col].shift(lag))

            if not pd.isna(corr) and abs(corr) > 0.5:
                # 特征名映射
                f1_name = self._get_feature_name(factor1_col)
                f2_name = self._get_feature_name(factor2_col)

                descriptions = [
                    f"{f1_name}与{f2_name}存在较强{"正" if corr > 0 else "负"}相关",
                    f"滞后{lag}小时时相关性最强"
                ]

                correlations.append(CorrelationInsight(
                    factor1=factor1_col,
                    factor2=factor2_col,
                    correlation=float(corr),
                    lag=lag,
                    description="; ".join(descriptions)
                ))

        # 按相关性绝对值排序
        correlations.sort(key=lambda x: abs(x.correlation), reverse=True)

        logger.info(f"发现 {len(correlations)} 个显著相关性")

        return correlations[:5]  # 最多返回5个

    def _get_feature_name(self, col: str) -> str:
        """获取特征的中文名称"""
        name_map = {
            "temperature": "温度",
            "humidity": "湿度",
            "load": "负荷",
            "load_lag_1h": "前1小时负荷",
            "load_lag_24h": "前24小时负荷",
            "load_lag_168h": "上周同期负荷",
            "apparent_temperature": "体感温度",
        }
        return name_map.get(col, col)

    def generate_summary(
        self,
        trends: List[TrendInsight],
        anomalies: List[AnomalyInsight],
        correlations: List[CorrelationInsight]
    ) -> str:
        """
        生成洞察摘要

        Args:
            trends: 趋势列表
            anomalies: 异常列表
            correlations: 关联列表

        Returns:
            摘要文本
        """
        lines = []

        # 趋势总结
        if trends:
            lines.append(f"\n📈 趋势检测 ({len(trends)}项):")
            for t in trends[:3]:
                trend_name = {
                    "increasing": "上升",
                    "decreasing": "下降",
                    "stable": "平稳"
                }.get(t.trend_type, t.trend_type)

                lines.append(
                    f"  - 检测到{trend_name}趋势，持续{t.duration}小时，"
                    f"变化率{abs(t.change_rate)*100:.1f}%"
                )

        # 异常总结
        if anomalies:
            high_count = sum(1 for a in anomalies if a.severity == "high")
            medium_count = sum(1 for a in anomalies if a.severity == "medium")

            lines.append(f"\n⚠️ 异常检测 ({len(anomalies)}项):")
            if high_count > 0:
                lines.append(f"  - 严重异常: {high_count}项")
            if medium_count > 0:
                lines.append(f"  - 中度异常: {medium_count}项")

        # 关联总结
        if correlations:
            lines.append(f"\n🔗 关联发现 ({len(correlations)}项):")
            for c in correlations[:3]:
                lines.append(f"  - {c.description}")

        return "\n".join(lines) if lines else "未检测到显著洞察"


# 便捷函数
def detect_trends(
    df: pd.DataFrame,
    value_col: str = "load",
    datetime_col: str = "datetime",
    direction: Optional[str] = None
) -> List[TrendInsight]:
    """检测趋势"""
    engine = InsightEngine()
    return engine.detect_trends(df, value_col, datetime_col, direction)


def detect_anomalies(
    df: pd.DataFrame,
    actual_col: str = "load",
    predicted_col: str = "predicted_load",
    datetime_col: str = "datetime"
) -> List[AnomalyInsight]:
    """检测异常"""
    engine = InsightEngine()
    return engine.detect_anomalies(df, actual_col, predicted_col, datetime_col)


def find_correlations(
    df: pd.DataFrame,
    factor1_col: str,
    factor2_col: str,
    max_lag: int = 24
) -> List[CorrelationInsight]:
    """寻找相关性"""
    engine = InsightEngine()
    return engine.find_correlations(df, factor1_col, factor2_col, max_lag)


if __name__ == "__main__":
    # 演示洞察提取
    logging.basicConfig(level=logging.INFO)

    # 模拟数据
    np.random.seed(42)
    n = 200

    data = {
        "datetime": pd.date_range("2023-06-01", periods=n, freq="H"),
        "load": 2000 + np.cumsum(np.random.randn(n) * 20),
        "temperature": 25 + np.sin(np.arange(n) * np.pi / 12) * 10,
    }

    df = pd.DataFrame(data)

    # 添加预测列（模拟）
    df["predicted_load"] = df["load"] + np.random.randn(n) * 50

    # 趋势检测
    engine = InsightEngine()
    trends = engine.detect_trends(df)

    # 异常检测
    anomalies = engine.detect_anomalies(df)

    # 关联分析
    correlations = engine.find_correlations(
        df, "temperature", "load", max_lag=12
    )

    # 生成摘要
    summary = engine.generate_summary(trends, anomalies, correlations)

    print("\n" + "=" * 60)
    print("洞察提取结果")
    print("=" * 60)
    print(summary)
