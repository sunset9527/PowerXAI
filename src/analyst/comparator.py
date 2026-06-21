"""
多时段对比分析器

功能：
- 对比两个时段的预测+SHAP分析
- 分析差异原因
- 归因分析
- 多粒度对比（日/周/月/年）
- LLM智能解读
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from config import settings
from ..explainer.shap_analyzer import explain_prediction, get_feature_contribution
from ..explainer.report import generate_report

logger = logging.getLogger(__name__)


@dataclass
class FeatureChange:
    """特征变化"""

    feature: str
    value1: float
    value2: float
    shap1: float
    shap2: float
    value_change: float
    shap_change: float
    impact_type: str  # "driver", "reducer", "neutral"


@dataclass
class ComparisonResult:
    """对比结果"""

    period1_info: Dict
    period2_info: Dict
    prediction_diff: float
    prediction_diff_pct: float
    feature_changes: List[FeatureChange]
    key_drivers: List[str]  # 主要推动因素
    key_reducers: List[str]  # 主要降低因素
    explanation: str


class Comparator:
    """
    对比分析器
    """

    def __init__(
        self,
        api_key: Optional[str] = None
    ):
        """
        初始化对比分析器

        Args:
            api_key: DeepSeek API密钥（可选）
        """
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.DEEPSEEK_BASE_URL
            )
        else:
            self.client = None

        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.LLM_TEMPERATURE

    def compare(
        self,
        features1: Dict,
        features2: Dict,
        prediction1: float,
        prediction2: float,
        contributions1: Optional[List[Dict]] = None,
        contributions2: Optional[List[Dict]] = None
    ) -> ComparisonResult:
        """
        对比两个时段的特征和预测

        Args:
            features1: 时段1的特征
            features2: 时段2的特征
            prediction1: 时段1的预测
            prediction2: 时段2的预测
            contributions1: 时段1的SHAP贡献（可选）
            contributions2: 时段2的SHAP贡献（可选）

        Returns:
            对比结果
        """
        # 计算SHAP贡献
        if contributions1 is None:
            contributions1 = explain_prediction(prediction1, features1)
        if contributions2 is None:
            contributions2 = explain_prediction(prediction2, features2)

        # 构建贡献字典
        contrib_dict1 = {c["feature"]: c for c in contributions1}
        contrib_dict2 = {c["feature"]: c for c in contributions2}

        # 获取共同特征
        all_features = set(contrib_dict1.keys()) | set(contrib_dict2.keys())

        # 分析特征变化
        feature_changes = []
        for feat in all_features:
            c1 = contrib_dict1.get(feat, {"shap_value": 0, "feature_value": 0})
            c2 = contrib_dict2.get(feat, {"shap_value": 0, "feature_value": 0})

            value1 = features1.get(feat, c1.get("feature_value", 0))
            value2 = features2.get(feat, c2.get("feature_value", 0))

            shap1 = c1.get("shap_value", 0)
            shap2 = c2.get("shap_value", 0)

            value_change = value2 - value1
            shap_change = shap2 - shap1

            # 判断影响类型
            if shap_change > 20:  # 超过20MW的变化
                impact_type = "driver"
            elif shap_change < -20:
                impact_type = "reducer"
            else:
                impact_type = "neutral"

            feature_changes.append(FeatureChange(
                feature=feat,
                value1=float(value1),
                value2=float(value2),
                shap1=float(shap1),
                shap2=float(shap2),
                value_change=float(value_change),
                shap_change=float(shap_change),
                impact_type=impact_type
            ))

        # 按变化量排序
        feature_changes.sort(key=lambda x: abs(x.shap_change), reverse=True)

        # 识别主要驱动和降低因素
        key_drivers = [
            fc.feature for fc in feature_changes
            if fc.impact_type == "driver"
        ][:3]

        key_reducers = [
            fc.feature for fc in feature_changes
            if fc.impact_type == "reducer"
        ][:3]

        # 生成解释
        explanation = self._generate_explanation(
            features1, features2, prediction1, prediction2,
            feature_changes, key_drivers, key_reducers
        )

        return ComparisonResult(
            period1_info=self._get_period_summary(features1, prediction1),
            period2_info=self._get_period_summary(features2, prediction2),
            prediction_diff=round(prediction2 - prediction1, 2),
            prediction_diff_pct=round(
                (prediction2 - prediction1) / prediction1 * 100, 2
            ) if prediction1 != 0 else 0,
            feature_changes=feature_changes,
            key_drivers=key_drivers,
            key_reducers=key_reducers,
            explanation=explanation
        )

    def compare_multi_granularity(
        self,
        df: pd.DataFrame,
        period1_range: Tuple[str, str],
        period2_range: Tuple[str, str],
        granularity: str = "daily"
    ) -> Dict:
        """
        多粒度对比分析

        Args:
            df: 完整数据DataFrame
            period1_range: 时段1起止日期 (start, end)
            period2_range: 时段2起止日期 (start, end)
            granularity: 粒度 (daily/weekly/monthly/yearly)

        Returns:
            多粒度对比结果
        """
        # 解析日期范围
        p1_start, p1_end = pd.to_datetime(period1_range[0]), pd.to_datetime(period1_range[1])
        p2_start, p2_end = pd.to_datetime(period2_range[0]), pd.to_datetime(period2_range[1])

        # 筛选数据
        df_p1 = df[(df["datetime"] >= p1_start) & (df["datetime"] <= p1_end)].copy()
        df_p2 = df[(df["datetime"] >= p2_start) & (df["datetime"] <= p2_end)].copy()

        # 根据粒度进行聚合
        if granularity == "daily":
            # 按小时聚合
            df_p1["period"] = df_p1["datetime"].dt.date
            df_p2["period"] = df_p2["datetime"].dt.date
            agg_func = "mean"
        elif granularity == "weekly":
            # 按周聚合
            df_p1["period"] = df_p1["datetime"].dt.to_period("W").astype(str)
            df_p2["period"] = df_p2["datetime"].dt.to_period("W").astype(str)
            agg_func = "mean"
        elif granularity == "monthly":
            # 按月聚合
            df_p1["period"] = df_p1["datetime"].dt.to_period("M").astype(str)
            df_p2["period"] = df_p2["datetime"].dt.to_period("M").astype(str)
            agg_func = "mean"
        elif granularity == "yearly":
            # 按年聚合
            df_p1["period"] = df_p1["datetime"].dt.year
            df_p2["period"] = df_p2["datetime"].dt.year
            agg_func = "mean"
        else:
            raise ValueError(f"不支持的粒度: {granularity}")

        # 计算各时段的统计信息
        load_col = "load" if "load" in df.columns else "actual_load"

        if load_col not in df.columns:
            logger.warning(f"未找到负荷列 {load_col}")
            return {}

        # 时段1统计
        p1_stats = {
            "count": len(df_p1),
            "mean": float(df_p1[load_col].mean()),
            "std": float(df_p1[load_col].std()),
            "min": float(df_p1[load_col].min()),
            "max": float(df_p1[load_col].max()),
            "median": float(df_p1[load_col].median()),
            "sum": float(df_p1[load_col].sum()),
        }

        # 时段2统计
        p2_stats = {
            "count": len(df_p2),
            "mean": float(df_p2[load_col].mean()),
            "std": float(df_p2[load_col].std()),
            "min": float(df_p2[load_col].min()),
            "max": float(df_p2[load_col].max()),
            "median": float(df_p2[load_col].median()),
            "sum": float(df_p2[load_col].sum()),
        }

        # 计算差异
        stats_diff = {
            "mean_diff": round(p2_stats["mean"] - p1_stats["mean"], 2),
            "mean_diff_pct": round(
                (p2_stats["mean"] - p1_stats["mean"]) / p1_stats["mean"] * 100, 2
            ) if p1_stats["mean"] != 0 else 0,
            "max_diff": round(p2_stats["max"] - p1_stats["max"], 2),
            "min_diff": round(p2_stats["min"] - p1_stats["min"], 2),
            "sum_diff": round(p2_stats["sum"] - p1_stats["sum"], 2),
        }

        # 按小时/日期分析差异模式
        hourly_diff = self._analyze_hourly_pattern(df_p1, df_p2, load_col, granularity)

        # 按日期类型分析
        workday_diff = self._analyze_workday_pattern(df_p1, df_p2, load_col)

        return {
            "granularity": granularity,
            "period1": {
                "range": period1_range,
                "stats": p1_stats
            },
            "period2": {
                "range": period2_range,
                "stats": p2_stats
            },
            "differences": stats_diff,
            "hourly_pattern": hourly_diff,
            "workday_pattern": workday_diff
        }

    def _analyze_hourly_pattern(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        load_col: str,
        granularity: str
    ) -> Dict:
        """分析小时模式差异"""
        if "hour" not in df1.columns or "hour" not in df2.columns:
            return {}

        # 计算各小时平均值
        hour_stats1 = df1.groupby("hour")[load_col].mean()
        hour_stats2 = df2.groupby("hour")[load_col].mean()

        # 计算差异
        hourly_diff = {}
        for hour in range(24):
            if hour in hour_stats1 and hour in hour_stats2:
                diff = hour_stats2[hour] - hour_stats1[hour]
                hourly_diff[hour] = {
                    "p1_avg": float(hour_stats1[hour]),
                    "p2_avg": float(hour_stats2[hour]),
                    "diff": float(diff),
                    "diff_pct": float(diff / hour_stats1[hour] * 100) if hour_stats1[hour] != 0 else 0
                }

        # 找出差异最大的时段
        if hourly_diff:
            max_diff_hour = max(hourly_diff.items(), key=lambda x: abs(x[1]["diff"]))
            min_diff_hour = min(hourly_diff.items(), key=lambda x: abs(x[1]["diff"]))

            return {
                "hourly_details": hourly_diff,
                "max_diff_hour": {
                    "hour": max_diff_hour[0],
                    "diff": max_diff_hour[1]["diff"],
                    "diff_pct": max_diff_hour[1]["diff_pct"]
                },
                "min_diff_hour": {
                    "hour": min_diff_hour[0],
                    "diff": min_diff_hour[1]["diff"],
                    "diff_pct": min_diff_hour[1]["diff_pct"]
                }
            }

        return {}

    def _analyze_workday_pattern(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        load_col: str
    ) -> Dict:
        """分析工作日模式差异"""
        if "is_workday" not in df1.columns or "is_workday" not in df2.columns:
            return {}

        patterns = {}

        # 工作日对比
        wd1 = df1[df1["is_workday"] == 1][load_col].mean()
        wd2 = df2[df2["is_workday"] == 1][load_col].mean()

        if not pd.isna(wd1) and not pd.isna(wd2):
            patterns["workday"] = {
                "p1_avg": float(wd1),
                "p2_avg": float(wd2),
                "diff": float(wd2 - wd1),
                "diff_pct": float((wd2 - wd1) / wd1 * 100) if wd1 != 0 else 0
            }

        # 周末对比
        we1 = df1[df1["is_workday"] == 0][load_col].mean()
        we2 = df2[df2["is_workday"] == 0][load_col].mean()

        if not pd.isna(we1) and not pd.isna(we2):
            patterns["weekend"] = {
                "p1_avg": float(we1),
                "p2_avg": float(we2),
                "diff": float(we2 - we1),
                "diff_pct": float((we2 - we1) / we1 * 100) if we1 != 0 else 0
            }

        return patterns

    def generate_llm_interpretation(
        self,
        comparison_result: Dict,
        df: pd.DataFrame
    ) -> str:
        """
        调用LLM解读对比结果

        Args:
            comparison_result: compare_multi_granularity的返回结果
            df: 原始数据DataFrame（用于补充上下文）

        Returns:
            LLM生成的解读文本
        """
        if not self.client:
            return self._rule_based_interpretation(comparison_result)

        # 构建对比摘要
        p1_stats = comparison_result.get("period1", {}).get("stats", {})
        p2_stats = comparison_result.get("period2", {}).get("stats", {})
        diff = comparison_result.get("differences", {})

        # 构建特征变化描述
        hourly = comparison_result.get("hourly_pattern", {})
        workday = comparison_result.get("workday_pattern", {})

        prompt = f"""你是一位专业的电力负荷预测分析师，负责解读两个时段的对比分析结果。

## 对比摘要
时段1: {comparison_result.get('period1', {}).get('range', 'N/A')}
- 平均负荷: {p1_stats.get('mean', 0):.2f} MW
- 最大负荷: {p1_stats.get('max', 0):.2f} MW
- 最小负荷: {p1_stats.get('min', 0):.2f} MW
- 样本数: {p1_stats.get('count', 0)}

时段2: {comparison_result.get('period2', {}).get('range', 'N/A')}
- 平均负荷: {p2_stats.get('mean', 0):.2f} MW
- 最大负荷: {p2_stats.get('max', 0):.2f} MW
- 最小负荷: {p2_stats.get('min', 0):.2f} MW
- 样本数: {p2_stats.get('count', 0)}

## 差异分析
- 平均负荷差异: {diff.get('mean_diff', 0):.2f} MW ({diff.get('mean_diff_pct', 0):.1f}%)
- 最大负荷差异: {diff.get('max_diff', 0):.2f} MW
- 最小负荷差异: {diff.get('min_diff', 0):.2f} MW

## 时段模式分析
{self._format_hourly_pattern(hourly)}

## 日期类型分析
{self._format_workday_pattern(workday)}

## 分析任务
请提供：
1. 两个时段的主要差异总结（2-3句话）
2. 主要驱动因素推断
3. 运营建议（1-2条）

请用中文回答，直接输出分析内容："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的电力负荷预测分析师，擅长数据分析解读。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=800,
                request_timeout=30,
                max_retries=1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM解读生成失败: {e}")
            return self._rule_based_interpretation(comparison_result)

    def _format_hourly_pattern(self, hourly: Dict) -> str:
        """格式化小时模式分析"""
        if not hourly:
            return "（无可用小时数据）"

        lines = []
        if "max_diff_hour" in hourly:
            max_info = hourly["max_diff_hour"]
            lines.append(f"- 差异最大时段: {max_info['hour']}时，差异 {max_info['diff']:.1f} MW ({max_info['diff_pct']:.1f}%)")

        if "min_diff_hour" in hourly:
            min_info = hourly["min_diff_hour"]
            lines.append(f"- 差异最小时段: {min_info['hour']}时，差异 {min_info['diff']:.1f} MW ({min_info['diff_pct']:.1f}%)")

        return "\n".join(lines) if lines else "（无可用小时数据）"

    def _format_workday_pattern(self, workday: Dict) -> str:
        """格式化工作日模式分析"""
        if not workday:
            return "（无可用工作日数据）"

        lines = []
        if "workday" in workday:
            w = workday["workday"]
            lines.append(f"- 工作日: 时段1平均 {w['p1_avg']:.1f} MW → 时段2平均 {w['p2_avg']:.1f} MW (变化 {w['diff']:.1f} MW, {w['diff_pct']:.1f}%)")

        if "weekend" in workday:
            w = workday["weekend"]
            lines.append(f"- 周末: 时段1平均 {w['p1_avg']:.1f} MW → 时段2平均 {w['p2_avg']:.1f} MW (变化 {w['diff']:.1f} MW, {w['diff_pct']:.1f}%)")

        return "\n".join(lines) if lines else "（无可用工作日数据）"

    def _rule_based_interpretation(self, comparison_result: Dict) -> str:
        """基于规则的解读（当LLM不可用时）"""
        diff = comparison_result.get("differences", {})
        mean_diff = diff.get("mean_diff", 0)
        mean_diff_pct = diff.get("mean_diff_pct", 0)

        lines = []

        # 整体趋势
        if mean_diff > 0:
            lines.append(f"时段2的平均负荷比时段1高 {mean_diff:.1f} MW ({mean_diff_pct:.1f}%)。")
        else:
            lines.append(f"时段2的平均负荷比时段1低 {abs(mean_diff):.1f} MW ({abs(mean_diff_pct):.1f}%)。")

        # 变化幅度评估
        if abs(mean_diff_pct) > 20:
            lines.append("负荷变化幅度较大，建议关注是否存在特殊事件或异常。")
        elif abs(mean_diff_pct) > 10:
            lines.append("负荷有一定变化，可能受季节性或气象因素影响。")
        else:
            lines.append("负荷变化在正常范围内，运营状态稳定。")

        # 工作日模式
        workday = comparison_result.get("workday_pattern", {})
        if workday:
            if "workday" in workday:
                wd = workday["workday"]
                if wd["diff"] > 50:
                    lines.append("工作日负荷显著上升，可能反映了经济增长或产业结构变化。")
                elif wd["diff"] < -50:
                    lines.append("工作日负荷明显下降，可能与节能政策或产业转移有关。")

        return " ".join(lines)

    def _get_period_summary(self, features: Dict, prediction: float) -> Dict:
        """获取时段摘要"""
        return {
            "prediction": round(prediction, 2),
            "hour": features.get("hour", 0),
            "day_of_week": features.get("day_of_week", 0),
            "temperature": features.get("temperature", 20),
            "humidity": features.get("humidity", 50),
            "is_workday": features.get("is_workday", 0) == 1,
            "is_weekend": features.get("is_weekend", False),
        }

    def _generate_explanation(
        self,
        features1: Dict,
        features2: Dict,
        prediction1: float,
        prediction2: float,
        feature_changes: List[FeatureChange],
        key_drivers: List[str],
        key_reducers: List[str]
    ) -> str:
        """生成对比解释"""
        diff = prediction2 - prediction1
        diff_pct = (diff / prediction1 * 100) if prediction1 != 0 else 0

        lines = []

        # 基础对比
        if diff > 0:
            lines.append(
                f"时段2的预测负荷({prediction2:.2f} MW)比时段1({prediction1:.2f} MW)"
                f"高 {diff:.2f} MW ({diff_pct:.1f}%)"
            )
        else:
            lines.append(
                f"时段2的预测负荷({prediction2:.2f} MW)比时段1({prediction1:.2f} MW)"
                f"低 {abs(diff):.2f} MW ({abs(diff_pct):.1f}%)"
            )

        # 主要因素
        if key_drivers:
            driver_names = [self._get_feature_cn(f) for f in key_drivers]
            lines.append(f"主要推动因素: {', '.join(driver_names)}")

        if key_reducers:
            reducer_names = [self._get_feature_cn(f) for f in key_reducers]
            lines.append(f"主要降低因素: {', '.join(reducer_names)}")

        # 温度变化
        temp1 = features1.get("temperature", 0)
        temp2 = features2.get("temperature", 0)
        temp_diff = temp2 - temp1

        if abs(temp_diff) > 2:
            if temp_diff > 0:
                lines.append(
                    f"温度升高 {temp_diff:.1f}°C (从{temp1:.1f}°C到{temp2:.1f}°C)"
                    "可能增加制冷负荷"
                )
            else:
                lines.append(
                    f"温度降低 {abs(temp_diff):.1f}°C (从{temp1:.1f}°C到{temp2:.1f}°C)"
                    "可能减少制冷负荷"
                )

        # 工作日变化
        workday1 = features1.get("is_workday", 0) == 1
        workday2 = features2.get("is_workday", 0) == 1

        if workday1 != workday2:
            if workday2:
                lines.append("日期类型从非工作日变为工作日，负荷通常上升")
            else:
                lines.append("日期类型从工作日变为非工作日，负荷通常下降")

        return " ".join(lines)

    def _get_feature_cn(self, feature: str) -> str:
        """获取特征中文名"""
        name_map = {
            "temperature": "温度",
            "humidity": "湿度",
            "load_lag_1h": "前1小时负荷",
            "load_lag_24h": "前24小时负荷",
            "load_lag_168h": "上周同期负荷",
            "is_workday": "工作日效应",
            "hour": "时段",
            "thermal_comfort_index": "热舒适指数",
            "heating_cooling_index": "冷热指数",
        }
        return name_map.get(feature, feature)


def compare_periods(
    features1: Dict,
    features2: Dict,
    prediction1: float,
    prediction2: float,
    contributions1: Optional[List[Dict]] = None,
    contributions2: Optional[List[Dict]] = None
) -> ComparisonResult:
    """对比两个时段"""
    comparator = Comparator()
    return comparator.compare(
        features1, features2, prediction1, prediction2,
        contributions1, contributions2
    )


def compare_with_yesterday(
    features: Dict,
    yesterday_features: Dict,
    prediction: float,
    yesterday_prediction: float
) -> ComparisonResult:
    """
    与昨日对比

    Args:
        features: 今日特征
        yesterday_features: 昨日特征
        prediction: 今日预测
        yesterday_prediction: 昨日预测

    Returns:
        对比结果
    """
    return compare_periods(
        yesterday_features, features,
        yesterday_prediction, prediction
    )


def compare_with_last_week(
    features: Dict,
    last_week_features: Dict,
    prediction: float,
    last_week_prediction: float
) -> ComparisonResult:
    """
    与上周同期对比

    Args:
        features: 当前时段特征
        last_week_features: 上周同期特征
        prediction: 当前预测
        last_week_prediction: 上周同期预测

    Returns:
        对比结果
    """
    return compare_periods(
        last_week_features, features,
        last_week_prediction, prediction
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 演示多粒度对比
    np.random.seed(42)
    n = 2000

    dates = pd.date_range("2023-01-01", periods=n, freq="H")
    data = {
        "datetime": dates,
        "load": 2500 + np.sin(np.arange(n) * np.pi / 12) * 300 + np.random.randn(n) * 100,
        "temperature": 20 + np.sin(np.arange(n) * np.pi / 12) * 10,
        "humidity": 50 + np.random.randn(n) * 15,
        "hour": [h % 24 for h in range(n)],
        "is_workday": [1 if d.weekday() < 5 else 0 for d in dates],
    }

    df = pd.DataFrame(data)

    # 多粒度对比
    comparator = Comparator()

    # 月度对比
    result = comparator.compare_multi_granularity(
        df,
        period1_range=("2023-01-01", "2023-01-31"),
        period2_range=("2023-06-01", "2023-06-30"),
        granularity="daily"
    )

    print("\n" + "=" * 60)
    print("多粒度对比分析")
    print("=" * 60)

    print(f"\n时段1 (2023-01): 平均负荷 {result['period1']['stats']['mean']:.2f} MW")
    print(f"时段2 (2023-06): 平均负荷 {result['period2']['stats']['mean']:.2f} MW")
    print(f"差异: {result['differences']['mean_diff']:.2f} MW ({result['differences']['mean_diff_pct']:.1f}%)")

    # LLM解读
    interpretation = comparator.generate_llm_interpretation(result, df)
    print(f"\n解读:\n{interpretation}")
