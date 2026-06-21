"""
异常检测与根因分析模块

功能：
- 扫描预测偏差大的时段
- 调用LLM分析根因
- 生成结构化异常报告
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from openai import OpenAI

from config import settings
from ..explainer.shap_analyzer import explain_prediction, get_feature_contribution

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRecord:
    """异常记录"""

    timestamp: str
    actual: float
    predicted: float
    error_pct: float
    severity: str  # "low", "medium", "high", "critical"
    root_cause: str = ""
    shap_highlights: List[Dict] = field(default_factory=list)
    features: Dict = field(default_factory=dict)


class AnomalyDetector:
    """
    异常检测器

    负责扫描预测偏差并调用LLM进行根因分析
    """

    def __init__(
        self,
        threshold_pct: float = 15.0,
        max_results: int = 20,
        api_key: Optional[str] = None
    ):
        """
        初始化异常检测器

        Args:
            threshold_pct: 异常偏差阈值(%)
            max_results: 最大返回异常数
            api_key: DeepSeek API密钥（可选）
        """
        self.threshold_pct = threshold_pct
        self.max_results = max_results

        # 初始化LLM客户端
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.DEEPSEEK_BASE_URL
            )
        else:
            self.client = None
            logger.warning("未设置DEEPSEEK_API_KEY，根因分析将使用规则推断")

        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.LLM_TEMPERATURE

    def scan_deviations(
        self,
        predictions_df: pd.DataFrame,
        threshold_pct: Optional[float] = None
    ) -> List[AnomalyRecord]:
        """
        扫描预测偏差大的时段

        Args:
            predictions_df: 包含actual和predicted_load列的DataFrame
            threshold_pct: 临时覆盖阈值（可选）

        Returns:
            异常记录列表
        """
        threshold = threshold_pct or self.threshold_pct

        if "actual_load" not in predictions_df.columns:
            logger.warning("未找到actual_load列，跳过异常检测")
            return []

        if "predicted_load" not in predictions_df.columns:
            logger.warning("未找到predicted_load列，跳过异常检测")
            return []

        anomalies = []
        datetime_col = "datetime" if "datetime" in predictions_df.columns else predictions_df.index.name or "index"

        for idx, row in predictions_df.iterrows():
            actual = row["actual_load"]
            predicted = row["predicted_load"]

            if predicted == 0 or pd.isna(actual) or pd.isna(predicted):
                continue

            error = actual - predicted
            error_pct = abs(error) / predicted * 100

            if error_pct > threshold:
                # 确定严重程度
                if error_pct > 50:
                    severity = "critical"
                elif error_pct > 30:
                    severity = "high"
                elif error_pct > threshold:
                    severity = "medium" if error_pct > threshold else "low"
                else:
                    severity = "low"

                # 获取时间戳
                if hasattr(row.get("datetime"), "strftime"):
                    timestamp = str(row["datetime"])
                elif isinstance(idx, (pd.Timestamp, str)):
                    timestamp = str(idx)
                else:
                    timestamp = f"row_{idx}"

                # 提取特征（排除预测相关列）
                features = {
                    k: v for k, v in row.items()
                    if k not in ["actual_load", "predicted_load", "load", datetime_col]
                    and not pd.isna(v)
                }

                anomalies.append(AnomalyRecord(
                    timestamp=timestamp,
                    actual=float(actual),
                    predicted=float(predicted),
                    error_pct=float(error_pct),
                    severity=severity,
                    features=dict(features)
                ))

        # 按严重程度排序
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda x: severity_order[x.severity])

        # 限制返回数量
        return anomalies[:self.max_results]

    def analyze_root_cause(
        self,
        anomaly: AnomalyRecord,
        contributions_df: Optional[pd.DataFrame] = None
    ) -> AnomalyRecord:
        """
        调用LLM分析异常根因

        Args:
            anomaly: 异常记录
            contributions_df: 该时段的SHAP贡献DataFrame（可选）

        Returns:
            更新后的异常记录（含根因分析）
        """
        # 计算该时段的SHAP贡献
        if contributions_df is not None:
            # 从DataFrame中提取该时段的贡献
            shap_data = self._extract_shap_for_timestamp(
                contributions_df, anomaly.timestamp
            )
        else:
            # 根据特征计算SHAP
            shap_data = self._calculate_shap(anomaly.features)

        anomaly.shap_highlights = shap_data

        # 调用LLM分析
        if self.client:
            root_cause = self._llm_analyze_root_cause(anomaly, shap_data)
        else:
            root_cause = self._rule_based_analysis(anomaly, shap_data)

        anomaly.root_cause = root_cause

        return anomaly

    def _extract_shap_for_timestamp(
        self,
        contributions_df: pd.DataFrame,
        timestamp: str
    ) -> List[Dict]:
        """从SHAP DataFrame中提取特定时间戳的数据"""
        if "datetime" in contributions_df.columns:
            mask = contributions_df["datetime"].astype(str) == timestamp
            row = contributions_df[mask]
        elif hasattr(contributions_df.index, "astype"):
            mask = contributions_df.index.astype(str) == timestamp
            row = contributions_df[mask]
        else:
            return []

        if row.empty:
            return []

        shap_list = []
        for col in contributions_df.columns:
            if col == "datetime":
                continue
            val = row.iloc[0][col]
            if not pd.isna(val):
                shap_list.append({
                    "feature": col,
                    "shap_value": float(val)
                })

        # 按绝对值排序
        shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return shap_list[:10]

    def _calculate_shap(self, features: Dict) -> List[Dict]:
        """根据特征计算SHAP贡献"""
        try:
            contributions = explain_prediction(None, features)
            return contributions[:10]
        except Exception as e:
            logger.warning(f"SHAP计算失败: {e}")
            return []

    def _llm_analyze_root_cause(
        self,
        anomaly: AnomalyRecord,
        shap_data: List[Dict]
    ) -> str:
        """使用LLM分析根因"""
        # 构建SHAP Top5描述
        shap_desc = []
        for i, s in enumerate(shap_data[:5], 1):
            feat_name = self._get_feature_cn(s["feature"])
            direction = "正向贡献" if s["shap_value"] > 0 else "负向贡献"
            shap_desc.append(f"{i}. {feat_name}: {direction} {abs(s['shap_value']):.1f} MW")

        # 构建特征上下文
        feature_context = self._build_feature_context(anomaly.features)

        prompt = f"""你是一位资深的电力负荷预测分析师，负责分析预测偏差的根因。

## 异常信息
- 时间: {anomaly.timestamp}
- 实际负荷: {anomaly.actual:.2f} MW
- 预测负荷: {anomaly.predicted:.2f} MW
- 偏差率: {anomaly.error_pct:.1f}%
- 严重程度: {anomaly.severity}

## 特征上下文
{feature_context}

## SHAP Top5贡献
{chr(10).join(shap_desc)}

## 分析任务
请分析这个预测偏差的可能根因，要求：
1. 识别主要的影响因素
2. 推断可能的外部原因（如突发天气变化、重大事件等）
3. 给出简明的根因总结（1-2句话）

请用中文回答，直接输出分析结果，不要使用markdown格式："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的电力负荷预测分析师，擅长根因分析。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500,
                request_timeout=30,
                max_retries=1
            )

            result = response.choices[0].message.content
            return result.strip()

        except Exception as e:
            logger.error(f"LLM根因分析失败: {e}")
            return self._rule_based_analysis(anomaly, shap_data)

    def _build_feature_context(self, features: Dict) -> str:
        """构建特征上下文"""
        context_items = []

        # 关键特征
        key_features = [
            ("temperature", "温度", "°C"),
            ("humidity", "湿度", "%"),
            ("hour", "小时", "时"),
            ("day_of_week", "星期", ""),
            ("is_workday", "工作日", ""),
            ("is_holiday", "节假日", ""),
            ("season", "季节", ""),
        ]

        for feat_key, feat_name, unit in key_features:
            if feat_key in features:
                val = features[feat_key]
                if not pd.isna(val):
                    context_items.append(f"- {feat_name}: {val}{unit}")

        # 历史负荷
        lag_features = ["load_lag_1h", "load_lag_24h", "load_lag_168h"]
        for lf in lag_features:
            if lf in features and not pd.isna(features.get(lf)):
                lag_name = {
                    "load_lag_1h": "前1小时负荷",
                    "load_lag_24h": "前24小时负荷",
                    "load_lag_168h": "上周同期负荷"
                }.get(lf, lf)
                context_items.append(f"- {lag_name}: {features[lf]:.1f} MW")

        return chr(10).join(context_items) if context_items else "（无详细特征数据）"

    def _rule_based_analysis(
        self,
        anomaly: AnomalyRecord,
        shap_data: List[Dict]
    ) -> str:
        """基于规则的分析（当LLM不可用时）"""
        causes = []

        features = anomaly.features

        # 温度分析
        temp = features.get("temperature", 20)
        if temp > 38:
            causes.append("极端高温导致制冷负荷超预期")
        elif temp < 2:
            causes.append("极端低温导致取暖负荷超预期")

        # 工作日分析
        is_workday = features.get("is_workday", 0)
        is_holiday = features.get("is_holiday", False)

        if is_holiday:
            causes.append("节假日活动模式与预测模型假设不符")
        elif is_workday == 0 and anomaly.actual > anomaly.predicted:
            causes.append("非工作日负荷模式异常")

        # 滞后特征分析
        lag_1h = features.get("load_lag_1h")
        if lag_1h and abs(anomaly.actual - lag_1h) > abs(anomaly.predicted - lag_1h) * 1.5:
            causes.append("负荷突变未能在预测中及时反映")

        # SHAP Top贡献
        if shap_data:
            top_shap = shap_data[0]
            feat_name = self._get_feature_cn(top_shap["feature"])
            causes.append(f"主要受{feat_name}异常影响")

        return "；".join(causes) if causes else "根因未能明确确定"

    def _get_feature_cn(self, feature: str) -> str:
        """获取特征中文名"""
        name_map = {
            "temperature": "温度",
            "humidity": "湿度",
            "load_lag_1h": "前1小时负荷",
            "load_lag_24h": "前24小时负荷",
            "load_lag_168h": "上周同期负荷",
            "is_workday": "工作日效应",
            "is_holiday": "节假日效应",
            "hour": "时段",
            "day_of_week": "星期",
            "thermal_comfort_index": "热舒适指数",
            "heating_cooling_index": "冷热指数",
            "apparent_temperature": "体感温度",
            "temp_workday_interaction": "温度工作日交互",
            "heat_humidity_index": "高温高湿指数",
        }
        return name_map.get(feature, feature)

    def format_anomaly_report(
        self,
        anomalies: List[AnomalyRecord]
    ) -> str:
        """
        格式化异常报告

        Args:
            anomalies: 异常记录列表

        Returns:
            格式化的报告文本
        """
        if not anomalies:
            return "## 异常检测报告\n\n未检测到显著异常。"

        lines = ["## 异常检测报告\n"]
        lines.append(f"\n共检测到 **{len(anomalies)}** 个异常记录：\n")

        # 按严重程度分组
        severity_groups = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        for a in anomalies:
            severity_groups.get(a.severity, severity_groups["low"]).append(a)

        # 输出各严重程度的异常
        severity_labels = {
            "critical": "🔴 严重异常",
            "high": "🟠 高异常",
            "medium": "🟡 中异常",
            "low": "🟢 轻度异常"
        }

        for severity, label in severity_labels.items():
            group = severity_groups[severity]
            if not group:
                continue

            lines.append(f"\n### {label} ({len(group)}项)\n")

            for a in group[:5]:  # 每组最多显示5个
                lines.append(f"**{a.timestamp}**")
                lines.append(f"- 实际: {a.actual:.2f} MW | 预测: {a.predicted:.2f} MW | 偏差: {a.error_pct:.1f}%")

                if a.root_cause:
                    lines.append(f"- 根因: {a.root_cause}")

                if a.shap_highlights:
                    top_shap = a.shap_highlights[:3]
                    shap_str = ", ".join([
                        f"{self._get_feature_cn(s['feature'])}({s['shap_value']:.1f}MW)"
                        for s in top_shap
                    ])
                    lines.append(f"- 主因: {shap_str}")

                lines.append("")

        return "\n".join(lines)


def _clean_json_response(text: str) -> str:
    """清理LLM返回的markdown代码块包裹"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 演示异常检测
    np.random.seed(42)

    # 模拟数据
    dates = pd.date_range("2023-06-15 00:00", periods=48, freq="H")
    data = {
        "datetime": dates,
        "actual_load": 2500 + np.random.randn(48) * 100,
        "predicted_load": 2500 + np.random.randn(48) * 80,
        "temperature": 25 + np.sin(np.arange(48) * np.pi / 12) * 8,
        "humidity": 50 + np.random.randn(48) * 10,
        "hour": [h % 24 for h in range(48)],
        "is_workday": [1] * 40 + [0] * 8,
    }

    # 人为制造几个异常
    data["actual_load"][5] = 3200  # 高异常
    data["actual_load"][20] = 1800  # 低异常
    data["actual_load"][35] = 3500  # 严重异常

    df = pd.DataFrame(data)

    # 异常检测
    detector = AnomalyDetector(threshold_pct=15.0)
    anomalies = detector.scan_deviations(df)

    print("\n" + "=" * 60)
    print("异常检测结果")
    print("=" * 60)

    # 分析每个异常的根因
    for a in anomalies[:5]:
        detector.analyze_root_cause(a)
        print(f"\n时间: {a.timestamp}")
        print(f"实际: {a.actual:.2f} | 预测: {a.predicted:.2f} | 偏差: {a.error_pct:.1f}%")
        print(f"根因: {a.root_cause}")

    # 生成报告
    report = detector.format_anomaly_report(anomalies)
    print("\n" + "=" * 60)
    print("异常报告")
    print("=" * 60)
    print(report)
