"""
LLM自然语言解释生成

核心亮点模块：
- 调用DeepSeek API生成自然语言解释
- 将数值预测转化为专业业务分析
- 支持不同详细程度
"""

import logging
import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from config import settings
from ..explainer.shap_analyzer import explain_prediction, get_feature_contribution
from ..explainer.report import generate_report, format_report_text

logger = logging.getLogger(__name__)


class LLMExplainer:
    """
    LLM解释器类

    负责调用LLM API生成自然语言解释
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化LLM解释器

        Args:
            api_key: DeepSeek API密钥（可选，默认从环境变量或配置获取）
        """
        self.api_key = api_key or os.environ.get(
            "DEEPSEEK_API_KEY",
            settings.DEEPSEEK_API_KEY
        )

        if not self.api_key:
            logger.warning(
                "未设置DEEPSEEK_API_KEY，将使用模拟解释。"
                "请设置环境变量或修改config.py中的DEEPSEEK_API_KEY"
            )
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.DEEPSEEK_BASE_URL
            )

        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS

    def _build_prompt(
        self,
        prediction: float,
        features: Dict,
        contributions: List[Dict],
        report: Dict,
        detail_level: str = "standard"
    ) -> str:
        """
        构建提示词

        Args:
            prediction: 预测负荷值
            features: 特征字典
            contributions: SHAP贡献列表
            report: 分析报告字典
            detail_level: 详细程度

        Returns:
            提示词字符串
        """
        # 获取top贡献
        top_features = contributions[:5]

        # 温度信息
        temperature = features.get("temperature", 20)
        humidity = features.get("humidity", 50)

        # 时间信息
        hour = features.get("hour", 12)
        day_of_week = features.get("day_of_week", 0)
        day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        day_name = day_names[day_of_week]

        # 判断日期类型
        is_workday = features.get("is_workday", 0) == 1
        is_weekend = features.get("is_weekend", False)
        is_holiday = features.get("is_holiday", False)

        if is_holiday:
            date_type = "节假日"
        elif is_weekend:
            date_type = "周末"
        elif is_workday:
            date_type = "工作日"
        else:
            date_type = "普通日"

        # 季节
        season = features.get("season", "unknown")
        season_names = {
            "spring": "春季",
            "summer": "夏季",
            "autumn": "秋季",
            "winter": "冬季"
        }
        season_cn = season_names.get(season, season)

        # 构建特征描述
        feature_desc = []
        for c in top_features:
            feat = c["feature"]
            value = c["feature_value"]
            shap = c["shap_value"]

            feat_cn = {
                "temperature": "温度",
                "humidity": "湿度",
                "load_lag_1h": "前1小时负荷",
                "load_lag_24h": "前24小时负荷",
                "load_lag_168h": "上周同期负荷",
                "hour": "小时",
                "day_of_week": "星期",
                "is_workday": "工作日效应",
                "thermal_comfort_index": "热舒适指数",
                "heating_cooling_index": "冷热指数",
            }.get(feat, feat)

            direction = "增加" if shap > 0 else "减少"
            feature_desc.append(
                f"- {feat_cn}: {direction}预测负荷 {abs(shap):.1f} MW "
                f"(当前值: {value:.1f})"
            )

        # 根据详细程度调整输出要求
        if detail_level == "brief":
            output_req = "生成1-2段简洁的分析"
        elif detail_level == "detailed":
            output_req = "生成详细的5段以上分析，包含原因分析、影响量化、趋势判断、风险提示、建议措施"
        else:
            output_req = "生成3-4段标准分析，包含原因分析、影响量化、趋势判断、建议"

        prompt = f"""你是一位资深电力负荷预测分析师，负责为每一次预测生成专业、可理解的自然语言解释。

## 背景信息

**预测结果**: {prediction:.2f} MW

**时间信息**:
- 日期: {day_name}
- 时段: {hour}:00
- 日期类型: {date_type}
- 季节: {season_cn}

**气象信息**:
- 温度: {temperature:.1f}°C
- 湿度: {humidity:.0f}%

**主要影响因素 (SHAP分析)**:
{chr(10).join(feature_desc)}

## 任务要求

{output_req}

## 输出格式

直接输出分析内容，使用中文，不要使用markdown格式的报告模板或表格。

请开始分析："""

        return prompt

    def _generate_fallback_explanation(
        self,
        prediction: float,
        features: Dict,
        contributions: List[Dict],
        report: Dict,
        detail_level: str = "standard"
    ) -> str:
        """
        生成备用解释（当API不可用时）

        Args:
            prediction: 预测负荷值
            features: 特征字典
            contributions: SHAP贡献列表
            report: 分析报告字典
            detail_level: 详细程度

        Returns:
            模拟的解释文本
        """
        # 获取top贡献
        top_features = contributions[:3]

        # 提取关键信息
        temperature = features.get("temperature", 20)
        hour = features.get("hour", 12)
        day_of_week = features.get("day_of_week", 0)
        day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        day_name = day_names[day_of_week]

        # 总贡献
        total_pos = sum(c["shap_value"] for c in contributions if c["shap_value"] > 0)
        total_neg = sum(abs(c["shap_value"]) for c in contributions if c["shap_value"] < 0)

        # 构建解释
        explanations = []

        # 基础分析
        base = f"{day_name} {hour}:00 预测负荷为 {prediction:.2f} MW。"
        explanations.append(base)

        # 主要因素分析
        if top_features:
            main_factor = top_features[0]
            feat = main_factor["feature"]
            shap = main_factor["shap_value"]

            feat_cn = {
                "temperature": "高温",
                "humidity": "高湿度",
                "load_lag_1h": "前日负荷惯性",
                "load_lag_24h": "昨日同期负荷",
                "load_lag_168h": "上周同期负荷",
                "is_workday": "工作日效应",
            }.get(feat, "该因素")

            direction = "推动" if shap > 0 else "拉低"
            explanations.append(
                f"主要受{feat_cn}影响，{direction}预测值约 {abs(shap):.1f} MW。"
            )

        # 温度影响
        if temperature > 30:
            explanations.append(
                f"当日气温较高({temperature:.1f}°C)，制冷负荷需求明显增加。"
            )
        elif temperature < 10:
            explanations.append(
                f"当日气温较低({temperature:.1f}°C)，制热负荷需求显著。"
            )

        # 工作日效应
        if features.get("is_workday", 0) == 1:
            explanations.append("工作日工业和商业用电需求较为旺盛。")

        # 滞后影响
        lag_1h = features.get("load_lag_1h")
        if lag_1h:
            lag_effect = sum(
                c["shap_value"] for c in contributions
                if "lag" in c["feature"]
            )
            if abs(lag_effect) > 50:
                explanations.append(
                    f"历史负荷惯性贡献约 {lag_effect:.1f} MW。"
                )

        # 建议
        explanations.append(
            f"综合各因素，预计峰值负荷将达到 {prediction:.2f} MW，建议持续关注气象变化。"
        )

        if detail_level == "detailed":
            explanations.insert(2, "各特征贡献度分析显示，正向贡献总计约 {:.1f} MW，负向贡献约 {:.1f} MW。".format(
                total_pos, total_neg
            ))
            explanations.insert(3, "预测置信区间反映模型对当前输入的确定性水平，在极端天气条件下不确定性可能增加。")

        return " ".join(explanations)

    def explain(
        self,
        prediction: float,
        features: Dict,
        contributions: List[Dict],
        report: Dict,
        detail_level: str = "standard"
    ) -> str:
        """
        生成解释

        Args:
            prediction: 预测负荷值
            features: 特征字典
            contributions: SHAP贡献列表
            report: 分析报告字典
            detail_level: 详细程度

        Returns:
            自然语言解释
        """
        # 如果没有API密钥，使用备用解释
        if not self.client:
            logger.info("使用备用解释（无API密钥）")
            return self._generate_fallback_explanation(
                prediction, features, contributions, report, detail_level
            )

        # 构建提示词
        prompt = self._build_prompt(
            prediction, features, contributions, report, detail_level
        )

        try:
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的电力负荷预测分析师，擅长用通俗易懂的语言解释复杂的预测结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            explanation = response.choices[0].message.content

            return explanation

        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            logger.info("回退到备用解释")
            return self._generate_fallback_explanation(
                prediction, features, contributions, report, detail_level
            )


# 全局解释器实例
_explainer: Optional[LLMExplainer] = None


def get_explainer() -> LLMExplainer:
    """获取LLM解释器实例（带缓存）"""
    global _explainer
    if _explainer is None:
        _explainer = LLMExplainer()
    return _explainer


def generate_explanation_sync(
    prediction: float,
    features: Dict,
    contributions: Optional[List[Dict]] = None,
    detail_level: str = "standard"
) -> str:
    """
    同步生成解释（便捷函数）

    Args:
        prediction: 预测负荷值
        features: 特征字典
        contributions: SHAP贡献列表（可选，会自动计算）
        detail_level: 详细程度

    Returns:
        自然语言解释
    """
    # 如果没有提供SHAP贡献，自动计算
    if contributions is None:
        contributions = explain_prediction(prediction, features)

    # 生成报告
    report = generate_report(prediction, features, contributions, detail_level)

    # 获取解释器
    explainer = get_explainer()

    # 生成解释
    return explainer.explain(
        prediction=prediction,
        features=features,
        contributions=contributions,
        report=report,
        detail_level=detail_level
    )


async def generate_explanation(
    prediction: float,
    features: Dict,
    contributions: Optional[List[Dict]] = None,
    detail_level: str = "standard"
) -> str:
    """
    异步生成解释（主要接口）

    Args:
        prediction: 预测负荷值
        features: 特征字典
        contributions: SHAP贡献列表（可选）
        detail_level: 详细程度

    Returns:
        自然语言解释
    """
    # 复用同步版本
    return generate_explanation_sync(
        prediction, features, contributions, detail_level
    )


# 同步别名
generate_explanation = generate_explanation_sync


if __name__ == "__main__":
    # 演示解释生成
    logging.basicConfig(level=logging.INFO)

    # 示例特征
    features = {
        "hour": 14,
        "day_of_week": 2,
        "temperature": 35.0,
        "humidity": 60,
        "is_workday": 1,
        "is_weekend": False,
        "is_holiday": False,
        "season": "summer",
        "month": 6,
        "temperature_squared": 1225.0,
        "load_lag_1h": 2700.0,
        "load_lag_24h": 2600.0,
        "load_lag_168h": 2500.0,
        "thermal_comfort_index": 10.0,
        "heating_cooling_index": 19.5,
        "apparent_temperature": 38.5,
        "hour_sin": 0.975,
        "hour_cos": -0.222,
        "day_of_week_sin": 0.974,
        "day_of_week_cos": -0.225,
        "month_sin": 0.5,
        "month_cos": 0.866,
        "day_of_month": 18,
        "day_of_month_sin": 0.485,
        "day_of_month_cos": 0.875,
        "temp_workday_interaction": 35.0,
        "temp_humidity_interaction": 21.0,
        "heat_humidity_index": 35.5,
        "load_same_hour_last_week": 2500.0,
        "load_change_1h": 0.02,
        "load_change_24h": 0.05,
        "load_ma_3h": 2680.0,
        "load_ma_24h": 2650.0,
    }

    # SHAP分析
    contributions = explain_prediction(None, features)

    # 生成解释
    explanation = generate_explanation(
        prediction=2850.5,
        features=features,
        contributions=contributions,
        detail_level="standard"
    )

    print("\n" + "=" * 60)
    print("LLM自然语言解释")
    print("=" * 60)
    print(explanation)
