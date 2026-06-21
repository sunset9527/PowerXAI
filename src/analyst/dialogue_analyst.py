"""
对话式分析模块

功能：
- 意图分类
- 上下文管理
- What-If场景分析
- 多轮对话支持
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from config import settings
from ..explainer.shap_analyzer import explain_prediction, get_feature_contribution
from .anomaly_detector import AnomalyDetector
from .comparator import Comparator
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """对话条目"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None
    result: Optional[Dict] = None


@dataclass
class AnalysisResult:
    """分析结果"""

    intent: str
    result: Any
    explanation: str
    suggestions: List[str] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)


class DialogueAnalyst:
    """
    对话式分析师

    支持多轮对话、意图识别和What-If场景分析
    """

    # 意图关键词映射
    INTENT_PATTERNS = {
        "anomaly": ["异常", "偏差", "不正常", "哪里出了问题", "为什么不对", "错误"],
        "comparison": ["对比", "比较", "和...比", "差异", "变化", "增长", "下降", "昨天", "上周", "上周同期"],
        "report": ["报告", "总结", "摘要", "汇总", "分析报告", "生成报告"],
        "attribution": ["原因", "为什么", "是什么导致的", "归因", "因素", "贡献", "影响", "驱动力"],
        "what_if": ["如果", "假如", "假设", "预测", "会", "将要", "将会", "会到多少"],
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_history: int = 5
    ):
        """
        初始化对话分析师

        Args:
            api_key: DeepSeek API密钥（可选）
            max_history: 最大保存对话历史轮数
        """
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.max_history = max_history or settings.DIALOGUE_MAX_HISTORY

        # 初始化LLM客户端
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.DEEPSEEK_BASE_URL
            )
        else:
            self.client = None
            logger.warning("未设置DEEPSEEK_API_KEY，部分功能可能受限")

        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.LLM_TEMPERATURE

        # 初始化组件
        self.anomaly_detector = AnomalyDetector(api_key=self.api_key)
        self.comparator = Comparator(api_key=self.api_key)
        self.report_generator = ReportGenerator(api_key=self.api_key)

        # 对话历史
        self.conversation_history: List[ConversationEntry] = []

    def classify_intent(self, query: str) -> str:
        """
        分类用户意图

        Args:
            query: 用户查询

        Returns:
            意图类型：anomaly/comparison/report/attribution/what_if
        """
        query_lower = query.lower()

        # 基于关键词匹配
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            scores[intent] = score

        # 返回最高分意图
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]

        # 默认归因为解释/归因
        return "attribution"

    def prepare_context(
        self,
        intent: str,
        query: str,
        data_cache: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        根据意图准备上下文数据

        Args:
            intent: 意图类型
            query: 用户查询
            data_cache: 数据缓存字典

        Returns:
            上下文数据
        """
        context = {"intent": intent}

        if intent == "anomaly":
            # 异常检测上下文
            if "predictions" in data_cache:
                df = data_cache["predictions"]
                anomalies = self.anomaly_detector.scan_deviations(df)
                context["anomalies"] = anomalies
                context["summary"] = self.anomaly_detector.format_anomaly_report(anomalies[:10])

        elif intent == "comparison":
            # 对比分析上下文
            if "data" in data_cache:
                df = data_cache["data"]
                # 尝试解析时间范围
                period1, period2 = self._parse_periods(query)
                if period1 and period2:
                    result = self.comparator.compare_multi_granularity(
                        df, period1, period2
                    )
                    context["comparison"] = result
                    context["interpretation"] = self.comparator.generate_llm_interpretation(
                        result, df
                    )

        elif intent == "report":
            # 报告生成上下文
            if "data" in data_cache:
                df = data_cache["data"]
                predictions_df = data_cache.get("predictions", df)
                context["date_range"] = self._parse_date_range(query)
                context["report_ready"] = True

        elif intent == "what_if":
            # What-If场景上下文
            context["scenario"] = self._parse_whatif_scenario(query)
            if "data" in data_cache:
                df = data_cache["data"]
                context["current_features"] = self._get_current_features(df)

        # 通用上下文
        if "data" in data_cache:
            df = data_cache["data"]
            context["stats"] = {
                "mean": float(df["load"].mean()) if "load" in df.columns else 0,
                "max": float(df["load"].max()) if "load" in df.columns else 0,
                "min": float(df["load"].min()) if "load" in df.columns else 0,
            }

        return context

    def analyze(
        self,
        query: str,
        conversation_history: Optional[List[ConversationEntry]] = None,
        data_cache: Optional[Dict[str, pd.DataFrame]] = None
    ) -> AnalysisResult:
        """
        完整分析流程

        Args:
            query: 用户查询
            conversation_history: 之前的对话历史（可选）
            data_cache: 数据缓存（可选）

        Returns:
            分析结果
        """
        # 分类意图
        intent = self.classify_intent(query)
        logger.info(f"识别意图: {intent}")

        # 准备上下文
        context = self.prepare_context(intent, query, data_cache or {})

        # 执行分析
        if intent == "anomaly":
            result = self._handle_anomaly_query(query, context)
        elif intent == "comparison":
            result = self._handle_comparison_query(query, context)
        elif intent == "report":
            result = self._handle_report_query(query, context, data_cache)
        elif intent == "what_if":
            result = self._handle_whatif_query(query, context, data_cache)
        else:  # attribution
            result = self._handle_attribution_query(query, context)

        # 添加到历史
        self._add_to_history("user", query, intent, context)
        self._add_to_history("assistant", result.explanation, intent, result.result)

        return result

    def follow_up(
        self,
        query: str,
        previous_result: AnalysisResult,
        data_cache: Optional[Dict[str, pd.DataFrame]] = None
    ) -> AnalysisResult:
        """
        追问处理

        Args:
            query: 追问内容
            previous_result: 之前的分析结果
            data_cache: 数据缓存

        Returns:
            新的分析结果
        """
        # 根据之前的意图和追问内容决定下一步
        prev_intent = previous_result.intent

        # 特殊处理：如果追问涉及What-If
        if any(p in query for p in ["如果", "假设", "假如"]):
            prev_intent = "what_if"

        # 特殊处理：如果追问涉及报告
        if any(p in query for p in ["报告", "导出", "生成"]):
            prev_intent = "report"

        # 重新分类
        intent = self.classify_intent(query)

        # 如果之前有相关上下文，尝试复用
        context = {"follow_up": True, "previous_intent": prev_intent}

        if prev_intent == "anomaly" and intent == "what_if":
            # 追问异常原因并想知道假设情况
            context["anomalies"] = previous_result.result
        elif prev_intent == "comparison" and intent == "what_if":
            # 追问对比并想知道假设情况
            context["comparison"] = previous_result.result

        # 执行分析
        if intent == "what_if":
            result = self._handle_whatif_query(query, context, data_cache)
        elif intent == "report":
            result = self._handle_report_query(query, context, data_cache)
        else:
            result = self._handle_general_query(query, context, data_cache)

        return result

    def _add_to_history(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        result: Optional[Dict] = None
    ):
        """添加到对话历史"""
        entry = ConversationEntry(
            role=role,
            content=content,
            intent=intent,
            result=result
        )
        self.conversation_history.append(entry)

        # 保持最大历史长度
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def _handle_anomaly_query(
        self,
        query: str,
        context: Dict
    ) -> AnalysisResult:
        """处理异常查询"""
        anomalies = context.get("anomalies", [])
        summary = context.get("summary", "")

        if not anomalies:
            explanation = "在当前数据范围内未检测到显著的预测偏差。"
        else:
            # 生成详细解释
            lines = [f"共检测到 {len(anomalies)} 个异常：\n"]
            for a in anomalies[:5]:
                lines.append(
                    f"- {a.timestamp}: 实际 {a.actual:.1f} MW，"
                    f"预测 {a.predicted:.1f} MW，偏差 {a.error_pct:.1f}%"
                )
                if a.root_cause:
                    lines.append(f"  根因: {a.root_cause}")
            explanation = "\n".join(lines)

        return AnalysisResult(
            intent="anomaly",
            result={"anomalies": [vars(a) for a in anomalies]},
            explanation=explanation,
            suggestions=["可以询问'为什么会出现这些异常'", "可以询问'如果温度升高会怎样'"]
        )

    def _handle_comparison_query(
        self,
        query: str,
        context: Dict
    ) -> AnalysisResult:
        """处理对比查询"""
        comparison = context.get("comparison")
        interpretation = context.get("interpretation", "")

        if not comparison:
            explanation = "需要指定两个时段进行对比，例如'对比6月和7月的负荷'。"
        else:
            diff = comparison.get("differences", {})
            explanation = f"""
时段对比分析结果：

时段1: {comparison.get('period1', {}).get('range', 'N/A')}
- 平均负荷: {comparison.get('period1', {}).get('stats', {}).get('mean', 0):.2f} MW

时段2: {comparison.get('period2', {}).get('range', 'N/A')}
- 平均负荷: {comparison.get('period2', {}).get('stats', {}).get('mean', 0):.2f} MW

差异: {diff.get('mean_diff', 0):.2f} MW ({diff.get('mean_diff_pct', 0):.1f}%)

{interpretation}
""".strip()

        return AnalysisResult(
            intent="comparison",
            result=comparison,
            explanation=explanation,
            suggestions=["可以询问'如果明天下雨会怎样'", "可以生成详细报告"]
        )

    def _handle_report_query(
        self,
        query: str,
        context: Dict,
        data_cache: Optional[Dict[str, pd.DataFrame]]
    ) -> AnalysisResult:
        """处理报告查询"""
        if not data_cache or "data" not in data_cache:
            explanation = "需要提供数据才能生成报告。"
            return AnalysisResult(
                intent="report",
                result=None,
                explanation=explanation,
                suggestions=["请先上传数据"]
            )

        df = data_cache["data"]
        predictions_df = data_cache.get("predictions", df)
        date_range = context.get("date_range", {})

        start_date = date_range.get("start", str(df["datetime"].min().date()) if "datetime" in df.columns else None)
        end_date = date_range.get("end", str(df["datetime"].max().date()) if "datetime" in df.columns else None)

        if not start_date or not end_date:
            # 尝试自动获取
            if "datetime" in df.columns:
                start_date = str(df["datetime"].min().date())
                end_date = str(df["datetime"].max().date())
            else:
                start_date = "未知"
                end_date = "未知"

        # 生成报告
        report = self.report_generator.generate_report(
            df, predictions_df, start_date, end_date
        )

        # 保存报告
        report_path = self.report_generator.export_markdown(report)

        explanation = f"""
报告已生成！

**报告概要:**
- 周期: {start_date} 至 {end_date}
- 摘要: {report['summary']}

**报告文件:** {report_path}

报告包含：
- 负荷统计摘要
- 预测准确性分析
- 异常时段检测
- 主要影响因素
- 运营建议
"""

        return AnalysisResult(
            intent="report",
            result={"report": report, "path": str(report_path)},
            explanation=explanation,
            suggestions=["可以询问特定时段的详细分析", "可以询问负荷变化的原因"]
        )

    def _handle_whatif_query(
        self,
        query: str,
        context: Dict,
        data_cache: Optional[Dict[str, pd.DataFrame]]
    ) -> AnalysisResult:
        """处理What-If场景查询"""
        scenario = context.get("scenario", {})
        current_features = context.get("current_features", {})

        if not scenario:
            explanation = "请提供假设条件，例如'如果明天温度是35°C会怎样'。"
            return AnalysisResult(
                intent="what_if",
                result=None,
                explanation=explanation
            )

        # 修改特征
        modified_features = current_features.copy()
        for key, value in scenario.items():
            modified_features[key] = value

        # 预测（使用SHAP模拟）
        try:
            prediction = self._simulate_prediction(modified_features)
            current_pred = context.get("stats", {}).get("mean", 2500)

            # 生成解释
            contributions = explain_prediction(prediction, modified_features)
            top_contrib = contributions[:3]

            explanation = f"""
**What-If 场景分析**

假设条件:
"""
            for key, value in scenario.items():
                feat_cn = self._get_feature_cn(key)
                explanation += f"- {feat_cn}: {value}\n"

            explanation += f"""
预测结果: **{prediction:.2f} MW**

对比基准: {current_pred:.2f} MW
变化量: {prediction - current_pred:+.2f} MW ({((prediction - current_pred) / current_pred * 100):+.1f}%)

主要影响因素:
"""
            for c in top_contrib:
                feat_cn = self._get_feature_cn(c["feature"])
                direction = "增加" if c["shap_value"] > 0 else "减少"
                explanation += f"- {feat_cn}: {direction} {abs(c['shap_value']):.1f} MW\n"

            # LLM解读
            if self.client:
                llm_interpretation = self._llm_interpret_scenario(
                    scenario, prediction, current_pred, top_contrib
                )
                explanation += f"\n{llm_interpretation}"

            result = {
                "scenario": scenario,
                "prediction": prediction,
                "baseline": current_pred,
                "contributions": contributions
            }

            return AnalysisResult(
                intent="what_if",
                result=result,
                explanation=explanation,
                suggestions=["可以调整假设条件继续分析", "可以生成报告保存分析结果"]
            )

        except Exception as e:
            logger.error(f"What-If预测失败: {e}")
            return AnalysisResult(
                intent="what_if",
                result=None,
                explanation=f"预测失败: {str(e)}"
            )

    def _handle_attribution_query(
        self,
        query: str,
        context: Dict
    ) -> AnalysisResult:
        """处理归因查询"""
        stats = context.get("stats", {})
        top_drivers = []

        # 生成归因解释
        if self.client:
            explanation = self._llm_explain_attribution(query, context)
        else:
            explanation = f"""
根据当前数据分析：

负荷统计:
- 平均负荷: {stats.get('mean', 0):.2f} MW
- 最大负荷: {stats.get('max', 0):.2f} MW
- 最小负荷: {stats.get('min', 0):.2f} MW

主要影响因素包括：
- 温度（夏季制冷/冬季取暖需求）
- 时段（白天高于夜间）
- 工作日效应（工作日高于周末）

如需详细归因分析，请提供具体时段的数据。
"""

        return AnalysisResult(
            intent="attribution",
            result={"stats": stats, "top_drivers": top_drivers},
            explanation=explanation,
            suggestions=["可以询问'为什么负荷上升了'", "可以询问'对比上周同期'"]
        )

    def _handle_general_query(
        self,
        query: str,
        context: Dict,
        data_cache: Optional[Dict[str, pd.DataFrame]]
    ) -> AnalysisResult:
        """处理通用查询"""
        # 使用LLM生成响应
        if self.client:
            explanation = self._llm_respond_to_query(query, context)
        else:
            explanation = "我理解您的查询，但需要更具体的信息才能提供准确的分析。"

        return AnalysisResult(
            intent="general",
            result=context,
            explanation=explanation
        )

    def _llm_interpret_scenario(
        self,
        scenario: Dict,
        prediction: float,
        baseline: float,
        contributions: List[Dict]
    ) -> str:
        """LLM解读What-If场景"""
        prompt = f"""作为电力负荷分析师，请解读以下What-If场景预测结果：

场景：{scenario}
预测负荷：{prediction:.2f} MW
基准负荷：{baseline:.2f} MW
变化：{prediction - baseline:+.2f} MW

Top贡献因素：
{chr(10).join([f"- {c['feature']}: {c['shap_value']:+.1f} MW" for c in contributions[:3]])}

请用2-3句话解读这个场景的业务意义：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的电力负荷预测分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=300,
                request_timeout=30,
                max_retries=1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM解读失败: {e}")
            return ""

    def _llm_explain_attribution(
        self,
        query: str,
        context: Dict
    ) -> str:
        """LLM解释归因"""
        stats = context.get("stats", {})
        prompt = f"""用户询问：{query}

当前数据统计：
- 平均负荷：{stats.get('mean', 0):.2f} MW
- 最大负荷：{stats.get('max', 0):.2f} MW
- 最小负荷：{stats.get('min', 0):.2f} MW

请回答用户的问题，用专业但易懂的语言解释归因分析结果。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的电力负荷预测分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500,
                request_timeout=30,
                max_retries=1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM归因失败: {e}")
            return "分析失败，请稍后重试。"

    def _llm_respond_to_query(
        self,
        query: str,
        context: Dict
    ) -> str:
        """LLM响应通用查询"""
        prompt = f"""用户问题：{query}

上下文：{json.dumps(context, ensure_ascii=False, indent=2)}

请用中文回答用户的问题。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的电力负荷预测分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500,
                request_timeout=30,
                max_retries=1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM响应失败: {e}")
            return "抱歉，分析过程中遇到问题，请稍后重试。"

    def _parse_periods(self, query: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """解析查询中的时间段"""
        # 简化实现，实际应用中需要更复杂的解析
        # 这里返回默认的两个月对比
        import re

        # 查找月份提及
        months = re.findall(r'(\d+)月', query)

        if len(months) >= 2:
            m1, m2 = int(months[0]), int(months[1])
            return (f"2023-{m1:02d}-01", f"2023-{m1:02d}-28"), (f"2023-{m2:02d}-01", f"2023-{m2:02d}-28")

        # 默认6月和7月对比
        return ("2023-06-01", "2023-06-30"), ("2023-07-01", "2023-07-31")

    def _parse_date_range(self, query: str) -> Dict:
        """解析日期范围"""
        import re

        # 查找日期格式
        dates = re.findall(r'\d{4}[-/]\d{2}[-/]\d{2}', query)

        if len(dates) >= 2:
            return {"start": dates[0], "end": dates[1]}
        elif len(dates) == 1:
            return {"start": dates[0], "end": dates[0]}

        return {}

    def _parse_whatif_scenario(self, query: str) -> Dict:
        """解析What-If场景"""
        import re

        scenario = {}

        # 温度
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*°?C', query)
        if temp_match:
            scenario["temperature"] = float(temp_match.group(1))

        # 湿度
        humidity_match = re.search(r'湿度\s*(\d+(?:\.\d+)?)', query)
        if humidity_match:
            scenario["humidity"] = float(humidity_match.group(1))

        # 工作日
        workday_match = re.search(r'(工作日|周末|节假日)', query)
        if workday_match:
            workday_text = workday_match.group(1)
            if workday_text == "工作日":
                scenario["is_workday"] = 1
            elif workday_text in ["周末"]:
                scenario["is_workday"] = 0

        return scenario

    def _get_current_features(self, df: pd.DataFrame) -> Dict:
        """获取当前最新的特征"""
        if df.empty:
            return {}

        row = df.iloc[-1]
        features = {}

        # 复制所有数值特征
        for col in df.columns:
            if col in ["load", "actual_load", "predicted_load", "datetime"]:
                continue
            val = row[col]
            if not pd.isna(val):
                features[col] = float(val) if isinstance(val, (int, float, np.number)) else str(val)

        return features

    def _simulate_prediction(self, features: Dict) -> float:
        """模拟预测（基于特征计算）"""
        # 简化实现
        base_load = 2500

        # 温度影响
        temp = features.get("temperature", 25)
        if temp > 25:
            temp_effect = (temp - 25) * 15  # 高温增加负荷
        elif temp < 15:
            temp_effect = (15 - temp) * 10  # 低温增加负荷
        else:
            temp_effect = 0

        # 时段影响
        hour = features.get("hour", 12)
        if 8 <= hour <= 22:
            hour_effect = 200
        else:
            hour_effect = -200

        # 工作日影响
        is_workday = features.get("is_workday", 1)
        workday_effect = 150 if is_workday == 1 else -150

        prediction = base_load + temp_effect + hour_effect + workday_effect
        return max(500, prediction)  # 确保最小值

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
            "day_of_week": "星期",
        }
        return name_map.get(feature, feature)


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

    # 演示对话分析
    np.random.seed(42)
    n = 500

    dates = pd.date_range("2023-06-01", periods=n, freq="H")
    data = {
        "datetime": dates,
        "load": 2500 + np.sin(np.arange(n) * np.pi / 12) * 300 + np.random.randn(n) * 100,
        "temperature": 25 + np.sin(np.arange(n) * np.pi / 12) * 10 + np.random.randn(n) * 3,
        "humidity": 50 + np.random.randn(n) * 15,
        "hour": [h % 24 for h in range(n)],
        "is_workday": [1 if d.weekday() < 5 else 0 for d in dates],
    }

    df = pd.DataFrame(data)
    predictions_df = df.copy()
    predictions_df["predicted_load"] = predictions_df["load"] + np.random.randn(n) * 80

    # 模拟异常
    predictions_df.loc[predictions_df.index[20], "predicted_load"] = predictions_df.loc[predictions_df.index[20], "load"] * 1.4

    # 初始化分析师
    analyst = DialogueAnalyst(max_history=5)

    # 数据缓存
    data_cache = {"data": df, "predictions": predictions_df}

    # 测试各种意图
    test_queries = [
        "有哪些预测异常？",
        "对比6月和7月的负荷",
        "如果明天35度会怎样？",
    ]

    print("\n" + "=" * 60)
    print("对话式分析演示")
    print("=" * 60)

    for query in test_queries:
        print(f"\n用户: {query}")
        result = analyst.analyze(query, data_cache=data_cache)
        print(f"意图: {result.intent}")
        print(f"分析: {result.explanation[:300]}...")
