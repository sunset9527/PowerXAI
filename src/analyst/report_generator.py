"""
自动报告生成模块

功能：
- 生成结构化分析报告
- 支持Markdown/PDF格式导出
- LLM生成摘要和建议
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from config import settings
from ..explainer.shap_analyzer import explain_prediction, get_feature_contribution
from ..explainer.report import generate_report as shap_generate_report
from .anomaly_detector import AnomalyDetector
from .comparator import Comparator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    报告生成器

    负责生成结构化的电力负荷预测分析报告
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """
        初始化报告生成器

        Args:
            api_key: DeepSeek API密钥（可选）
            output_dir: 报告输出目录（可选）
        """
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.output_dir = output_dir or settings.REPORT_OUTPUT_DIR

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化LLM客户端
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.DEEPSEEK_BASE_URL
            )
        else:
            self.client = None
            logger.warning("未设置DEEPSEEK_API_KEY，摘要和建议将使用模板生成")

        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.LLM_TEMPERATURE

        # 初始化组件
        self.anomaly_detector = AnomalyDetector(api_key=self.api_key)
        self.comparator = Comparator(api_key=self.api_key)

    def generate_report(
        self,
        df: pd.DataFrame,
        predictions_df: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        format: str = "markdown"
    ) -> Dict:
        """
        生成结构化报告

        Args:
            df: 原始数据DataFrame
            predictions_df: 预测结果DataFrame（包含actual_load和predicted_load列）
            start_date: 报告起始日期
            end_date: 报告结束日期
            format: 报告格式 (markdown/dict)

        Returns:
            报告字典
        """
        # 确定时间范围
        if "datetime" in df.columns:
            df = df.sort_values("datetime")
            if not start_date:
                start_date = str(df["datetime"].min().date())
            if not end_date:
                end_date = str(df["datetime"].max().date())

        # 计算基础统计
        stats = self._calculate_statistics(df)

        # 计算预测准确性（如果有预测数据）
        accuracy = {}
        if predictions_df is not None and "actual_load" in predictions_df.columns:
            accuracy = self._calculate_accuracy(predictions_df)

        # 检测异常
        anomalies = []
        if predictions_df is not None:
            anomaly_records = self.anomaly_detector.scan_deviations(predictions_df)
            for a in anomaly_records[:10]:  # 最多分析10个异常
                self.anomaly_detector.analyze_root_cause(a)
                anomalies.append({
                    "timestamp": a.timestamp,
                    "actual": a.actual,
                    "predicted": a.predicted,
                    "error_pct": a.error_pct,
                    "severity": a.severity,
                    "root_cause": a.root_cause
                })

        # 分析Top驱动因素
        top_drivers = self._analyze_top_drivers(df)

        # 生成摘要和建议（使用LLM）
        summary = self._generate_summary(stats, accuracy, anomalies, start_date, end_date)
        recommendations = self._generate_recommendations(stats, anomalies, top_drivers)

        # 构建报告
        report = {
            "title": f"电力负荷预测分析报告",
            "period": f"{start_date} 至 {end_date}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "statistics": stats,
            "accuracy": accuracy,
            "anomalies": anomalies,
            "top_drivers": top_drivers,
            "recommendations": recommendations
        }

        if format == "markdown":
            report["markdown"] = self._dict_to_markdown(report)

        return report

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """计算基础统计"""
        load_col = "load" if "load" in df.columns else "actual_load"
        if load_col not in df.columns:
            return {}

        stats = {
            "total_records": len(df),
            "mean": float(df[load_col].mean()),
            "std": float(df[load_col].std()),
            "min": float(df[load_col].min()),
            "max": float(df[load_col].max()),
            "median": float(df[load_col].median()),
            "q25": float(df[load_col].quantile(0.25)),
            "q75": float(df[load_col].quantile(0.75)),
        }

        # 计算日统计
        if "datetime" in df.columns:
            df_temp = df.copy()
            df_temp["date"] = df_temp["datetime"].dt.date
            daily_stats = df_temp.groupby("date")[load_col].agg(["mean", "max", "min"])
            stats["daily_mean_max"] = float(daily_stats["mean"].max())
            stats["daily_mean_min"] = float(daily_stats["mean"].min())

        # 工作日/周末统计
        if "is_workday" in df.columns:
            workday_load = df[df["is_workday"] == 1][load_col].mean()
            weekend_load = df[df["is_workday"] == 0][load_col].mean()
            stats["workday_mean"] = float(workday_load) if not pd.isna(workday_load) else None
            stats["weekend_mean"] = float(weekend_load) if not pd.isna(weekend_load) else None

        # 温度相关统计
        if "temperature" in df.columns:
            stats["temperature_mean"] = float(df["temperature"].mean())
            stats["temperature_max"] = float(df["temperature"].max())
            stats["temperature_min"] = float(df["temperature"].min())

        return stats

    def _calculate_accuracy(self, predictions_df: pd.DataFrame) -> Dict:
        """计算预测准确性"""
        if "actual_load" not in predictions_df.columns or "predicted_load" not in predictions_df.columns:
            return {}

        actual = predictions_df["actual_load"]
        predicted = predictions_df["predicted_load"]

        # 计算误差
        errors = actual - predicted
        abs_errors = abs(errors)
        pct_errors = abs_errors / actual.replace(0, 1) * 100

        accuracy = {
            "mae": float(abs_errors.mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "mape": float(pct_errors.mean()),
            "max_error_pct": float(pct_errors.max()),
        }

        # 计算R²
        ss_res = ((actual - predicted) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        accuracy["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0

        return accuracy

    def _analyze_top_drivers(self, df: pd.DataFrame) -> List[Dict]:
        """分析Top驱动因素"""
        drivers = []

        # 基于相关性分析
        load_col = "load" if "load" in df.columns else "actual_load"

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in [load_col, "predicted_load", "actual_load"]:
                continue
            corr = df[col].corr(df[load_col])
            if not pd.isna(corr) and abs(corr) > 0.3:
                drivers.append({
                    "feature": col,
                    "feature_cn": self._get_feature_cn(col),
                    "correlation": float(corr)
                })

        # 按相关性排序
        drivers.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return drivers[:10]

    def _generate_summary(
        self,
        stats: Dict,
        accuracy: Dict,
        anomalies: List[Dict],
        start_date: str,
        end_date: str
    ) -> str:
        """生成摘要（使用LLM）"""
        if self.client:
            return self._llm_generate_summary(stats, accuracy, anomalies, start_date, end_date)
        else:
            return self._template_summary(stats, accuracy, anomalies, start_date, end_date)

    def _llm_generate_summary(
        self,
        stats: Dict,
        accuracy: Dict,
        anomalies: List[Dict],
        start_date: str,
        end_date: str
    ) -> str:
        """使用LLM生成摘要"""
        # 构建上下文
        context_parts = []

        # 统计摘要
        context_parts.append(f"## 基础统计 ({start_date} 至 {end_date})")
        context_parts.append(f"- 平均负荷: {stats.get('mean', 0):.2f} MW")
        context_parts.append(f"- 最大负荷: {stats.get('max', 0):.2f} MW")
        context_parts.append(f"- 最小负荷: {stats.get('min', 0):.2f} MW")

        if "temperature_mean" in stats:
            context_parts.append(f"- 平均温度: {stats['temperature_mean']:.1f}°C")

        if "workday_mean" in stats and stats["workday_mean"]:
            context_parts.append(f"- 工作日平均: {stats['workday_mean']:.2f} MW")
        if "weekend_mean" in stats and stats["weekend_mean"]:
            context_parts.append(f"- 周末平均: {stats['weekend_mean']:.2f} MW")

        # 准确性摘要
        if accuracy:
            context_parts.append(f"\n## 预测准确性")
            context_parts.append(f"- MAE: {accuracy.get('mae', 0):.2f} MW")
            context_parts.append(f"- RMSE: {accuracy.get('rmse', 0):.2f} MW")
            context_parts.append(f"- MAPE: {accuracy.get('mape', 0):.1f}%")
            context_parts.append(f"- R²: {accuracy.get('r2', 0):.3f}")

        # 异常摘要
        if anomalies:
            high_anomalies = [a for a in anomalies if a["severity"] in ["high", "critical"]]
            context_parts.append(f"\n## 异常情况")
            context_parts.append(f"- 共检测到 {len(anomalies)} 个异常")
            if high_anomalies:
                context_parts.append(f"- 其中 {len(high_anomalies)} 个为高/严重异常")

        prompt = f"""你是一位专业的电力负荷预测分析师，请根据以下数据生成一段简洁的分析摘要（3-5句话）。

{chr(10).join(context_parts)}

要求：
1. 用通俗易懂的语言总结主要发现
2. 突出关键变化和趋势
3. 不要使用markdown格式
4. 直接输出摘要内容"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的电力负荷预测分析师。"
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

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM摘要生成失败: {e}")
            return self._template_summary(stats, accuracy, anomalies, start_date, end_date)

    def _template_summary(
        self,
        stats: Dict,
        accuracy: Dict,
        anomalies: List[Dict],
        start_date: str,
        end_date: str
    ) -> str:
        """使用模板生成摘要"""
        summary_parts = []

        # 基础信息
        summary_parts.append(
            f"本报告覆盖 {start_date} 至 {end_date} 时段，共分析 {stats.get('total_records', 0)} 条记录。"
        )

        # 负荷特征
        mean_load = stats.get("mean", 0)
        max_load = stats.get("max", 0)
        summary_parts.append(
            f"期间平均负荷为 {mean_load:.2f} MW，峰值达到 {max_load:.2f} MW。"
        )

        # 准确性
        if accuracy:
            mape = accuracy.get("mape", 0)
            r2 = accuracy.get("r2", 0)
            summary_parts.append(
                f"预测模型平均绝对百分比误差为 {mape:.1f}%，R²为 {r2:.3f}。"
            )

        # 异常情况
        if anomalies:
            high_count = len([a for a in anomalies if a["severity"] in ["high", "critical"]])
            summary_parts.append(
                f"检测到 {len(anomalies)} 个异常，其中 {high_count} 个需要重点关注。"
            )

        return " ".join(summary_parts)

    def _generate_recommendations(
        self,
        stats: Dict,
        anomalies: List[Dict],
        top_drivers: List[Dict]
    ) -> List[str]:
        """生成运营建议（使用LLM）"""
        if self.client:
            return self._llm_generate_recommendations(stats, anomalies, top_drivers)
        else:
            return self._template_recommendations(stats, anomalies, top_drivers)

    def _llm_generate_recommendations(
        self,
        stats: Dict,
        anomalies: List[Dict],
        top_drivers: List[Dict]
    ) -> List[str]:
        """使用LLM生成建议"""
        # 构建上下文
        context_parts = []

        if top_drivers:
            context_parts.append("## 主要影响因素")
            for d in top_drivers[:5]:
                context_parts.append(f"- {d['feature_cn']}: 相关系数 {d['correlation']:.3f}")

        if anomalies:
            context_parts.append("\n## 异常情况")
            for a in anomalies[:3]:
                context_parts.append(f"- {a['timestamp']}: 偏差 {a['error_pct']:.1f}%，根因: {a['root_cause']}")

        prompt = f"""基于以下分析数据，生成3-5条运营建议：

{chr(10).join(context_parts)}

要求：
1. 建议应具体、可操作
2. 涵盖短期调度和长期规划两个方面
3. 用中文回答，每条建议一行
4. 不要使用编号或markdown格式"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的电力负荷预测分析师，擅长提出运营建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=600,
                request_timeout=30,
                max_retries=1
            )

            result = response.choices[0].message.content.strip()
            # 分割成列表
            recommendations = [line.strip() for line in result.split("\n") if line.strip()]
            return recommendations[:5]

        except Exception as e:
            logger.error(f"LLM建议生成失败: {e}")
            return self._template_recommendations(stats, anomalies, top_drivers)

    def _template_recommendations(
        self,
        stats: Dict,
        anomalies: List[Dict],
        top_drivers: List[Dict]
    ) -> List[str]:
        """使用模板生成建议"""
        recommendations = []

        # 基于Top驱动因素的建议
        if top_drivers:
            top_driver = top_drivers[0]
            recommendations.append(
                f"关注{top_driver['feature_cn']}变化，该因素与负荷相关性最高({top_driver['correlation']:.3f})，建议建立实时监控机制。"
            )

        # 基于温度的建议
        if "temperature_max" in stats and stats["temperature_max"] > 35:
            recommendations.append(
                f"夏季高温期间({stats['temperature_max']:.1f}°C峰值)应提前做好制冷负荷预案，确保电网稳定运行。"
            )

        # 基于工作日/周末差异的建议
        if "workday_mean" in stats and "weekend_mean" in stats:
            if stats["workday_mean"] and stats["weekend_mean"]:
                diff = stats["workday_mean"] - stats["weekend_mean"]
                if abs(diff) > 200:
                    recommendations.append(
                        f"工作日与周末负荷差异显著(约{abs(diff):.0f}MW)，建议优化发电机组组合，降低周末弃风弃光风险。"
                    )

        # 基于异常的建议
        if anomalies:
            high_anomalies = [a for a in anomalies if a["severity"] in ["high", "critical"]]
            if high_anomalies:
                recommendations.append(
                    f"检测到{len(high_anomalies)}个高偏差时段，建议排查数据采集设备，排除传感器故障或计量误差。"
                )

        # 默认建议
        if len(recommendations) < 3:
            recommendations.append("持续优化预测模型，定期更新训练数据以适应负荷变化新趋势。")
            recommendations.append("建议开展季节性负荷特性分析，提前规划电源和电网建设。")

        return recommendations[:5]

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
            "month": "月份",
        }
        return name_map.get(feature, feature)

    def _dict_to_markdown(self, report: Dict) -> str:
        """将报告字典转换为Markdown格式"""
        lines = []

        # 标题
        lines.append(f"# {report['title']}\n")
        lines.append(f"**报告周期**: {report['period']}\n")
        lines.append(f"**生成时间**: {report['generated_at']}\n")
        lines.append("\n---\n")

        # 摘要
        lines.append("## 执行摘要\n")
        lines.append(f"{report['summary']}\n")
        lines.append("\n---\n")

        # 统计信息
        lines.append("## 负荷统计\n")
        stats = report.get("statistics", {})

        lines.append(f"| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 记录数 | {stats.get('total_records', 'N/A')} |")
        lines.append(f"| 平均负荷 | {stats.get('mean', 0):.2f} MW |")
        lines.append(f"| 最大负荷 | {stats.get('max', 0):.2f} MW |")
        lines.append(f"| 最小负荷 | {stats.get('min', 0):.2f} MW |")
        lines.append(f"| 标准差 | {stats.get('std', 0):.2f} MW |")

        if stats.get("workday_mean"):
            lines.append(f"| 工作日平均 | {stats['workday_mean']:.2f} MW |")
        if stats.get("weekend_mean"):
            lines.append(f"| 周末平均 | {stats['weekend_mean']:.2f} MW |")
        if stats.get("temperature_mean"):
            lines.append(f"| 平均温度 | {stats['temperature_mean']:.1f}°C |")

        lines.append("\n")

        # 预测准确性
        accuracy = report.get("accuracy", {})
        if accuracy:
            lines.append("## 预测准确性\n")
            lines.append(f"| 指标 | 数值 |")
            lines.append("|------|------|")
            lines.append(f"| MAE | {accuracy.get('mae', 0):.2f} MW |")
            lines.append(f"| RMSE | {accuracy.get('rmse', 0):.2f} MW |")
            lines.append(f"| MAPE | {accuracy.get('mape', 0):.1f}% |")
            lines.append(f"| R² | {accuracy.get('r2', 0):.3f} |")
            lines.append("\n")

        # 异常时段
        anomalies = report.get("anomalies", [])
        if anomalies:
            lines.append("## 异常时段\n")
            lines.append(f"共检测到 **{len(anomalies)}** 个异常：\n")
            lines.append("| 时间 | 实际(MW) | 预测(MW) | 偏差率 | 严重程度 | 根因 |")
            lines.append("|------|----------|----------|--------|----------|------|")

            for a in anomalies[:10]:
                lines.append(
                    f"| {a['timestamp'][:16]} | {a['actual']:.1f} | {a['predicted']:.1f} | "
                    f"{a['error_pct']:.1f}% | {a['severity']} | {a['root_cause'][:30]}... |"
                )
            lines.append("\n")

        # Top驱动因素
        top_drivers = report.get("top_drivers", [])
        if top_drivers:
            lines.append("## 主要影响因素\n")
            lines.append("| 因素 | 相关系数 |")
            lines.append("|------|----------|")
            for d in top_drivers[:10]:
                lines.append(f"| {d['feature_cn']} | {d['correlation']:.3f} |")
            lines.append("\n")

        # 运营建议
        recommendations = report.get("recommendations", [])
        if recommendations:
            lines.append("## 运营建议\n")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("\n")

        return "\n".join(lines)

    def export_markdown(
        self,
        report: Dict,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        导出Markdown报告

        Args:
            report: 报告字典
            filepath: 输出文件路径（可选）

        Returns:
            输出文件路径
        """
        if filepath is None:
            filename = f"report_{report['period'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            filepath = self.output_dir / filename

        markdown_content = report.get("markdown", self._dict_to_markdown(report))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"报告已导出: {filepath}")
        return filepath

    def export_pdf_report(
        self,
        report: Dict,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        导出PDF报告

        使用markdown→HTML→PDF的转换方式

        Args:
            report: 报告字典
            filepath: 输出文件路径（可选）

        Returns:
            输出文件路径
        """
        if filepath is None:
            filename = f"report_{report['period'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            filepath = self.output_dir / filename

        # 尝试导入pdfkit
        try:
            import pdfkit
        except ImportError:
            logger.warning("pdfkit未安装，回退到纯Markdown格式")
            md_filepath = self.export_markdown(report, filepath.with_suffix(".md"))
            logger.info(f"PDF转换不可用，已导出Markdown: {md_filepath}")
            return md_filepath

        # 检查wkhtmltopdf
        try:
            pdfkit.from_file([], str(filepath), options={"quiet": ""})
        except Exception:
            logger.warning("wkhtmltopdf不可用，回退到纯Markdown格式")
            md_filepath = self.export_markdown(report, filepath.with_suffix(".md"))
            return md_filepath

        # Markdown → HTML
        import markdown

        markdown_content = report.get("markdown", self._dict_to_markdown(report))
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        code {{ background-color: #f4f4f4; padding: 2px 5px; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}
    </style>
</head>
<body>
{markdown.markdown(markdown_content, extensions=['tables'])}
</body>
</html>
"""

        # 保存HTML临时文件
        html_filepath = filepath.with_suffix(".html")
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        # HTML → PDF
        try:
            pdfkit.from_file(str(html_filepath), str(filepath))
            logger.info(f"报告已导出: {filepath}")
        except Exception as e:
            logger.error(f"PDF转换失败: {e}")
            # 回退到HTML
            return html_filepath
        finally:
            # 清理临时HTML
            if html_filepath.exists():
                html_filepath.unlink()

        return filepath


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

    # 演示报告生成
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

    # 模拟预测数据
    predictions_df = df.copy()
    predictions_df["predicted_load"] = predictions_df["load"] + np.random.randn(n) * 80
    predictions_df.loc[predictions_df.index[10], "predicted_load"] = predictions_df.loc[predictions_df.index[10], "load"] * 1.3
    predictions_df.loc[predictions_df.index[30], "predicted_load"] = predictions_df.loc[predictions_df.index[30], "load"] * 0.7

    # 生成报告
    generator = ReportGenerator()
    report = generator.generate_report(
        df,
        predictions_df=predictions_df,
        start_date="2023-06-01",
        end_date="2023-06-21"
    )

    print("\n" + "=" * 60)
    print("报告摘要")
    print("=" * 60)
    print(report["summary"])

    print("\n" + "=" * 60)
    print("运营建议")
    print("=" * 60)
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"{i}. {rec}")

    print("\n" + "=" * 60)
    print("Markdown报告预览")
    print("=" * 60)
    print(report["markdown"][:2000] + "...")
