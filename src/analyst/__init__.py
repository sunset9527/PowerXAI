"""
分析服务模块
"""

from .llm_explainer import (
    LLMExplainer,
    generate_explanation,
    generate_explanation_sync
)
from .insight import (
    InsightEngine,
    detect_trends,
    detect_anomalies,
    find_correlations
)
from .comparator import (
    Comparator,
    compare_periods,
    compare_with_yesterday,
    compare_with_last_week
)
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyRecord,
)
from .report_generator import (
    ReportGenerator,
)
from .dialogue_analyst import (
    DialogueAnalyst,
    AnalysisResult,
    ConversationEntry,
)

__all__ = [
    # LLM解释器
    "LLMExplainer",
    "generate_explanation",
    "generate_explanation_sync",
    # 洞察引擎
    "InsightEngine",
    "detect_trends",
    "detect_anomalies",
    "find_correlations",
    # 对比器
    "Comparator",
    "compare_periods",
    "compare_with_yesterday",
    "compare_with_last_week",
    # 异常检测器
    "AnomalyDetector",
    "AnomalyRecord",
    # 报告生成器
    "ReportGenerator",
    # 对话分析师
    "DialogueAnalyst",
    "AnalysisResult",
    "ConversationEntry",
]
