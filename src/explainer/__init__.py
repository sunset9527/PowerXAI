"""
可解释性分析模块
"""

from .shap_analyzer import (
    SHAPAnalyzer,
    explain_prediction,
    explain_global,
    get_feature_contribution,
    get_analyzer,
)
from .lime_analyzer import (
    LIMEAnalyzer,
    explain_lime_prediction,
)
from .pdp_analyzer import (
    PDPAnalyzer,
    compute_partial_dependence,
)
from .ice_analyzer import (
    ICEAnalyzer,
    compute_individual_conditional_expectation,
)
from .xai_comparator import (
    XAIComparator,
    compare_xai_methods,
)
from .report import (
    generate_report,
    get_top_contributors,
    format_shap_values,
)

__all__ = [
    # SHAP
    "SHAPAnalyzer",
    "explain_prediction",
    "explain_global",
    "get_feature_contribution",
    "get_analyzer",
    # LIME
    "LIMEAnalyzer",
    "explain_lime_prediction",
    # PDP
    "PDPAnalyzer",
    "compute_partial_dependence",
    # ICE
    "ICEAnalyzer",
    "compute_individual_conditional_expectation",
    # 比较器
    "XAIComparator",
    "compare_xai_methods",
    # 报告
    "generate_report",
    "get_top_contributors",
    "format_shap_values",
]
