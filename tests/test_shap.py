"""
SHAP分析测试
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainer.shap_analyzer import (
    SHAPAnalyzer,
    explain_prediction,
    explain_global,
    get_feature_contribution
)
from src.model.trainer import ensure_model_exists
from src.data.generator import ensure_data_exists
from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns


@pytest.fixture(scope="module")
def setup_analyzer():
    """设置SHAP分析器"""
    # 确保数据存在
    raw_df = ensure_data_exists()

    # 预处理
    processed_df, _ = preprocess_data(raw_df)

    # 划分数据集
    train_df, test_df = split_train_test(processed_df)

    # 确保模型存在
    feature_cols = get_feature_columns(processed_df)
    ensure_model_exists(train_df, feature_cols)

    # 创建分析器
    analyzer = SHAPAnalyzer()

    return {
        "analyzer": analyzer,
        "processed_df": processed_df,
        "test_df": test_df,
    }


class TestSHAPAnalyzer:
    """SHAP分析器测试类"""

    def test_explain_single(self, setup_analyzer):
        """测试单次预测的SHAP分析"""
        features = {
            "hour": 14,
            "day_of_week": 2,
            "day_of_week_sin": 0.974,
            "day_of_week_cos": -0.225,
            "is_weekend": 0,
            "is_holiday": 0,
            "is_workday": 1,
            "month": 6,
            "month_sin": 0.5,
            "month_cos": 0.866,
            "day_of_month": 18,
            "day_of_month_sin": 0.485,
            "day_of_month_cos": 0.875,
            "hour_sin": 0.975,
            "hour_cos": -0.222,
            "temperature": 35.0,
            "humidity": 60,
            "apparent_temperature": 38.5,
            "thermal_comfort_index": 10.0,
            "heating_cooling_index": 19.5,
            "temperature_squared": 1225.0,
            "temp_workday_interaction": 35.0,
            "temp_humidity_interaction": 21.0,
            "heat_humidity_index": 35.5,
            "load_lag_1h": 2700.0,
            "load_lag_24h": 2600.0,
            "load_lag_168h": 2500.0,
            "load_same_hour_last_week": 2500.0,
            "load_change_1h": 0.02,
            "load_change_24h": 0.05,
            "load_ma_3h": 2680.0,
            "load_ma_24h": 2650.0,
        }

        # SHAP分析
        contributions = explain_prediction(None, features)

        # 验证
        assert isinstance(contributions, list)
        assert len(contributions) > 0

        # 检查结构
        for c in contributions:
            assert "feature" in c
            assert "shap_value" in c
            assert "feature_value" in c
            assert "direction" in c
            assert "abs_contribution" in c
            assert c["direction"] in ["positive", "negative"]

        # 按绝对值排序
        abs_values = [c["abs_contribution"] for c in contributions]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_explain_global(self, setup_analyzer):
        """测试全局特征重要性分析"""
        analyzer = setup_analyzer["analyzer"]
        test_df = setup_analyzer["test_df"]

        # 全局分析
        importance = explain_global(test_df.head(100))

        # 验证
        assert isinstance(importance, dict)
        assert len(importance) > 0

        # 检查结构
        for feat, values in importance.items():
            assert "mean_abs_shap" in values
            assert "std_shap" in values
            assert "positive_ratio" in values
            assert "feature_value_mean" in values

        # 检查排序
        values = [v["mean_abs_shap"] for v in importance.values()]
        assert values == sorted(values, reverse=True)

    def test_get_feature_contribution(self, setup_analyzer):
        """测试特征贡献摘要"""
        features = {
            "hour": 14,
            "day_of_week": 2,
            "day_of_week_sin": 0.974,
            "day_of_week_cos": -0.225,
            "is_weekend": 0,
            "is_holiday": 0,
            "is_workday": 1,
            "month": 6,
            "month_sin": 0.5,
            "month_cos": 0.866,
            "day_of_month": 18,
            "day_of_month_sin": 0.485,
            "day_of_month_cos": 0.875,
            "hour_sin": 0.975,
            "hour_cos": -0.222,
            "temperature": 35.0,
            "humidity": 60,
            "apparent_temperature": 38.5,
            "thermal_comfort_index": 10.0,
            "heating_cooling_index": 19.5,
            "temperature_squared": 1225.0,
            "temp_workday_interaction": 35.0,
            "temp_humidity_interaction": 21.0,
            "heat_humidity_index": 35.5,
            "load_lag_1h": 2700.0,
            "load_lag_24h": 2600.0,
            "load_lag_168h": 2500.0,
            "load_same_hour_last_week": 2500.0,
            "load_change_1h": 0.02,
            "load_change_24h": 0.05,
            "load_ma_3h": 2680.0,
            "load_ma_24h": 2650.0,
        }

        contributions = explain_prediction(None, features)
        summary = get_feature_contribution(contributions, top_n=5)

        # 验证
        assert "top_contributors" in summary
        assert len(summary["top_contributors"]) == 5
        assert "n_positive" in summary
        assert "n_negative" in summary
        assert "total_positive_contribution" in summary
        assert "total_negative_contribution" in summary
        assert "net_contribution" in summary

    def test_shap_value_consistency(self, setup_analyzer):
        """测试SHAP值一致性"""
        features = {
            "hour": 14,
            "day_of_week": 2,
            "day_of_week_sin": 0.974,
            "day_of_week_cos": -0.225,
            "is_weekend": 0,
            "is_holiday": 0,
            "is_workday": 1,
            "month": 6,
            "month_sin": 0.5,
            "month_cos": 0.866,
            "day_of_month": 18,
            "day_of_month_sin": 0.485,
            "day_of_month_cos": 0.875,
            "hour_sin": 0.975,
            "hour_cos": -0.222,
            "temperature": 35.0,
            "humidity": 60,
            "apparent_temperature": 38.5,
            "thermal_comfort_index": 10.0,
            "heating_cooling_index": 19.5,
            "temperature_squared": 1225.0,
            "temp_workday_interaction": 35.0,
            "temp_humidity_interaction": 21.0,
            "heat_humidity_index": 35.5,
            "load_lag_1h": 2700.0,
            "load_lag_24h": 2600.0,
            "load_lag_168h": 2500.0,
            "load_same_hour_last_week": 2500.0,
            "load_change_1h": 0.02,
            "load_change_24h": 0.05,
            "load_ma_3h": 2680.0,
            "load_ma_24h": 2650.0,
        }

        # 两次分析应该得到相同的结果
        contributions1 = explain_prediction(None, features)
        contributions2 = explain_prediction(None, features)

        # 比较SHAP值
        for c1, c2 in zip(contributions1, contributions2):
            assert c1["feature"] == c2["feature"]
            assert abs(c1["shap_value"] - c2["shap_value"]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
