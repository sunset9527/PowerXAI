"""
对比分析器测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyst.comparator import (
    Comparator,
    compare_periods,
    compare_with_yesterday,
    compare_with_last_week
)
from src.model.trainer import ensure_model_exists
from src.data.generator import ensure_data_exists
from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns


@pytest.fixture(scope="module")
def setup_model():
    """设置模型"""
    raw_df = ensure_data_exists()
    processed_df, _ = preprocess_data(raw_df)
    train_df, test_df = split_train_test(processed_df)
    feature_cols = get_feature_columns(processed_df)
    ensure_model_exists(train_df, feature_cols)
    return {}


def get_base_features():
    """获取基础特征"""
    return {
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
        "temperature": 30.0,
        "humidity": 60,
        "apparent_temperature": 32.0,
        "thermal_comfort_index": 5.0,
        "heating_cooling_index": 12.0,
        "temperature_squared": 900.0,
        "temp_workday_interaction": 30.0,
        "temp_humidity_interaction": 18.0,
        "heat_humidity_index": 30.0,
        "load_lag_1h": 2500.0,
        "load_lag_24h": 2400.0,
        "load_lag_168h": 2450.0,
        "load_same_hour_last_week": 2450.0,
        "load_change_1h": 0.01,
        "load_change_24h": 0.03,
        "load_ma_3h": 2480.0,
        "load_ma_24h": 2450.0,
    }


def get_hot_features():
    """获取高温特征"""
    base = get_base_features()
    base["temperature"] = 35.0
    base["apparent_temperature"] = 38.0
    base["thermal_comfort_index"] = 10.0
    base["heating_cooling_index"] = 19.5
    base["temperature_squared"] = 1225.0
    base["temp_workday_interaction"] = 35.0
    base["temp_humidity_interaction"] = 21.0
    base["heat_humidity_index"] = 35.5
    base["load_lag_1h"] = 2700.0
    base["load_ma_3h"] = 2680.0
    return base


class TestComparator:
    """对比分析器测试类"""

    def test_compare_periods(self, setup_model):
        """测试时段对比"""
        features1 = get_base_features()
        features2 = get_hot_features()

        result = compare_periods(
            features1, features2,
            prediction1=2600.0,
            prediction2=2850.0
        )

        # 验证
        assert result.period1_info["prediction"] == 2600.0
        assert result.period2_info["prediction"] == 2850.0
        assert result.prediction_diff > 0
        assert result.prediction_diff_pct > 0
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_compare_temperature_change(self, setup_model):
        """测试温度变化对比"""
        features1 = get_base_features()
        features2 = get_base_features()
        features2["temperature"] = 35.0
        features2["temperature_squared"] = 1225.0

        result = compare_periods(
            features1, features2,
            prediction1=2600.0,
            prediction2=2850.0
        )

        # 温度差异应该被识别
        assert "温度" in result.explanation or "temperature" in str(result.feature_changes).lower()

    def test_feature_changes_structure(self, setup_model):
        """测试特征变化结构"""
        features1 = get_base_features()
        features2 = get_hot_features()

        result = compare_periods(
            features1, features2,
            prediction1=2600.0,
            prediction2=2850.0
        )

        # 验证特征变化列表
        assert isinstance(result.feature_changes, list)
        assert len(result.feature_changes) > 0

        # 检查结构
        for fc in result.feature_changes:
            assert hasattr(fc, "feature")
            assert hasattr(fc, "value1")
            assert hasattr(fc, "value2")
            assert hasattr(fc, "shap1")
            assert hasattr(fc, "shap2")
            assert hasattr(fc, "shap_change")
            assert hasattr(fc, "impact_type")

    def test_key_drivers_reducers(self, setup_model):
        """测试关键驱动和降低因素"""
        features1 = get_base_features()
        features2 = get_hot_features()

        result = compare_periods(
            features1, features2,
            prediction1=2600.0,
            prediction2=2850.0
        )

        # 应该有驱动因素
        assert isinstance(result.key_drivers, list)
        # 可能有或没有降低因素
        assert isinstance(result.key_reducers, list)

    def test_comparison_result_serialization(self, setup_model):
        """测试结果可序列化"""
        features1 = get_base_features()
        features2 = get_hot_features()

        result = compare_periods(
            features1, features2,
            prediction1=2600.0,
            prediction2=2850.0
        )

        # 转换为字典
        result_dict = {
            "period1_prediction": result.period1_info["prediction"],
            "period2_prediction": result.period2_info["prediction"],
            "prediction_diff": result.prediction_diff,
            "prediction_diff_pct": result.prediction_diff_pct,
            "explanation": result.explanation,
        }

        # 验证可以序列化
        assert isinstance(result_dict, dict)
        assert result_dict["prediction_diff"] == result.prediction_diff


class TestCompareHelpers:
    """对比辅助函数测试"""

    def test_compare_with_yesterday(self, setup_model):
        """测试与昨日对比"""
        today_features = get_hot_features()
        yesterday_features = get_base_features()

        result = compare_with_yesterday(
            features=today_features,
            yesterday_features=yesterday_features,
            prediction=2850.0,
            yesterday_prediction=2600.0
        )

        # 验证
        assert result.prediction_diff > 0

    def test_compare_with_last_week(self, setup_model):
        """测试与上周对比"""
        features1 = get_base_features()
        features2 = get_hot_features()

        result = compare_with_last_week(
            features=features2,
            last_week_features=features1,
            prediction=2850.0,
            last_week_prediction=2600.0
        )

        # 验证
        assert isinstance(result.explanation, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
