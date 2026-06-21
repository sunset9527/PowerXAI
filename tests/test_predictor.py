"""
预测器测试
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.predictor import predict, predict_batch
from src.model.trainer import ensure_model_exists
from src.data.generator import ensure_data_exists
from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns


@pytest.fixture(scope="module")
def setup_model():
    """设置模型（所有测试共享）"""
    # 确保数据存在
    raw_df = ensure_data_exists()

    # 预处理
    processed_df, _ = preprocess_data(raw_df)

    # 划分数据集
    train_df, test_df = split_train_test(processed_df)

    # 确保模型存在
    feature_cols = get_feature_columns(processed_df)
    model_data = ensure_model_exists(train_df, feature_cols)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_columns": feature_cols,
    }


class TestPredictor:
    """预测器测试类"""

    def test_predict_single(self, setup_model):
        """测试单次预测"""
        # 准备特征
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

        # 预测
        result = predict(features)

        # 验证
        assert "prediction" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert result["prediction"] > 0
        assert result["lower_bound"] <= result["prediction"] <= result["upper_bound"]
        assert result["unit"] == "MW"

    def test_predict_batch(self, setup_model):
        """测试批量预测"""
        test_df = setup_model["test_df"]

        # 取前10条
        batch_df = test_df.head(10)

        # 批量预测
        result_df = predict_batch(batch_df)

        # 验证
        assert len(result_df) == 10
        assert "predicted_load" in result_df.columns
        assert all(result_df["predicted_load"] > 0)

    def test_prediction_range(self, setup_model):
        """测试预测值范围合理性"""
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

        result = predict(features)

        # 负荷应该在合理范围内（500-5000 MW）
        assert 500 < result["prediction"] < 5000
        assert result["lower_bound"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
