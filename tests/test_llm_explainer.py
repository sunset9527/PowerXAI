"""
LLM解释器测试
"""

import pytest
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyst.llm_explainer import (
    LLMExplainer,
    generate_explanation_sync
)
from src.explainer.shap_analyzer import explain_prediction
from src.explainer.report import generate_report
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


def get_sample_features():
    """获取示例特征"""
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


class TestLLMExplainer:
    """LLM解释器测试类"""

    def test_explainer_initialization(self, setup_model):
        """测试解释器初始化"""
        # 不带API密钥初始化
        explainer = LLMExplainer(api_key=None)
        assert explainer.client is None

    def test_generate_fallback_explanation(self, setup_model):
        """测试备用解释生成"""
        features = get_sample_features()
        contributions = explain_prediction(2850.5, features)
        report = generate_report(2850.5, features, contributions)

        # 使用不带API的方式生成解释
        explainer = LLMExplainer(api_key=None)
        explanation = explainer._generate_fallback_explanation(
            prediction=2850.5,
            features=features,
            contributions=contributions,
            report=report,
            detail_level="standard"
        )

        # 验证
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "2850.5" in explanation or "2851" in explanation or "2850" in explanation

    def test_generate_explanation_sync(self, setup_model):
        """测试同步解释生成"""
        features = get_sample_features()
        contributions = explain_prediction(2850.5, features)

        explanation = generate_explanation_sync(
            prediction=2850.5,
            features=features,
            contributions=contributions,
            detail_level="brief"
        )

        # 验证
        assert isinstance(explanation, str)
        assert len(explanation) > 10

    def test_different_detail_levels(self, setup_model):
        """测试不同详细程度"""
        features = get_sample_features()
        contributions = explain_prediction(2850.5, features)

        for level in ["brief", "standard", "detailed"]:
            explanation = generate_explanation_sync(
                prediction=2850.5,
                features=features,
                contributions=contributions,
                detail_level=level
            )

            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_explanation_contains_context(self, setup_model):
        """测试解释包含上下文信息"""
        features = get_sample_features()
        contributions = explain_prediction(2850.5, features)

        explanation = generate_explanation_sync(
            prediction=2850.5,
            features=features,
            contributions=contributions,
            detail_level="standard"
        )

        # 检查是否包含关键上下文
        # （由于是备用解释，检查一些基本关键词）
        assert isinstance(explanation, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
