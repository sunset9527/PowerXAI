"""
模型模块
"""

# 训练器
from .trainer import (
    train_model,
    train_xgboost,
    train_lightgbm,
    train_all_models,
    save_model,
    load_model,
    get_model_info,
    ensure_model_exists,
)

# 预测器
from .predictor import predict, predict_batch, load_model as load_model_for_predict

# 评估器
from .evaluator import (
    evaluate_model,
    calculate_metrics,
    compare_models_on_test,
    format_comparison_table,
)

# LSTM
from .lstm_model import LSTMPredictor, create_lstm_predictor

# 集成
from .ensemble import EnsemblePredictor, create_ensemble_from_models

# 交叉验证
from .cross_validator import (
    TimeSeriesCrossValidator,
    cross_validate_models,
    compare_cv_results,
)

# 模型选择
from .selector import ModelSelector, select_best_model

__all__ = [
    # 训练器
    "train_model",
    "train_xgboost",
    "train_lightgbm",
    "train_all_models",
    "save_model",
    "load_model",
    "get_model_info",
    "ensure_model_exists",
    # 预测器
    "predict",
    "predict_batch",
    "load_model_for_predict",
    # 评估器
    "evaluate_model",
    "calculate_metrics",
    "compare_models_on_test",
    "format_comparison_table",
    # LSTM
    "LSTMPredictor",
    "create_lstm_predictor",
    # 集成
    "EnsemblePredictor",
    "create_ensemble_from_models",
    # 交叉验证
    "TimeSeriesCrossValidator",
    "cross_validate_models",
    "compare_cv_results",
    # 模型选择
    "ModelSelector",
    "select_best_model",
]
