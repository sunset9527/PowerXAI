"""
集成预测器

通过加权平均集成多个模型的预测结果。

功能：
- XGBoost + LightGBM + LSTM 集成
- 支持手动权重和自动权重计算
- 返回集成预测和各模型预测
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    集成预测器

    通过加权平均组合多个模型的预测结果。
    """

    def __init__(
        self,
        models: Optional[Dict] = None,
        weights: Optional[Dict] = None,
        model_names: Optional[List[str]] = None
    ):
        """
        初始化集成预测器

        Args:
            models: 模型字典 {'xgboost': model, 'lightgbm': model, 'lstm': predictor}
            weights: 权重字典 {'xgboost': 0.4, 'lightgbm': 0.3, 'lstm': 0.3}
            model_names: 模型名称列表（用于排序）
        """
        self.models = models or {}
        self.weights = weights or {}
        self.model_names = model_names or ['xgboost', 'lightgbm', 'lstm']

        # 验证权重
        if self.weights and sum(self.weights.values()) != 1.0:
            logger.warning(
                f"权重之和不为1: {sum(self.weights.values())}，将进行归一化"
            )
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

    def add_model(
        self,
        name: str,
        model: any,
        weight: Optional[float] = None
    ):
        """
        添加模型到集成

        Args:
            name: 模型名称
            model: 模型对象
            weight: 模型权重（可选）
        """
        self.models[name] = model

        if weight is not None:
            self.weights[name] = weight

        if name not in self.model_names:
            self.model_names.append(name)

        logger.info(f"已将模型 '{name}' 添加到集成，权重: {weight}")

    def set_weights(self, weights: Dict[str, float]):
        """
        设置模型权重

        Args:
            weights: 权重字典
        """
        # 归一化
        total = sum(weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = {k: 1.0 / len(weights) for k in weights.keys()}

        logger.info(f"集成权重已更新: {self.weights}")

    def compute_weights_from_validation(
        self,
        validation_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        metric: str = 'mae'
    ) -> Dict[str, float]:
        """
        根据验证集表现计算权重

        Args:
            validation_predictions: 验证集预测字典
            y_true: 真实值
            metric: 评估指标 ('mae', 'rmse', 'mape')

        Returns:
            权重字典
        """
        errors = {}

        for name, preds in validation_predictions.items():
            if metric == 'mae':
                errors[name] = np.mean(np.abs(y_true - preds))
            elif metric == 'rmse':
                errors[name] = np.sqrt(np.mean((y_true - preds) ** 2))
            elif metric == 'mape':
                mask = y_true != 0
                errors[name] = np.mean(np.abs((y_true[mask] - preds[mask]) / y_true[mask]))
            else:
                errors[name] = np.mean(np.abs(y_true - preds))

        # 将误差转换为权重（误差越小权重越大）
        # 使用softmax风格的转换
        error_array = np.array(list(errors.values()))
        # 使用负数使得误差越小，exp后的值越大
        exp_errors = np.exp(-error_array * 10)  # 乘以系数放大差异
        weights = exp_errors / exp_errors.sum()

        weights_dict = dict(zip(errors.keys(), weights))

        logger.info(f"基于{metric}计算的集成权重: {weights_dict}")
        logger.info(f"各模型{metric}: {errors}")

        self.weights = weights_dict

        return weights_dict

    def predict(self, X: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        集成预测

        Args:
            X: 特征数组

        Returns:
            包含集成预测和各模型预测的字典
        """
        if not self.models:
            raise ValueError("集成中没有模型，请先添加模型")

        individual_predictions = {}
        weighted_predictions = np.zeros(len(X))

        for name in self.model_names:
            if name not in self.models:
                continue

            model = self.models[name]

            # 获取预测
            try:
                if name == 'lstm':
                    # LSTM预测器
                    preds = model.predict(X)
                elif name == 'xgboost':
                    # XGBoost
                    preds = model.predict(X)
                elif name == 'lightgbm':
                    # LightGBM
                    preds = model.predict(X)
                else:
                    # 默认假设有predict方法
                    preds = model.predict(X)

                individual_predictions[name] = preds

                # 加权累加
                weight = self.weights.get(name, 0)
                weighted_predictions += weight * preds

            except Exception as e:
                logger.warning(f"模型 '{name}' 预测失败: {e}")

        result = {
            'ensemble': weighted_predictions,
            'individual': individual_predictions,
            'weights': self.weights.copy()
        }

        return result

    def predict_with_confidence(
        self,
        X: np.ndarray,
        n_bootstrap: int = 100
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        带置信区间的集成预测

        Args:
            X: 特征数组
            n_bootstrap: Bootstrap采样次数

        Returns:
            包含预测、置信区间和各模型预测的字典
        """
        individual_predictions = self.predict(X)['individual']

        # 收集所有模型的预测
        all_preds = []
        for name, preds in individual_predictions.items():
            all_preds.append(preds)

        if len(all_preds) == 0:
            raise ValueError("没有可用的模型预测")

        all_preds = np.array(all_preds)  # (n_models, n_samples)

        # 加权平均
        weights = np.array([self.weights.get(name, 0) for name in individual_predictions.keys()])
        ensemble_pred = np.sum(all_preds * weights[:, np.newaxis], axis=0)

        # Bootstrap计算置信区间
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            # 随机选择模型（带权重）
            model_indices = np.random.choice(
                len(all_preds),
                size=len(all_preds),
                p=weights
            )
            bootstrap_pred = np.mean(all_preds[model_indices], axis=0)
            bootstrap_preds.append(bootstrap_pred)

        bootstrap_preds = np.array(bootstrap_preds)

        # 计算置信区间
        lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper = np.percentile(bootstrap_preds, 97.5, axis=0)

        return {
            'ensemble': ensemble_pred,
            'lower_bound': lower,
            'upper_bound': upper,
            'individual': individual_predictions,
            'weights': self.weights.copy()
        }

    def get_feature_importance(self) -> Optional[Dict]:
        """
        获取特征重要性（如果有模型支持）

        Returns:
            特征重要性字典
        """
        importances = {}

        # XGBoost
        if 'xgboost' in self.models:
            try:
                importances['xgboost'] = dict(zip(
                    self.models['xgboost'].feature_names_in_,
                    self.models['xgboost'].feature_importances_
                ))
            except AttributeError:
                pass

        # LightGBM
        if 'lightgbm' in self.models:
            try:
                importances['lightgbm'] = dict(zip(
                    self.models['lightgbm'].feature_name_,
                    self.models['lightgbm'].feature_importances_
                ))
            except AttributeError:
                pass

        return importances if importances else None

    def save(self, filepath: Path) -> Path:
        """
        保存集成预测器配置

        Args:
            filepath: 保存路径

        Returns:
            保存的文件路径
        """
        import joblib

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'weights': self.weights,
            'model_names': self.model_names,
            'config': {
                'hidden_size': self.models.get('lstm').hidden_size
                    if 'lstm' in self.models else None,
                'num_layers': self.models.get('lstm').num_layers
                    if 'lstm' in self.models else None,
            }
        }

        joblib.save(save_data, filepath)
        logger.info(f"集成预测器配置已保存至: {filepath}")

        return filepath

    def load(self, filepath: Path) -> 'EnsemblePredictor':
        """
        加载集成预测器配置

        Args:
            filepath: 配置文件路径

        Returns:
            self
        """
        import joblib

        save_data = joblib.load(filepath)

        self.weights = save_data['weights']
        self.model_names = save_data['model_names']

        logger.info(f"集成预测器配置已从 {filepath} 加载")

        return self


def create_ensemble_from_models(
    xgb_model: any = None,
    lgbm_model: any = None,
    lstm_predictor: any = None,
    weights: Optional[Dict] = None,
    auto_weights: bool = False,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> EnsemblePredictor:
    """
    从已有模型创建集成预测器

    Args:
        xgb_model: XGBoost模型
        lgbm_model: LightGBM模型
        lstm_predictor: LSTM预测器
        weights: 手动权重
        auto_weights: 是否自动计算权重
        X_val: 验证集特征
        y_val: 验证集目标

    Returns:
        EnsemblePredictor实例
    """
    ensemble = EnsemblePredictor()

    # 添加模型
    if xgb_model is not None:
        ensemble.add_model('xgboost', xgb_model)
    if lgbm_model is not None:
        ensemble.add_model('lightgbm', lgbm_model)
    if lstm_predictor is not None:
        ensemble.add_model('lstm', lstm_predictor)

    # 设置权重
    if weights is not None:
        ensemble.set_weights(weights)
    elif auto_weights and X_val is not None and y_val is not None:
        # 自动计算权重
        predictions = {}

        if xgb_model is not None:
            predictions['xgboost'] = xgb_model.predict(X_val)
        if lgbm_model is not None:
            predictions['lightgbm'] = lgbm_model.predict(X_val)
        if lstm_predictor is not None:
            predictions['lstm'] = lstm_predictor.predict(X_val)

        if predictions:
            # 跳过前24个（LSTM序列长度）
            eval_y = y_val[24:]
            eval_preds = {k: v[24:] for k, v in predictions.items()}
            ensemble.compute_weights_from_validation(eval_preds, eval_y)
    else:
        # 默认等权重
        n_models = len(ensemble.models)
        if n_models > 0:
            default_weights = {name: 1.0 / n_models for name in ensemble.models.keys()}
            ensemble.set_weights(default_weights)

    return ensemble


if __name__ == "__main__":
    # 演示集成预测
    logging.basicConfig(level=logging.INFO)

    # 模拟三个模型的预测
    np.random.seed(42)
    n_samples = 100

    xgb_pred = np.random.normal(2000, 100, n_samples)
    lgbm_pred = np.random.normal(2010, 90, n_samples)
    lstm_pred = np.random.normal(1995, 110, n_samples)
    true_values = np.random.normal(2000, 100, n_samples)

    # 创建集成
    ensemble = EnsemblePredictor()
    ensemble.add_model('xgboost', None)  # 模拟占位
    ensemble.add_model('lightgbm', None)
    ensemble.add_model('lstm', None)

    # 模拟带权重的预测
    weights = {'xgboost': 0.4, 'lightgbm': 0.4, 'lstm': 0.2}
    ensemble.set_weights(weights)

    # 计算模拟的集成预测
    ensemble_pred = (
        weights['xgboost'] * xgb_pred +
        weights['lightgbm'] * lgbm_pred +
        weights['lstm'] * lstm_pred
    )

    # 评估
    from src.model.evaluator import calculate_metrics

    print("\n各模型表现:")
    for name, preds in [('XGBoost', xgb_pred), ('LightGBM', lgbm_pred), ('LSTM', lstm_pred)]:
        metrics = calculate_metrics(true_values, preds)
        print(f"  {name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

    print("\n集成预测表现:")
    metrics = calculate_metrics(true_values, ensemble_pred)
    print(f"  集成: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")
