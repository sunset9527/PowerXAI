"""
时序交叉验证器

使用sklearn的TimeSeriesSplit进行时序数据的交叉验证。

功能：
- 支持XGBoost/LightGBM/LSTM三种模型
- 时间序列感知的交叉验证
- 详细的验证指标报告
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """
    时序交叉验证器

    使用滑动窗口方式进行交叉验证，保证时间顺序不泄露。
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None
    ):
        """
        初始化交叉验证器

        Args:
            n_splits: 折数
            test_size: 测试集大小（样本数或比例）
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.cv_results: List[Dict] = []

    def _create_splits(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        创建交叉验证分割

        Args:
            X: 特征数组
            y: 目标数组

        Returns:
            分割列表 [(X_train, X_test, y_train, y_test), ...]
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        splits = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

    def cross_validate(
        self,
        model_class: str,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        执行交叉验证

        Args:
            model_class: 模型类型 ('xgboost', 'lightgbm', 'lstm')
            X: 特征数组
            y: 目标数组
            model_params: 模型超参数

        Returns:
            每折的验证指标列表
        """
        from src.model.evaluator import calculate_metrics

        logger.info(f"开始{self.n_splits}折时序交叉验证，模型类型: {model_class}")

        model_params = model_params or {}

        splits = self._create_splits(X, y)
        results = []

        for fold, (X_train, X_test, y_train, y_test) in enumerate(splits):
            logger.info(f"\n--- Fold {fold + 1}/{self.n_splits} ---")
            logger.info(f"训练集: {len(X_train)} 样本")
            logger.info(f"测试集: {len(X_test)} 样本")

            # 训练模型
            if model_class == 'xgboost':
                from src.model.trainer import train_xgboost
                model = train_xgboost(X_train, y_train, X_test, y_test, model_params)
                predictions = model.predict(X_test)

            elif model_class == 'lightgbm':
                from src.model.trainer import train_lightgbm
                model = train_lightgbm(X_train, y_train, X_test, y_test, model_params)
                predictions = model.predict(X_test)

            elif model_class == 'lstm':
                from src.model.lstm_model import create_lstm_predictor
                predictor = create_lstm_predictor(model_params)
                predictor.train(X_train, y_train, X_test, y_test)
                predictions = predictor.predict(X_test)

                # LSTM预测结果长度会短于输入（序列长度）
                y_test_aligned = y_test[24:]
            else:
                raise ValueError(f"不支持的模型类型: {model_class}")

            # 对齐预测和真实值
            if model_class != 'lstm':
                y_test_aligned = y_test

            # 计算指标
            metrics = calculate_metrics(y_test_aligned, predictions)

            fold_result = {
                'fold': fold + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': metrics
            }

            results.append(fold_result)

            logger.info(
                f"Fold {fold + 1} - MAE: {metrics['MAE']:.2f}, "
                f"RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}"
            )

        self.cv_results = results
        return results

    def get_average_metrics(self) -> Dict[str, float]:
        """
        获取平均验证指标

        Returns:
            平均指标字典
        """
        if not self.cv_results:
            raise ValueError("请先执行cross_validate方法")

        all_metrics = {}

        for result in self.cv_results:
            for metric_name, value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[f'avg_{metric_name}'] = float(np.mean(values))
            avg_metrics[f'std_{metric_name}'] = float(np.std(values))

        return avg_metrics

    def get_summary(self) -> str:
        """
        获取交叉验证摘要

        Returns:
            格式化的摘要字符串
        """
        if not self.cv_results:
            return "尚无交叉验证结果"

        avg_metrics = self.get_average_metrics()

        lines = [
            "=" * 60,
            "时序交叉验证摘要",
            "=" * 60,
            f"\n验证配置:",
            f"  折数: {self.n_splits}",
            f"  总样本数: {sum(r['train_size'] + r['test_size'] for r in self.cv_results)}",
            f"\n各折结果:",
        ]

        for result in self.cv_results:
            metrics = result['metrics']
            lines.append(
                f"  Fold {result['fold']}: "
                f"MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
                f"R²={metrics['R2']:.4f}"
            )

        lines.extend([
            f"\n平均指标 (±标准差):",
            f"  MAE:  {avg_metrics['avg_MAE']:.2f} ± {avg_metrics['std_MAE']:.2f}",
            f"  RMSE: {avg_metrics['avg_RMSE']:.2f} ± {avg_metrics['std_RMSE']:.2f}",
            f"  MAPE: {avg_metrics['avg_MAPE']:.2f}% ± {avg_metrics['std_MAPE']:.2f}%",
            f"  R²:   {avg_metrics['avg_R2']:.4f} ± {avg_metrics['std_R2']:.4f}",
            "=" * 60,
        ])

        return "\n".join(lines)


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    model_types: List[str] = None,
    n_splits: int = 5
) -> Dict[str, Dict]:
    """
    对多个模型进行交叉验证比较

    Args:
        X: 特征数组
        y: 目标数组
        model_types: 模型类型列表
        n_splits: 折数

    Returns:
        各模型的验证结果字典
    """
    if model_types is None:
        model_types = ['xgboost', 'lightgbm']

    results = {}

    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"验证模型: {model_type.upper()}")
        logger.info('='*50)

        cv = TimeSeriesCrossValidator(n_splits=n_splits)

        try:
            cv.cross_validate(model_type, X, y)
            avg_metrics = cv.get_average_metrics()

            results[model_type] = {
                'cv_results': cv.cv_results,
                'avg_metrics': avg_metrics,
                'summary': cv.get_summary()
            }

            logger.info(f"\n{model_type.upper()} 平均指标:")
            logger.info(f"  MAE: {avg_metrics['avg_MAE']:.2f}")
            logger.info(f"  RMSE: {avg_metrics['avg_RMSE']:.2f}")
            logger.info(f"  R²: {avg_metrics['avg_R2']:.4f}")

        except Exception as e:
            logger.error(f"{model_type} 验证失败: {e}")
            results[model_type] = {'error': str(e)}

    return results


def compare_cv_results(results: Dict[str, Dict]) -> str:
    """
    比较多个模型的交叉验证结果

    Args:
        results: 交叉验证结果字典

    Returns:
        比较结果字符串
    """
    lines = [
        "=" * 70,
        "模型交叉验证对比",
        "=" * 70,
        f"\n{'模型':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE':<10}",
        "-" * 70,
    ]

    best_mae = float('inf')
    best_model = None

    for model_name, result in results.items():
        if 'avg_metrics' in result:
            metrics = result['avg_metrics']
            mae = metrics['avg_MAE']
            rmse = metrics['avg_RMSE']
            r2 = metrics['avg_R2']
            mape = metrics['avg_MAPE']

            lines.append(
                f"{model_name:<15} {mae:<12.2f} {rmse:<12.2f} "
                f"{r2:<12.4f} {mape:<10.2f}%"
            )

            if mae < best_mae:
                best_mae = mae
                best_model = model_name

    lines.extend([
        "-" * 70,
        f"\n最佳模型 (MAE): {best_model}",
        "=" * 70,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # 演示交叉验证
    logging.basicConfig(level=logging.INFO)

    from src.data.generator import ensure_data_exists
    from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns

    # 加载数据
    raw_df = ensure_data_exists()
    processed_df, _ = preprocess_data(raw_df)

    # 获取特征
    feature_cols = get_feature_columns(processed_df)
    X = processed_df[feature_cols].values
    y = processed_df['load'].values

    # 使用前1000个样本快速验证
    X_sample = X[:1000]
    y_sample = y[:1000]

    # 交叉验证
    results = cross_validate_models(X_sample, y_sample, model_types=['xgboost', 'lightgbm'])

    # 比较结果
    print("\n" + compare_cv_results(results))
