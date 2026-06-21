"""
PDP (Partial Dependence Plot) 分析

功能：
- 计算单特征PDP
- 计算双特征2D PDP
- 标准化输出格式
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PDPAnalyzer:
    """
    PDP（偏依赖图）分析器
    """

    def __init__(self, model):
        """
        初始化PDP分析器

        Args:
            model: 训练好的模型（需支持predict方法）
        """
        self.model = model

    def compute_pdp(
        self,
        X: pd.DataFrame,
        feature: str,
        grid_resolution: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        计算单特征PDP

        Args:
            X: 特征数据DataFrame
            feature: 特征名称
            grid_resolution: 网格分辨率

        Returns:
            包含feature_values和average_prediction的字典
        """
        if feature not in X.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")

        # 获取特征值范围
        feature_values = np.linspace(
            X[feature].min(),
            X[feature].max(),
            grid_resolution
        )

        # 保存原始特征值
        original_values = X[feature].copy()

        # 对每个网格点计算平均预测
        predictions = []
        for val in feature_values:
            # 替换特征值
            X_copy = X.copy()
            X_copy[feature] = val

            # 预测
            preds = self.model.predict(X_copy)
            predictions.append(float(np.mean(preds)))

        # 恢复原始值
        X[feature] = original_values

        return {
            "feature": feature,
            "feature_values": feature_values,
            "average_prediction": np.array(predictions),
            "feature_std": float(X[feature].std()),
        }

    def compute_2d_pdp(
        self,
        X: pd.DataFrame,
        feature1: str,
        feature2: str,
        grid_resolution: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        计算双特征2D PDP

        Args:
            X: 特征数据DataFrame
            feature1: 第一个特征名称
            feature2: 第二个特征名称
            grid_resolution: 网格分辨率

        Returns:
            包含网格数据和预测值的字典
        """
        if feature1 not in X.columns or feature2 not in X.columns:
            raise ValueError(f"特征不存在于数据中")

        # 获取特征值范围
        values1 = np.linspace(
            X[feature1].min(),
            X[feature1].max(),
            grid_resolution
        )
        values2 = np.linspace(
            X[feature2].min(),
            X[feature2].max(),
            grid_resolution
        )

        # 保存原始特征值
        original1 = X[feature1].copy()
        original2 = X[feature2].copy()

        # 创建网格预测矩阵
        grid_predictions = np.zeros((grid_resolution, grid_resolution))

        for i, val1 in enumerate(values1):
            for j, val2 in enumerate(values2):
                X_copy = X.copy()
                X_copy[feature1] = val1
                X_copy[feature2] = val2

                preds = self.model.predict(X_copy)
                grid_predictions[i, j] = np.mean(preds)

        # 恢复原始值
        X[feature1] = original1
        X[feature2] = original2

        return {
            "feature1": feature1,
            "feature2": feature2,
            "grid_values1": values1,
            "grid_values2": values2,
            "grid_predictions": grid_predictions,
        }

    def compute_pdp_with_ci(
        self,
        X: pd.DataFrame,
        feature: str,
        grid_resolution: int = 50,
        n_bootstrap: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        计算带置信区间的PDP

        Args:
            X: 特征数据DataFrame
            feature: 特征名称
            grid_resolution: 网格分辨率
            n_bootstrap: Bootstrap次数

        Returns:
            包含预测值和置信区间的字典
        """
        # 计算基础PDP
        base_pdp = self.compute_pdp(X, feature, grid_resolution)

        # Bootstrap采样计算置信区间
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            # 随机采样
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[sample_idx].copy()

            feature_values = base_pdp["feature_values"]
            predictions = []

            for val in feature_values:
                X_copy = X_sample.copy()
                X_copy[feature] = val
                preds = self.model.predict(X_copy)
                predictions.append(float(np.mean(preds)))

            bootstrap_preds.append(predictions)

        bootstrap_preds = np.array(bootstrap_preds)

        return {
            "feature": feature,
            "feature_values": base_pdp["feature_values"],
            "average_prediction": base_pdp["average_prediction"],
            "ci_lower": np.percentile(bootstrap_preds, 2.5, axis=0),
            "ci_upper": np.percentile(bootstrap_preds, 97.5, axis=0),
            "std": np.std(bootstrap_preds, axis=0),
        }

    def format_pdp_data(
        self,
        pdp_result: Dict
    ) -> pd.DataFrame:
        """
        将PDP结果格式化为DataFrame

        Args:
            pdp_result: compute_pdp的返回结果

        Returns:
            格式化后的DataFrame
        """
        df = pd.DataFrame({
            "feature_value": pdp_result["feature_values"],
            "predicted_value": pdp_result["average_prediction"],
        })

        if "ci_lower" in pdp_result:
            df["ci_lower"] = pdp_result["ci_lower"]
            df["ci_upper"] = pdp_result["ci_upper"]

        return df

    def format_2d_pdp_data(
        self,
        pdp_2d_result: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将2D PDP结果格式化

        Args:
            pdp_2d_result: compute_2d_pdp的返回结果

        Returns:
            (grid1, grid2, predictions)元组
        """
        grid1, grid2 = np.meshgrid(
            pdp_2d_result["grid_values1"],
            pdp_2d_result["grid_values2"],
            indexing="ij"
        )

        return (
            grid1,
            grid2,
            pdp_2d_result["grid_predictions"]
        )


# 全局分析器实例
_analyzer: Optional[PDPAnalyzer] = None


def get_analyzer(model=None) -> PDPAnalyzer:
    """
    获取PDP分析器实例（带缓存）

    Args:
        model: 模型实例

    Returns:
        PDPAnalyzer实例
    """
    global _analyzer
    if _analyzer is None:
        if model is None:
            raise ValueError("首次调用需要提供model")
        _analyzer = PDPAnalyzer(model)
    return _analyzer


def compute_partial_dependence(
    model,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 50
) -> Dict:
    """
    便捷函数：计算偏依赖

    Args:
        model: 模型
        X: 特征数据
        feature: 特征名称
        grid_resolution: 网格分辨率

    Returns:
        PDP结果字典
    """
    analyzer = get_analyzer(model)
    return analyzer.compute_pdp(X, feature, grid_resolution)


if __name__ == "__main__":
    # 演示PDP分析
    logging.basicConfig(level=logging.INFO)

    from sklearn.ensemble import GradientBoostingRegressor

    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "hour": np.random.randint(0, 24, n),
        "temperature": np.random.normal(25, 10, n),
        "humidity": np.random.uniform(30, 90, n),
        "load_lag_1h": np.random.normal(2500, 500, n),
    })
    y = 2000 + X["hour"] * 10 + X["temperature"] * 5 + np.random.randn(n) * 100

    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    analyzer = PDPAnalyzer(model)

    # 计算温度的PDP
    pdp_result = analyzer.compute_pdp(X, "temperature", grid_resolution=20)
    print(f"\n温度PDP (前5个点):")
    for i in range(5):
        print(f"  温度={pdp_result['feature_values'][i]:.1f}°C → 预测={pdp_result['average_prediction'][i]:.2f}")

    # 计算2D PDP
    pdp_2d = analyzer.compute_2d_pdp(X, "temperature", "humidity", grid_resolution=10)
    print(f"\n2D PDP形状: {pdp_2d['grid_predictions'].shape}")
    print(f"预测范围: {pdp_2d['grid_predictions'].min():.2f} - {pdp_2d['grid_predictions'].max():.2f}")
