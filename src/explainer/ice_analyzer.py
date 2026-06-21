"""
ICE (Individual Conditional Expectation) 分析

功能：
- 计算ICE曲线
- 支持采样和分组
- 标准化输出格式
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ICEAnalyzer:
    """
    ICE（个体条件期望）分析器
    """

    def __init__(self, model):
        """
        初始化ICE分析器

        Args:
            model: 训练好的模型（需支持predict方法）
        """
        self.model = model

    def compute_ice(
        self,
        X: pd.DataFrame,
        feature: str,
        sample_indices: Optional[List[int]] = None,
        grid_resolution: int = 50,
        sample_size: int = 50
    ) -> Dict:
        """
        计算ICE曲线

        Args:
            X: 特征数据DataFrame
            feature: 特征名称
            sample_indices: 指定样本索引（可选）
            grid_resolution: 网格分辨率
            sample_size: 采样数量（当sample_indices为None时使用）

        Returns:
            包含ICE曲线数据的字典
        """
        if feature not in X.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")

        # 采样
        if sample_indices is None:
            if len(X) > sample_size:
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
            else:
                sample_indices = list(range(len(X)))

        X_sample = X.iloc[sample_indices].copy()

        # 获取特征值范围
        feature_values = np.linspace(
            X[feature].min(),
            X[feature].max(),
            grid_resolution
        )

        # 计算每条ICE曲线
        ice_curves = []
        for idx in range(len(X_sample)):
            predictions = []
            for val in feature_values:
                X_copy = X_sample.iloc[[idx]].copy()
                X_copy[feature] = val
                pred = self.model.predict(X_copy)
                predictions.append(float(pred[0]))
            ice_curves.append(predictions)

        ice_curves = np.array(ice_curves)

        # 计算统计量
        mean_curve = np.mean(ice_curves, axis=0)
        std_curve = np.std(ice_curves, axis=0)

        return {
            "feature": feature,
            "feature_values": feature_values,
            "ice_curves": ice_curves,
            "sample_indices": sample_indices,
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "n_samples": len(sample_indices),
        }

    def compute_centered_ice(
        self,
        X: pd.DataFrame,
        feature: str,
        sample_indices: Optional[List[int]] = None,
        grid_resolution: int = 50,
        sample_size: int = 50
    ) -> Dict:
        """
        计算中心化ICE曲线（从0开始）

        Args:
            X: 特征数据DataFrame
            feature: 特征名称
            sample_indices: 指定样本索引
            grid_resolution: 网格分辨率
            sample_size: 采样数量

        Returns:
            包含中心化ICE曲线数据的字典
        """
        ice_result = self.compute_ice(
            X, feature, sample_indices, grid_resolution, sample_size
        )

        # 中心化：每条曲线减去第一个值
        centered_curves = ice_result["ice_curves"] - ice_result["ice_curves"][:, 0:1]
        mean_centered = np.mean(centered_curves, axis=0)

        return {
            "feature": feature,
            "feature_values": ice_result["feature_values"],
            "ice_curves": centered_curves,
            "sample_indices": ice_result["sample_indices"],
            "mean_curve": mean_centered,
            "std_curve": ice_result["std_curve"],
            "n_samples": ice_result["n_samples"],
            "centered": True,
        }

    def compute_decile_ice(
        self,
        X: pd.DataFrame,
        feature: str,
        grid_resolution: int = 50,
        n_deciles: int = 10
    ) -> Dict:
        """
        计算分位数ICE曲线（按预测值分桶）

        Args:
            X: 特征数据DataFrame
            feature: 特征名称
            grid_resolution: 网格分辨率
            n_deciles: 分位数数量

        Returns:
            包含分组ICE曲线数据的字典
        """
        # 获取原始预测值用于分组
        predictions = self.model.predict(X)

        # 按预测值分桶
        decile_labels = pd.qcut(predictions, n_deciles, labels=False, duplicates="drop")

        decile_curves = {}
        decile_means = {}

        for decile in range(n_deciles):
            indices = np.where(decile_labels == decile)[0]

            # 计算该分位数的ICE曲线
            ice_result = self.compute_ice(
                X, feature,
                sample_indices=list(indices),
                grid_resolution=grid_resolution,
                sample_size=len(indices)
            )

            decile_curves[decile] = ice_result["ice_curves"]
            decile_means[decile] = ice_result["mean_curve"]

        return {
            "feature": feature,
            "feature_values": ice_result["feature_values"],
            "decile_curves": decile_curves,
            "decile_means": decile_means,
            "n_deciles": n_deciles,
        }

    def format_ice_data(
        self,
        ice_result: Dict
    ) -> pd.DataFrame:
        """
        将ICE结果格式化为DataFrame

        Args:
            ice_result: compute_ice的返回结果

        Returns:
            格式化后的DataFrame
        """
        records = []
        for i, idx in enumerate(ice_result["sample_indices"]):
            for j, val in enumerate(ice_result["feature_values"]):
                records.append({
                    "sample_id": idx,
                    "feature_value": val,
                    "predicted_value": ice_result["ice_curves"][i, j],
                })

        return pd.DataFrame(records)

    def format_summary_data(
        self,
        ice_result: Dict
    ) -> pd.DataFrame:
        """
        将ICE汇总数据格式化为DataFrame

        Args:
            ice_result: compute_ice的返回结果

        Returns:
            汇总DataFrame
        """
        return pd.DataFrame({
            "feature_value": ice_result["feature_values"],
            "mean_prediction": ice_result["mean_curve"],
            "std_prediction": ice_result["std_curve"],
        })

    def get_ice_for_samples(
        self,
        ice_result: Dict,
        sample_ids: List[int]
    ) -> pd.DataFrame:
        """
        获取特定样本的ICE曲线

        Args:
            ice_result: compute_ice的返回结果
            sample_ids: 样本ID列表

        Returns:
            特定样本的ICE数据
        """
        records = []
        for sid in sample_ids:
            if sid in ice_result["sample_indices"]:
                idx = ice_result["sample_indices"].index(sid)
                for j, val in enumerate(ice_result["feature_values"]):
                    records.append({
                        "sample_id": sid,
                        "feature_value": val,
                        "predicted_value": ice_result["ice_curves"][idx, j],
                    })

        return pd.DataFrame(records)


# 全局分析器实例
_analyzer: Optional[ICEAnalyzer] = None


def get_analyzer(model=None) -> ICEAnalyzer:
    """
    获取ICE分析器实例（带缓存）

    Args:
        model: 模型实例

    Returns:
        ICEAnalyzer实例
    """
    global _analyzer
    if _analyzer is None:
        if model is None:
            raise ValueError("首次调用需要提供model")
        _analyzer = ICEAnalyzer(model)
    return _analyzer


def compute_individual_conditional_expectation(
    model,
    X: pd.DataFrame,
    feature: str,
    sample_indices: Optional[List[int]] = None,
    grid_resolution: int = 50,
    sample_size: int = 50
) -> Dict:
    """
    便捷函数：计算ICE曲线

    Args:
        model: 模型
        X: 特征数据
        feature: 特征名称
        sample_indices: 指定样本索引
        grid_resolution: 网格分辨率
        sample_size: 采样数量

    Returns:
        ICE结果字典
    """
    analyzer = get_analyzer(model)
    return analyzer.compute_ice(X, feature, sample_indices, grid_resolution, sample_size)


if __name__ == "__main__":
    # 演示ICE分析
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

    analyzer = ICEAnalyzer(model)

    # 计算温度的ICE
    ice_result = analyzer.compute_ice(X, "temperature", sample_size=20, grid_resolution=20)

    print(f"\nICE分析 - 温度特征:")
    print(f"样本数量: {ice_result['n_samples']}")
    print(f"特征值范围: {ice_result['feature_values'][0]:.1f} - {ice_result['feature_values'][-1]:.1f}")
    print(f"\n平均曲线 (前5个点):")
    for i in range(5):
        print(f"  温度={ice_result['feature_values'][i]:.1f}°C → 预测={ice_result['mean_curve'][i]:.2f} ± {ice_result['std_curve'][i]:.2f}")

    # 中心化ICE
    centered_result = analyzer.compute_centered_ice(X, "temperature", sample_size=20, grid_resolution=20)
    print(f"\n中心化ICE (第一条曲线前5个点):")
    for i in range(5):
        print(f"  {centered_result['ice_curves'][0, i]:.2f}")
