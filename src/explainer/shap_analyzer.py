"""
SHAP特征贡献分析

功能：
- 使用shap.TreeExplainer解释XGBoost
- 单次预测的SHAP值分析
- 全局特征重要性分析
- 返回结构化的特征贡献数据
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap

from config import settings
from ..model.predictor import get_model

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP特征贡献分析器
    """

    def __init__(self):
        """初始化分析器"""
        self.model_data = get_model()
        self.model = self.model_data["model"]
        self.feature_columns = self.model_data["training_info"]["feature_columns"]

        # 创建TreeExplainer
        logger.info("初始化SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

        # 计算期望值（基础值）
        self.expected_value = self.explainer.expected_value

    def analyze_single(
        self,
        features: Union[Dict, pd.DataFrame],
        return_raw: bool = False
    ) -> List[Dict]:
        """
        分析单次预测的SHAP值

        Args:
            features: 特征字典或单行DataFrame
            return_raw: 是否返回原始SHAP值

        Returns:
            特征贡献列表
        """
        # 转换输入格式
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features

        # 确保特征列顺序正确
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_columns].values

        # 计算SHAP值
        shap_values = self.explainer.shap_values(X)

        # 如果是批量预测，取第一行
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        # 构建结果
        contributions = []
        for i, col in enumerate(self.feature_columns):
            contribution = {
                "feature": col,
                "shap_value": float(shap_values[i]),
                "feature_value": float(df[col].values[0]),
                "direction": "positive" if shap_values[i] > 0 else "negative",
                "abs_contribution": float(abs(shap_values[i])),
            }
            contributions.append(contribution)

        # 按绝对值排序
        contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)

        if return_raw:
            return contributions, shap_values

        return contributions

    def analyze_global(
        self,
        X: Optional[pd.DataFrame] = None,
        n_samples: int = 100
    ) -> Dict:
        """
        全局特征重要性分析

        Args:
            X: 特征数据（如果为None，使用训练数据的采样）
            n_samples: 采样数量

        Returns:
            全局重要性字典
        """
        if X is None:
            logger.info(f"未提供数据，使用随机采样 {n_samples} 个样本")
            # 创建随机特征样本
            X = self._create_random_samples(n_samples)
        else:
            # 确保列顺序正确
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]

            # 限制样本数量
            if len(X) > n_samples:
                X = X.sample(n=n_samples, random_state=42)

        # 计算SHAP值
        shap_values = self.explainer.shap_values(X.values)

        # 计算全局重要性（平均绝对SHAP值）
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # 构建结果
        importance = {}
        for i, col in enumerate(self.feature_columns):
            importance[col] = {
                "mean_abs_shap": float(mean_abs_shap[i]),
                "std_shap": float(np.std(shap_values[:, i])),
                "positive_ratio": float(np.mean(shap_values[:, i] > 0)),
                "feature_value_mean": float(X[col].mean()),
            }

        # 排序
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1]["mean_abs_shap"], reverse=True)
        )

        return importance

    def analyze_interaction(
        self,
        features: Union[Dict, pd.DataFrame],
        feature1: str,
        feature2: str
    ) -> np.ndarray:
        """
        分析两个特征之间的交互效应

        Args:
            features: 特征数据
            feature1: 第一个特征名
            feature2: 第二个特征名

        Returns:
            交互SHAP值矩阵
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_columns].values

        # 使用TreeExplainer的交互功能
        shap_interaction = self.explainer.shap_interaction_values(X)

        # 获取特征索引
        idx1 = self.feature_columns.index(feature1)
        idx2 = self.feature_columns.index(feature2)

        # 返回交互值
        if len(shap_interaction.shape) == 3:
            return shap_interaction[0, idx1, idx2]
        else:
            return shap_interaction[:, idx1, idx2]

    def _create_random_samples(self, n_samples: int) -> pd.DataFrame:
        """
        创建随机特征样本（用于全局分析）

        Args:
            n_samples: 样本数量

        Returns:
            随机特征DataFrame
        """
        np.random.seed(42)

        data = {}
        for col in self.feature_columns:
            # 根据特征类型生成合理的随机值
            if col == "hour":
                data[col] = np.random.randint(0, 24, n_samples)
            elif col == "day_of_week":
                data[col] = np.random.randint(0, 7, n_samples)
            elif col in ["is_weekend", "is_holiday", "is_workday"]:
                data[col] = np.random.randint(0, 2, n_samples)
            elif col in ["month", "day_of_month"]:
                data[col] = np.random.randint(1, 13 if col == "month" else 32, n_samples)
            elif col == "temperature":
                data[col] = np.random.normal(20, 10, n_samples)
            elif col == "humidity":
                data[col] = np.random.uniform(30, 90, n_samples)
            elif "lag" in col or "load" in col:
                data[col] = np.random.normal(2000, 500, n_samples)
            elif "_sin" in col or "_cos" in col:
                data[col] = np.random.uniform(-1, 1, n_samples)
            else:
                data[col] = np.random.normal(0, 1, n_samples)

        return pd.DataFrame(data)

    def get_expected_value(self) -> float:
        """
        获取期望值（基础预测值）

        Returns:
            期望值
        """
        if isinstance(self.expected_value, (list, np.ndarray)):
            return float(self.expected_value[0])
        return float(self.expected_value)


# 全局分析器实例
_analyzer: Optional[SHAPAnalyzer] = None


def get_analyzer() -> SHAPAnalyzer:
    """
    获取SHAP分析器实例（带缓存）

    Returns:
        SHAPAnalyzer实例
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SHAPAnalyzer()
    return _analyzer


def explain_prediction(
    prediction: Union[float, Dict],
    features: Union[Dict, pd.DataFrame]
) -> List[Dict]:
    """
    解释单次预测

    Args:
        prediction: 预测值（可以是数值或包含预测值的字典）
        features: 特征数据

    Returns:
        特征贡献列表
    """
    analyzer = get_analyzer()
    contributions = analyzer.analyze_single(features)

    return contributions


def explain_global(
    X: Optional[pd.DataFrame] = None,
    n_samples: int = 100
) -> Dict:
    """
    全局特征重要性分析

    Args:
        X: 特征数据（可选）
        n_samples: 采样数量

    Returns:
        全局重要性字典
    """
    analyzer = get_analyzer()
    return analyzer.analyze_global(X, n_samples)


def get_feature_contribution(
    contributions: List[Dict],
    top_n: int = 5
) -> Dict:
    """
    获取主要特征贡献

    Args:
        contributions: 特征贡献列表
        top_n: 返回前N个最重要的特征

    Returns:
        主要贡献摘要
    """
    top_contributors = contributions[:top_n]

    positive_contrib = [c for c in contributions if c["direction"] == "positive"]
    negative_contrib = [c for c in contributions if c["direction"] == "negative"]

    total_positive = sum(c["shap_value"] for c in positive_contrib)
    total_negative = sum(abs(c["shap_value"]) for c in negative_contrib)

    return {
        "top_contributors": top_contributors,
        "n_positive": len(positive_contrib),
        "n_negative": len(negative_contrib),
        "total_positive_contribution": round(total_positive, 2),
        "total_negative_contribution": round(total_negative, 2),
        "net_contribution": round(total_positive - total_negative, 2),
    }


def format_shap_values(contributions: List[Dict]) -> pd.DataFrame:
    """
    将SHAP值格式化为DataFrame

    Args:
        contributions: 特征贡献列表

    Returns:
        格式化后的DataFrame
    """
    return pd.DataFrame(contributions)


if __name__ == "__main__":
    # 演示SHAP分析
    logging.basicConfig(level=logging.INFO)

    # 示例特征
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

    print("\nSHAP特征贡献分析:")
    print("-" * 60)
    for i, contrib in enumerate(contributions[:10]):
        direction = "↑" if contrib["direction"] == "positive" else "↓"
        print(
            f"{i+1:2d}. {contrib['feature']:<30} "
            f"{direction} {contrib['shap_value']:>8.2f} "
            f"(值: {contrib['feature_value']:>8.2f})"
        )

    # 获取主要贡献
    summary = get_feature_contribution(contributions)
    print(f"\n正向贡献总量: {summary['total_positive_contribution']:.2f}")
    print(f"负向贡献总量: {summary['total_negative_contribution']:.2f}")
    print(f"净贡献: {summary['net_contribution']:.2f}")


    def compute_shap_interaction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature1: str,
        feature2: str,
        sample_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        计算SHAP交互值（批量支持）

        Args:
            X: 特征数据
            feature1: 第一个特征名
            feature2: 第二个特征名
            sample_indices: 采样索引（可选）

        Returns:
            交互SHAP值数组
        """
        if isinstance(X, pd.DataFrame):
            # 确保列顺序
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X_values = X[self.feature_columns].values
        else:
            X_values = X

        # 采样
        if sample_indices is not None:
            X_values = X_values[sample_indices]

        # 计算交互值
        shap_interaction = self.explainer.shap_interaction_values(X_values)

        # 获取特征索引
        idx1 = self.feature_columns.index(feature1)
        idx2 = self.feature_columns.index(feature2)

        # 返回交互矩阵
        if len(shap_interaction.shape) == 3:
            # 单样本情况
            return shap_interaction[0, :, :]
        else:
            # 多样本情况，返回平均交互
            return np.mean(shap_interaction[:, :, :], axis=0)

    def compute_friedman_h(
        self,
        X: pd.DataFrame,
        feature1: str,
        feature2: str,
        n_samples: int = 500
    ) -> float:
        """
        计算Friedman's H统计量（衡量交互强度）

        基于SHAP交互值的二阶交叉项系数

        Args:
            X: 特征数据
            feature1: 第一个特征名
            feature2: 第二个特征名
            n_samples: 采样数量

        Returns:
            H统计量值
        """
        # 采样
        if len(X) > n_samples:
            X_sample = X.sample(n=n_samples, random_state=42)
        else:
            X_sample = X

        # 确保列存在
        for col in self.feature_columns:
            if col not in X_sample.columns:
                X_sample[col] = 0

        X_values = X_sample[self.feature_columns].values

        # 计算SHAP交互值
        shap_interaction = self.explainer.shap_interaction_values(X_values)

        # 获取特征索引
        idx1 = self.feature_columns.index(feature1)
        idx2 = self.feature_columns.index(feature2)

        # 获取双向交互值
        interaction_12 = shap_interaction[:, idx1, idx2]
        interaction_21 = shap_interaction[:, idx2, idx1]

        # 计算H统计量
        # H = mean(|v_ij * v_ji|) / (mean(v_ii^2 + v_jj^2) + mean(|v_ij * v_ji|))
        numerator = np.mean(np.abs(interaction_12 * interaction_21))
        denominator = np.mean(
            shap_interaction[:, idx1, idx1]**2 + 
            shap_interaction[:, idx2, idx2]**2 + 
            np.abs(interaction_12 * interaction_21)
        )

        if denominator == 0:
            return 0.0

        h_statistic = numerator / denominator

        return float(h_statistic)

    def compute_all_interactions(
        self,
        X: pd.DataFrame,
        top_n: int = 10,
        n_samples: int = 200
    ) -> Dict[str, float]:
        """
        计算所有特征对的Friedman H统计量

        Args:
            X: 特征数据
            top_n: 返回最重要的top_n个交互
            n_samples: 采样数量

        Returns:
            交互强度字典 {特征对: H值}
        """
        # 获取top特征
        importance = self.analyze_global(X, n_samples=n_samples)
        top_features = list(importance.keys())[:min(top_n, len(importance))]

        interactions = {}

        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                h = self.compute_friedman_h(X, feat1, feat2, n_samples)
                key = f"{feat1} x {feat2}"
                interactions[key] = h

        # 排序
        interactions = dict(
            sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        )

        return interactions

