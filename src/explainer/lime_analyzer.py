"""
LIME特征贡献分析

功能：
- 使用LIME解释单次预测
- 返回与SHAP一致的标准化格式
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME未安装，将使用降级实现")


class LIMEAnalyzer:
    """
    LIME特征贡献分析器
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        training_data: Optional[pd.DataFrame] = None,
        mode: str = "regression"
    ):
        """
        初始化LIME分析器

        Args:
            model: 训练好的模型（需支持predict方法）
            feature_columns: 特征列名列表
            training_data: 训练数据DataFrame（用于创建LIME解释器）
            mode: 模式，"regression"或"classification"
        """
        self.model = model
        self.feature_columns = feature_columns
        self.mode = mode

        if not LIME_AVAILABLE:
            logger.warning("LIME未安装，将使用简化实现")
            self.explainer = None
            return

        # 准备训练数据
        if training_data is not None:
            X_train = training_data[feature_columns].values
        else:
            # 创建随机训练数据
            logger.info("未提供训练数据，使用随机数据初始化LIME")
            X_train = self._create_random_training_data(1000)

        # 创建LIME TabularExplainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_columns,
            mode=mode,
            random_state=42
        )

    def _create_random_training_data(self, n_samples: int) -> np.ndarray:
        """
        创建随机训练数据

        Args:
            n_samples: 样本数量

        Returns:
            随机特征数组
        """
        np.random.seed(42)
        data = np.zeros((n_samples, len(self.feature_columns)))

        for i, col in enumerate(self.feature_columns):
            if "hour" in col:
                data[:, i] = np.random.randint(0, 24, n_samples)
            elif "day_of_week" in col:
                data[:, i] = np.random.randint(0, 7, n_samples)
            elif "is_weekend" in col or "is_holiday" in col or "is_workday" in col:
                data[:, i] = np.random.randint(0, 2, n_samples)
            elif "month" in col:
                data[:, i] = np.random.randint(1, 13, n_samples)
            elif "day_of_month" in col:
                data[:, i] = np.random.randint(1, 32, n_samples)
            elif "temperature" in col or "temp" in col:
                data[:, i] = np.random.normal(20, 10, n_samples)
            elif "humidity" in col:
                data[:, i] = np.random.uniform(30, 90, n_samples)
            elif "load" in col:
                data[:, i] = np.random.normal(2500, 500, n_samples)
            else:
                data[:, i] = np.random.normal(0, 1, n_samples)

        return data

    def explain_single(
        self,
        features: Union[Dict, pd.DataFrame, np.ndarray],
        num_features: int = 10
    ) -> "exp":
        """
        解释单次预测

        Args:
            features: 特征数据（字典/DataFrame/数组）
            num_features: 显示的特征数量

        Returns:
            LIME解释对象
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME未安装，请运行: pip install lime")

        # 转换输入格式
        if isinstance(features, dict):
            X = pd.DataFrame([features])[self.feature_columns].values
        elif isinstance(features, pd.DataFrame):
            X = features[self.feature_columns].values
        else:
            X = features

        # 确保是二维数组
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # 生成解释
        def predict_fn(x):
            """预测函数包装"""
            return self.model.predict(x)

        exp = self.explainer.explain_instance(
            X[0],
            predict_fn,
            num_features=num_features,
            num_samples=500
        )

        return exp

    def format_explanation(self, exp) -> List[Dict]:
        """
        将LIME解释转换为标准化格式（与SHAP一致）

        Args:
            exp: LIME解释对象

        Returns:
            标准化格式的特征贡献列表
        """
        # 获取特征和贡献
        explanation_list = exp.as_list()

        contributions = []
        for feature, value in explanation_list:
            # 处理特征名称（可能是索引或名称）
            if isinstance(feature, int):
                feature_name = self.feature_columns[feature]
            else:
                feature_name = str(feature)

            contribution = {
                "feature": feature_name,
                "lime_value": float(value),
                "direction": "positive" if value > 0 else "negative",
                "abs_contribution": float(abs(value)),
            }
            contributions.append(contribution)

        # 按绝对值排序
        contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)

        return contributions

    def explain_and_format(
        self,
        features: Union[Dict, pd.DataFrame],
        num_features: int = 10
    ) -> List[Dict]:
        """
        解释并格式化（便捷方法）

        Args:
            features: 特征数据
            num_features: 显示的特征数量

        Returns:
            标准化格式的特征贡献列表
        """
        exp = self.explain_single(features, num_features)
        return self.format_explanation(exp)

    def get_feature_importance(self, importance_scores: Dict[str, float]) -> List[Dict]:
        """
        从重要性得分获取标准格式

        Args:
            importance_scores: 特征重要性字典

        Returns:
            标准化格式列表
        """
        contributions = []
        for feature, score in importance_scores.items():
            contribution = {
                "feature": feature,
                "lime_value": float(score),
                "direction": "positive" if score > 0 else "negative",
                "abs_contribution": float(abs(score)),
            }
            contributions.append(contribution)

        contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
        return contributions


# 全局分析器实例
_analyzer: Optional[LIMEAnalyzer] = None


def get_analyzer(
    model=None,
    feature_columns: List[str] = None,
    training_data: pd.DataFrame = None
) -> LIMEAnalyzer:
    """
    获取LIME分析器实例（带缓存）

    Args:
        model: 模型实例
        feature_columns: 特征列
        training_data: 训练数据

    Returns:
        LIMEAnalyzer实例
    """
    global _analyzer
    if _analyzer is None:
        if model is None or feature_columns is None:
            raise ValueError("首次调用需要提供model和feature_columns")
        _analyzer = LIMEAnalyzer(model, feature_columns, training_data)
    return _analyzer


def explain_lime_prediction(
    model,
    features: Union[Dict, pd.DataFrame],
    feature_columns: List[str],
    num_features: int = 10
) -> List[Dict]:
    """
    便捷函数：解释LIME预测

    Args:
        model: 模型
        features: 特征数据
        feature_columns: 特征列
        num_features: 显示的特征数量

    Returns:
        标准化格式的特征贡献列表
    """
    analyzer = get_analyzer(model, feature_columns)
    return analyzer.explain_and_format(features, num_features)


if __name__ == "__main__":
    # 演示LIME分析
    logging.basicConfig(level=logging.INFO)

    if not LIME_AVAILABLE:
        print("LIME未安装，跳过演示")
    else:
        # 创建模拟模型
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "hour": np.random.randint(0, 24, n),
            "temperature": np.random.normal(25, 10, n),
            "humidity": np.random.uniform(30, 90, n),
            "load_lag_1h": np.random.normal(2500, 500, n),
        })
        y = 2000 + X["hour"] * 10 + X["temperature"] * 5 + np.random.randn(n) * 100

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # 示例特征
        features = {
            "hour": 14,
            "temperature": 35.0,
            "humidity": 60,
            "load_lag_1h": 2700.0,
        }

        # LIME分析
        analyzer = LIMEAnalyzer(model, list(X.columns), X)
        contributions = analyzer.explain_and_format(features)

        print("\nLIME特征贡献分析:")
        print("-" * 60)
        for i, contrib in enumerate(contributions):
            direction = "↑" if contrib["direction"] == "positive" else "↓"
            print(
                f"{i+1:2d}. {contrib['feature']:<20} "
                f"{direction} {contrib['lime_value']:>8.2f} "
                f"(|贡献|: {contrib['abs_contribution']:.2f})"
            )
