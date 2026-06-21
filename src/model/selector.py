"""
模型选择器

根据交叉验证结果自动选择最佳模型。

功能：
- 多模型对比表生成
- 基于指标的模型排名
- 支持MAE/RMSE/R²等指标选择
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    模型选择器

    基于交叉验证结果选择最佳模型。
    """

    def __init__(
        self,
        metric: str = 'MAE',
        higher_is_better: bool = False
    ):
        """
        初始化模型选择器

        Args:
            metric: 用于选择的指标名
            higher_is_better: 指标是否越高越好（如R²）
        """
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.models: Dict[str, Dict] = {}
        self.comparison_df: Optional[pd.DataFrame] = None

    def add_model(
        self,
        name: str,
        model: any,
        cv_results: Optional[List[Dict]] = None,
        avg_metrics: Optional[Dict] = None,
        config: Optional[Dict] = None
    ):
        """
        添加模型到候选池

        Args:
            name: 模型名称
            model: 模型对象
            cv_results: 交叉验证结果列表
            avg_metrics: 平均指标字典
            config: 模型配置信息
        """
        self.models[name] = {
            'model': model,
            'cv_results': cv_results,
            'avg_metrics': avg_metrics,
            'config': config or {}
        }

        logger.info(f"已将模型 '{name}' 添加到候选池")

    def add_cv_result(
        self,
        name: str,
        cv_results: List[Dict],
        avg_metrics: Optional[Dict] = None
    ):
        """
        添加交叉验证结果

        Args:
            name: 模型名称
            cv_results: 交叉验证结果列表
            avg_metrics: 平均指标字典
        """
        if name in self.models:
            self.models[name]['cv_results'] = cv_results
            self.models[name]['avg_metrics'] = avg_metrics
        else:
            self.add_model(name, None, cv_results, avg_metrics)

    def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_configs: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        在测试集上对比所有模型

        Args:
            X: 测试特征
            y: 测试目标
            model_configs: 模型配置 {'xgboost': {...}, 'lightgbm': {...}}

        Returns:
            对比DataFrame
        """
        from src.model.evaluator import calculate_metrics

        if model_configs is None:
            model_configs = {}

        comparison_data = []

        for name, model_info in self.models.items():
            model = model_info['model']

            if model is None:
                logger.warning(f"模型 '{name}' 未加载，跳过")
                continue

            try:
                # 获取预测
                if name == 'lstm':
                    predictions = model.predict(X)
                    y_aligned = y[24:]  # LSTM序列长度偏移
                else:
                    predictions = model.predict(X)
                    y_aligned = y

                # 计算指标
                metrics = calculate_metrics(y_aligned, predictions)

                row = {
                    'model': name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2'],
                    'SMAPE': metrics['SMAPE']
                }

                # 添加CV结果中的指标
                if model_info.get('avg_metrics'):
                    for key, value in model_info['avg_metrics'].items():
                        row[f'cv_{key}'] = value

                comparison_data.append(row)

            except Exception as e:
                logger.error(f"模型 '{name}' 预测失败: {e}")

        self.comparison_df = pd.DataFrame(comparison_data)

        return self.comparison_df

    def select_best(self, comparison_df: Optional[pd.DataFrame] = None) -> Tuple[str, Dict]:
        """
        选择最佳模型

        Args:
            comparison_df: 对比DataFrame（可选，使用当前缓存）

        Returns:
            (最佳模型名称, 最佳模型详情)
        """
        df = comparison_df or self.comparison_df

        if df is None or len(df) == 0:
            raise ValueError("没有可用的模型对比结果")

        # 确定排序方式
        if self.higher_is_better:
            best_idx = df[self.metric].idxmax()
        else:
            best_idx = df[self.metric].idxmin()

        best_row = df.loc[best_idx]
        best_model_name = best_row['model']

        # 获取完整信息
        best_info = self.models.get(best_model_name, {})
        best_info['test_metrics'] = best_row.to_dict()

        logger.info(
            f"最佳模型: {best_model_name} "
            f"({self.metric}={best_row[self.metric]:.4f})"
        )

        return best_model_name, best_info

    def get_ranking(self, comparison_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        获取模型排名

        Args:
            comparison_df: 对比DataFrame

        Returns:
            排名后的DataFrame
        """
        df = comparison_df or self.comparison_df

        if df is None:
            raise ValueError("没有可用的模型对比结果")

        # 按选择的指标排序
        ascending = not self.higher_is_better
        ranked = df.sort_values(self.metric, ascending=ascending).reset_index(drop=True)
        ranked['rank'] = range(1, len(ranked) + 1)

        return ranked

    def get_summary(self, comparison_df: Optional[pd.DataFrame] = None) -> str:
        """
        获取选择结果摘要

        Args:
            comparison_df: 对比DataFrame

        Returns:
            格式化的摘要字符串
        """
        df = comparison_df or self.comparison_df

        if df is None or len(df) == 0:
            return "没有可用的模型对比结果"

        ranked = self.get_ranking(df)
        best_name, best_info = self.select_best(df)

        lines = [
            "=" * 70,
            "模型选择报告",
            "=" * 70,
            f"\n选择指标: {self.metric} "
            f"({'越高越好' if self.higher_is_better else '越低越好'})",
            "\n【模型排名】",
        ]

        # 排名表
        lines.append(f"\n{'排名':<6} {'模型':<15} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
        lines.append("-" * 70)

        for _, row in ranked.iterrows():
            lines.append(
                f"{int(row['rank']):<6} {row['model']:<15} "
                f"{row['MAE']:<12.2f} {row['RMSE']:<12.2f} {row['R2']:<10.4f}"
            )

        # 最佳模型详情
        best_metrics = best_info.get('test_metrics', {})
        lines.extend([
            "-" * 70,
            f"\n【最佳模型】: {best_name}",
            f"\n  测试集表现:",
            f"    MAE:  {best_metrics.get('MAE', 'N/A'):.2f}",
            f"    RMSE: {best_metrics.get('RMSE', 'N/A'):.2f}",
            f"    R²:   {best_metrics.get('R2', 'N/A'):.4f}",
        ])

        # CV表现
        if best_info.get('avg_metrics'):
            avg = best_info['avg_metrics']
            lines.extend([
                f"\n  交叉验证表现:",
                f"    CV MAE:  {avg.get('avg_MAE', 'N/A'):.2f} ± {avg.get('std_MAE', 'N/A'):.2f}",
                f"    CV RMSE: {avg.get('avg_RMSE', 'N/A'):.2f} ± {avg.get('std_RMSE', 'N/A'):.2f}",
                f"    CV R²:   {avg.get('avg_R2', 'N/A'):.4f}",
            ])

        lines.append("=" * 70)

        return "\n".join(lines)

    def save_comparison(
        self,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        保存对比结果

        Args:
            filepath: 保存路径

        Returns:
            保存的文件路径
        """
        if self.comparison_df is None:
            raise ValueError("没有可保存的对比结果")

        filepath = filepath or settings.MODEL_DIR / "model_comparison.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.comparison_df.to_csv(filepath, index=False)
        logger.info(f"模型对比结果已保存至: {filepath}")

        return filepath


def select_best_model(
    models_dict: Dict[str, any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = 'MAE',
    higher_is_better: bool = False
) -> Tuple[str, pd.DataFrame]:
    """
    选择最佳模型的便捷函数

    Args:
        models_dict: 模型字典
        X_test: 测试特征
        y_test: 测试目标
        metric: 选择指标
        higher_is_better: 越高越好

    Returns:
        (最佳模型名称, 对比DataFrame)
    """
    selector = ModelSelector(metric=metric, higher_is_better=higher_is_better)

    # 添加模型
    for name, model in models_dict.items():
        selector.add_model(name, model)

    # 对比
    selector.compare_models(X_test, y_test)

    # 选择
    best_name, _ = selector.select_best()

    return best_name, selector.comparison_df


if __name__ == "__main__":
    # 演示模型选择
    logging.basicConfig(level=logging.INFO)

    from src.data.generator import ensure_data_exists
    from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns
    from src.model.trainer import train_xgboost, train_lightgbm

    # 加载数据
    raw_df = ensure_data_exists()
    processed_df, _ = preprocess_data(raw_df)

    # 划分数据
    train_df, test_df = split_train_test(processed_df)
    feature_cols = get_feature_columns(processed_df)

    X_train = train_df[feature_cols].values
    y_train = train_df['load'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['load'].values

    # 训练模型
    logger.info("训练XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    logger.info("训练LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train)

    # 模型选择
    models = {
        'xgboost': xgb_model,
        'lightgbm': lgbm_model
    }

    best_name, comparison_df = select_best_model(
        models, X_test, y_test, metric='MAE'
    )

    # 输出结果
    selector = ModelSelector(metric='MAE')
    for name, model in models.items():
        selector.add_model(name, model)
    selector.compare_models(X_test, y_test)

    print(selector.get_summary())
