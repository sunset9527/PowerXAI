"""
模型评估指标

功能：
- MAE、RMSE、MAPE、R²指标计算
- 生成评估报告
- 多模型对比
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==================== 基础指标计算 ====================

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (MAE)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        MAE值
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (RMSE)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        RMSE值
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对百分比误差 (MAPE)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        MAPE值（百分比）
    """
    # 避免除零
    mask = y_true != 0
    if np.sum(mask) == 0:
        return float(np.nan)

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R²)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        R²值
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return float(np.nan)

    return float(1 - ss_res / ss_tot)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算对称平均绝对百分比误差 (SMAPE)

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        SMAPE值（百分比）
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    mask = denominator != 0
    if np.sum(mask) == 0:
        return float(np.nan)

    return float(np.mean(numerator[mask] / denominator[mask]) * 100)


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24
) -> float:
    """
    计算平均绝对缩放误差 (MASE)

    Args:
        y_true: 真实值
        y_pred: 预测值
        y_train: 训练集真实值（用于计算尺度因子）
        seasonality: 季节性周期（小时数据默认为24）

    Returns:
        MASE值
    """
    mae = np.mean(np.abs(y_true - y_pred))

    # 计算训练集的季节性差异
    n = len(y_train)
    if n > seasonality:
        scale_factor = np.mean(
            np.abs(y_train[seasonality:] - y_train[:-seasonality])
        )
    else:
        scale_factor = np.mean(np.abs(np.diff(y_train)))

    if scale_factor == 0:
        return float(np.nan)

    return float(mae / scale_factor)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值
        y_train: 训练集真实值（用于MASE计算，可选）

    Returns:
        指标字典
    """
    metrics = {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred),
        "SMAPE": calculate_smape(y_true, y_pred),
    }

    # 如果有训练数据，计算MASE
    if y_train is not None:
        metrics["MASE"] = calculate_mase(y_true, y_pred, y_train)

    return metrics


# ==================== 模型评估 ====================

def evaluate_model(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    y_true_col: str = "load",
    y_pred_col: str = "predicted_load",
    y_train_col: Optional[str] = None,
    train_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    评估模型性能

    Args:
        test_df: 测试集DataFrame
        predictions: 预测结果DataFrame
        y_true_col: 真实值列名
        y_pred_col: 预测值列名
        y_train_col: 训练集目标列名（用于MASE）
        train_df: 训练集DataFrame

    Returns:
        评估报告字典
    """
    # 获取真实值和预测值
    y_true = test_df[y_true_col].values
    y_pred = predictions[y_pred_col].values

    # 获取训练集真实值（如果有）
    y_train = None
    if train_df is not None and y_train_col:
        y_train = train_df[y_train_col].values

    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, y_train)

    # 计算置信区间覆盖率
    if "lower_bound" in predictions.columns and "upper_bound" in predictions.columns:
        lower = predictions["lower_bound"].values
        upper = predictions["upper_bound"].values
        coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
        metrics["confidence_coverage_95"] = round(coverage, 2)

    # 识别误差较大的样本
    errors = np.abs(y_true - y_pred)
    threshold_95 = np.percentile(errors, 95)

    high_error_indices = np.where(errors > threshold_95)[0]

    report = {
        "metrics": metrics,
        "n_samples": len(y_true),
        "mean_actual": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
        "std_actual": float(np.std(y_true)),
        "std_predicted": float(np.std(y_pred)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "high_error_samples": len(high_error_indices),
    }

    return report


# ==================== 多模型对比 ====================

def compare_models_on_test(
    models_dict: Dict[str, any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_names: Optional[Dict[str, str]] = None
) -> Dict[str, Dict]:
    """
    在测试集上对比多个模型

    Args:
        models_dict: 模型字典 {'xgboost': model, 'lightgbm': model, 'lstm': predictor}
        X_test: 测试特征
        y_test: 测试目标
        model_names: 模型显示名称映射

    Returns:
        各模型的评估结果字典
    """
    if model_names is None:
        model_names = {}

    comparison = {}

    for name, model in models_dict.items():
        display_name = model_names.get(name, name)

        logger.info(f"评估模型: {display_name}")

        try:
            # 获取预测
            if name == 'lstm':
                predictions = model.predict(X_test)
                y_aligned = y_test[24:]  # LSTM序列长度偏移
            else:
                predictions = model.predict(X_test)
                y_aligned = y_test

            # 计算指标
            metrics = calculate_metrics(y_aligned, predictions)

            # 额外统计
            errors = y_aligned - predictions
            abs_errors = np.abs(errors)

            comparison[display_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'errors': errors,
                'abs_errors': abs_errors,
                'n_samples': len(y_aligned),
                'mean_actual': float(np.mean(y_aligned)),
                'std_actual': float(np.std(y_aligned)),
                'max_error': float(np.max(abs_errors)),
                'min_error': float(np.min(abs_errors)),
            }

            logger.info(
                f"  {display_name} - MAE: {metrics['MAE']:.2f}, "
                f"RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}"
            )

        except Exception as e:
            logger.error(f"模型 '{name}' 评估失败: {e}")
            comparison[display_name] = {'error': str(e)}

    return comparison


def format_comparison_table(comparison: Dict[str, Dict]) -> pd.DataFrame:
    """
    格式化对比结果为表格

    Args:
        comparison: compare_models_on_test返回的对比字典

    Returns:
        对比DataFrame
    """
    rows = []

    for name, result in comparison.items():
        if 'error' in result:
            continue

        row = {
            '模型': name,
            'MAE': result['metrics'].get('MAE', np.nan),
            'RMSE': result['metrics'].get('RMSE', np.nan),
            'MAPE(%)': result['metrics'].get('MAPE', np.nan),
            'R²': result['metrics'].get('R2', np.nan),
            'SMAPE(%)': result['metrics'].get('SMAPE', np.nan),
            '样本数': result.get('n_samples', 0),
        }

        # 如果有MASE
        if 'MASE' in result['metrics']:
            row['MASE'] = result['metrics']['MASE']

        rows.append(row)

    df = pd.DataFrame(rows)

    # 按MAE排序
    if 'MAE' in df.columns:
        df = df.sort_values('MAE').reset_index(drop=True)

    return df


def get_comparison_summary(
    comparison: Dict[str, Dict],
    best_metric: str = 'MAE',
    higher_is_better: bool = False
) -> str:
    """
    生成对比摘要

    Args:
        comparison: 对比结果字典
        best_metric: 用于评选最佳的指标
        higher_is_better: 该指标是否越高越好

    Returns:
        格式化的摘要字符串
    """
    table = format_comparison_table(comparison)

    if len(table) == 0:
        return "没有可用的对比结果"

    # 找出最佳模型
    if higher_is_better:
        best_idx = table[best_metric].idxmax()
    else:
        best_idx = table[best_metric].idxmin()

    best_model = table.loc[best_idx, '模型']

    lines = [
        "=" * 80,
        "多模型对比报告",
        "=" * 80,
        f"\n评估指标: {best_metric} {'(越高越好)' if higher_is_better else '(越低越好)'}",
        f"最佳模型: {best_model}",
        "\n" + "-" * 80,
        table.to_string(index=False),
        "-" * 80,
    ]

    # 添加详细分析
    lines.append("\n【分析】")

    # MAE分析
    maes = [(name, res['metrics']['MAE']) for name, res in comparison.items() if 'metrics' in res]
    maes.sort(key=lambda x: x[1])
    lines.append(f"\nMAE排名:")
    for i, (name, mae) in enumerate(maes, 1):
        lines.append(f"  {i}. {name}: {mae:.2f}")

    # R²分析
    r2s = [(name, res['metrics']['R2']) for name, res in comparison.items() if 'metrics' in res]
    r2s.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"\nR²排名:")
    for i, (name, r2) in enumerate(r2s, 1):
        lines.append(f"  {i}. {name}: {r2:.4f}")

    lines.append("=" * 80)

    return "\n".join(lines)


# ==================== 报告输出 ====================

def print_evaluation_report(report: Dict) -> str:
    """
    格式化输出评估报告

    Args:
        report: 评估报告字典

    Returns:
        格式化的报告字符串
    """
    metrics = report["metrics"]

    lines = [
        "=" * 50,
        "模型评估报告",
        "=" * 50,
        f"\n样本数量: {report['n_samples']}",
        f"\n核心指标:",
        f"  MAE:  {metrics['MAE']:.2f} MW",
        f"  RMSE: {metrics['RMSE']:.2f} MW",
        f"  MAPE: {metrics['MAPE']:.2f}%",
        f"  R²:   {metrics['R2']:.4f}",
        f"  SMAPE: {metrics['SMAPE']:.2f}%",
    ]

    if "MASE" in metrics:
        lines.append(f"  MASE: {metrics['MASE']:.4f}")

    if "confidence_coverage_95" in metrics:
        lines.append(f"\n置信区间覆盖率(95%): {metrics['confidence_coverage_95']:.2f}%")

    lines.extend([
        f"\n负荷统计:",
        f"  实际均值: {report['mean_actual']:.2f} MW",
        f"  预测均值: {report['mean_predicted']:.2f} MW",
        f"  实际标准差: {report['std_actual']:.2f} MW",
        f"  预测标准差: {report['std_predicted']:.2f} MW",
        f"\n误差统计:",
        f"  最大误差: {report['max_error']:.2f} MW",
        f"  最小误差: {report['min_error']:.2f} MW",
        f"  高误差样本数: {report['high_error_samples']}",
        "=" * 50,
    ])

    return "\n".join(lines)


# 导入Optional
from typing import Optional

if __name__ == "__main__":
    # 演示评估
    logging.basicConfig(level=logging.INFO)

    # 模拟数据
    np.random.seed(42)
    y_true = np.random.uniform(1500, 3000, 1000)
    y_pred = y_true + np.random.normal(0, 50, 1000)

    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)

    print("\n评估指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 多模型对比演示
    print("\n" + "=" * 50)
    print("多模型对比演示")
    print("=" * 50)

    # 模拟多个模型的预测
    models_predictions = {
        'XGBoost': y_true + np.random.normal(0, 50, 1000),
        'LightGBM': y_true + np.random.normal(0, 45, 1000),
        'LSTM': y_true + np.random.normal(0, 55, 1000),
    }

    # 对比
    comparison = {}
    for name, preds in models_predictions.items():
        comparison[name] = {
            'metrics': calculate_metrics(y_true, preds),
            'predictions': preds
        }

    # 输出表格
    table = format_comparison_table(comparison)
    print("\n对比表格:")
    print(table.to_string(index=False))

    # 输出摘要
    print("\n" + get_comparison_summary(comparison))
