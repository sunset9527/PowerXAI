"""
模型训练器

功能：
- XGBoost回归模型训练
- LightGBM回归模型训练
- 置信区间估计模型训练
- 超参数配置
- 多模型统一训练入口
- 模型保存与加载
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from config import settings

logger = logging.getLogger(__name__)


# ==================== XGBoost训练 ====================

def get_default_params() -> Dict:
    """
    获取默认的XGBoost超参数

    Returns:
        超参数字典
    """
    return {
        "n_estimators": settings.XGB_N_ESTIMATORS,
        "max_depth": settings.XGB_MAX_DEPTH,
        "learning_rate": settings.XGB_LEARNING_RATE,
        "subsample": settings.XGB_SUBSAMPLE,
        "colsample_bytree": settings.XGB_COLSAMPLE_BYTREE,
        "min_child_weight": settings.XGB_MIN_CHILD_WEIGHT,
        "random_state": settings.XGB_RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }


def train_xgboost(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    params: Optional[Dict] = None
) -> xgb.XGBRegressor:
    """
    训练XGBoost模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）
        params: 超参数（可选）

    Returns:
        训练好的模型
    """
    params = params or get_default_params()

    logger.info("开始训练XGBoost模型...")
    logger.info(f"训练数据: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

    # 创建模型
    model = xgb.XGBRegressor(**params)

    # 如果有验证集，使用早停
    if X_val is not None and y_val is not None:
        logger.info("使用验证集进行早停训练...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
    else:
        logger.info("训练中...")
        model.fit(X_train, y_train)

    logger.info("XGBoost模型训练完成")

    return model


# ==================== LightGBM训练 ====================

def get_lgbm_default_params() -> Dict:
    """
    获取默认的LightGBM超参数

    Returns:
        超参数字典
    """
    return {
        "n_estimators": settings.LGBM_N_ESTIMATORS,
        "max_depth": settings.LGBM_MAX_DEPTH,
        "learning_rate": settings.LGBM_LEARNING_RATE,
        "num_leaves": settings.LGBM_NUM_LEAVES,
        "subsample": settings.LGBM_SUBSAMPLE,
        "colsample_bytree": settings.LGBM_COLSAMPLE_BYTREE,
        "random_state": settings.XGB_RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }


def train_lightgbm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    params: Optional[Dict] = None
) -> "lightgbm.LGBMRegressor":
    """
    训练LightGBM模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）
        params: 超参数（可选）

    Returns:
        训练好的模型
    """
    import lightgbm as lgb

    params = params or get_lgbm_default_params()

    logger.info("开始训练LightGBM模型...")
    logger.info(f"训练数据: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)

    valid_sets = [train_data]
    valid_names = ['train']

    if X_val is not None and y_val is not None:
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(valid_data)
        valid_names.append('valid')

    # 训练参数
    train_params = params.copy()
    train_params.pop('n_estimators', None)

    # 训练模型
    callbacks = [lgb.early_stopping(stopping_rounds=50)]

    model = lgb.train(
        train_params,
        train_data,
        num_boost_round=params.get('n_estimators', settings.LGBM_N_ESTIMATORS),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    logger.info("LightGBM模型训练完成")

    # 转换为sklearn风格以保持接口一致
    sklearn_model = lgb.LGBMRegressor(**params)
    sklearn_model.fit(X_train, y_train)

    return sklearn_model


# ==================== 置信区间模型训练 ====================

def train_confidence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    residuals: np.ndarray
) -> RandomForestRegressor:
    """
    训练用于估计置信区间的随机森林模型

    通过学习预测误差来估计不确定性

    Args:
        X_train: 训练特征
        y_train: 训练标签
        residuals: 残差（真实值 - 预测值）

    Returns:
        训练好的随机森林模型
    """
    logger.info("训练置信区间估计模型...")

    # 使用随机森林学习残差的绝对值
    rf = RandomForestRegressor(
        n_estimators=settings.RF_N_ESTIMATORS,
        max_depth=8,
        random_state=settings.XGB_RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X_train, np.abs(residuals))

    logger.info("置信区间估计模型训练完成")

    return rf


# ==================== 统一训练入口 ====================

def train_all_models(
    train_df,
    feature_columns: List[str],
    target_column: str = "load",
    models_to_train: Optional[List[str]] = None,
    X_val=None,
    y_val=None
) -> Dict:
    """
    训练多个模型

    Args:
        train_df: 训练数据DataFrame
        feature_columns: 特征列名列表
        target_column: 目标列名
        models_to_train: 要训练的模型列表 ['xgboost', 'lightgbm']
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）

    Returns:
        包含所有模型和训练信息的字典
    """
    if models_to_train is None:
        models_to_train = ['xgboost']

    logger.info("=" * 50)
    logger.info(f"开始训练多个模型: {models_to_train}")
    logger.info("=" * 50)

    # 准备数据
    X = train_df[feature_columns].values
    y = train_df[target_column].values

    models = {}
    training_info = {
        'feature_columns': feature_columns,
        'target_column': target_column,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'models_trained': []
    }

    # 训练XGBoost
    if 'xgboost' in models_to_train:
        logger.info("\n>>> 训练 XGBoost 模型")
        xgb_model = train_xgboost(X, y, X_val, y_val)
        models['xgboost'] = xgb_model
        training_info['models_trained'].append('xgboost')

        # 计算训练集指标
        xgb_pred = xgb_model.predict(X)
        xgb_residuals = y - xgb_pred
        training_info['xgboost_train_metrics'] = calculate_model_metrics(y, xgb_pred)

    # 训练LightGBM
    if 'lightgbm' in models_to_train:
        logger.info("\n>>> 训练 LightGBM 模型")
        lgbm_model = train_lightgbm(X, y, X_val, y_val)
        models['lightgbm'] = lgbm_model
        training_info['models_trained'].append('lightgbm')

        # 计算训练集指标
        lgbm_pred = lgbm_model.predict(X)
        training_info['lightgbm_train_metrics'] = calculate_model_metrics(y, lgbm_pred)

    # 训练置信区间模型（使用XGBoost残差）
    if 'xgboost' in models:
        confidence_model = train_confidence_model(X, y, xgb_residuals)
        models['confidence_model'] = confidence_model
        training_info['confidence_model'] = True

    logger.info("=" * 50)
    logger.info("所有模型训练完成")
    logger.info(f"已训练模型: {training_info['models_trained']}")
    logger.info("=" * 50)

    return {
        'models': models,
        'training_info': training_info
    }


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算模型评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        指标字典
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


# ==================== 原有train_model（保持兼容） ====================

def train_model(
    train_df,
    feature_columns: list,
    target_column: str = "load",
    params: Optional[Dict] = None
) -> Tuple[xgb.XGBRegressor, RandomForestRegressor, Dict]:
    """
    完整的模型训练流程

    Args:
        train_df: 训练数据DataFrame
        feature_columns: 特征列名列表
        target_column: 目标列名
        params: 超参数（可选）

    Returns:
        (XGBoost模型, 置信区间模型, 训练信息)
    """
    from .evaluator import calculate_metrics

    logger.info("=" * 50)
    logger.info("开始完整的模型训练流程")
    logger.info("=" * 50)

    # 准备数据
    X = train_df[feature_columns].values
    y = train_df[target_column].values

    # 训练XGBoost模型
    model = train_xgboost(X, y, params=params)

    # 在训练集上预测以计算残差
    train_predictions = model.predict(X)
    residuals = y - train_predictions

    # 训练置信区间模型
    confidence_model = train_confidence_model(X, y, residuals)

    # 计算训练集指标
    train_metrics = calculate_metrics(y, train_predictions)

    # 训练信息
    training_info = {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "train_metrics": train_metrics,
    }

    logger.info("=" * 50)
    logger.info("模型训练完成")
    logger.info(f"训练集 MAE: {train_metrics['MAE']:.2f} MW")
    logger.info(f"训练集 RMSE: {train_metrics['RMSE']:.2f} MW")
    logger.info(f"训练集 R²: {train_metrics['R2']:.4f}")
    logger.info("=" * 50)

    return model, confidence_model, training_info


# ==================== 模型保存与加载 ====================

def save_model(
    model: Union[xgb.XGBRegressor, Dict],
    confidence_model: Optional[RandomForestRegressor] = None,
    training_info: Optional[Dict] = None,
    filepath: Optional[Path] = None,
    multi_model: bool = False
) -> Path:
    """
    保存模型到文件

    Args:
        model: XGBoost模型或模型字典（多模型时）
        confidence_model: 置信区间模型
        training_info: 训练信息
        filepath: 文件路径
        multi_model: 是否为多模型保存

    Returns:
        保存的文件路径
    """
    if multi_model and isinstance(model, dict):
        # 多模型保存
        filepath = filepath or settings.MODEL_DIR / "models_multi.pkl"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'models': model,
            'confidence_model': confidence_model,
            'training_info': training_info,
            'multi_model': True
        }
    else:
        # 单模型保存（向后兼容）
        filepath = filepath or settings.MODEL_DIR / "xgboost_model.pkl"

        if confidence_model is not None:
            model_data = {
                "model": model,
                "confidence_model": confidence_model,
                "training_info": training_info,
            }
        else:
            model_data = model

    joblib.dump(model_data, filepath)
    logger.info(f"模型已保存至: {filepath}")

    return filepath


def load_model(filepath: Optional[Path] = None) -> Dict:
    """
    从文件加载模型

    Args:
        filepath: 文件路径

    Returns:
        包含模型和训练信息的字典
    """
    filepath = filepath or settings.MODEL_DIR / "xgboost_model.pkl"

    # 尝试多个可能的路径
    possible_paths = [
        filepath,
        settings.MODEL_DIR / "xgboost_model.pkl",
        settings.MODEL_DIR / "models_multi.pkl"
    ]

    loaded_path = None
    for path in possible_paths:
        if path.exists():
            filepath = path
            loaded_path = path
            break

    if loaded_path is None:
        raise FileNotFoundError(f"模型文件不存在，搜索了: {possible_paths}")

    model_data = joblib.load(filepath)

    # 兼容处理：如果是单模型，包装成字典格式
    if not isinstance(model_data, dict):
        model_data = {"model": model_data, "multi_model": False}
    elif "multi_model" not in model_data:
        model_data["multi_model"] = False

    logger.info(f"模型已从 {filepath} 加载")

    return model_data


def get_model_info(filepath: Optional[Path] = None) -> Dict:
    """
    获取模型信息

    Args:
        filepath: 模型文件路径

    Returns:
        模型信息字典
    """
    model_data = load_model(filepath)
    training_info = model_data.get("training_info", {})

    info = {
        "n_features": training_info.get("n_features", 0),
        "n_samples": training_info.get("n_samples", 0),
        "feature_columns": training_info.get("feature_columns", []),
        "train_metrics": training_info.get("train_metrics", {}),
        "model_type": "multi" if model_data.get("multi_model") else "XGBoost",
    }

    # 如果是单XGBoost模型，添加特征重要性
    if not model_data.get("multi_model") and "model" in model_data:
        model = model_data["model"]
        if hasattr(model, "feature_importances_"):
            try:
                feature_columns = training_info.get("feature_columns", [])
                feature_importance = dict(zip(
                    feature_columns,
                    model.feature_importances_
                ))
                feature_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
                info["feature_importance"] = feature_importance
            except Exception:
                pass

    return info


def ensure_model_exists(
    train_df,
    feature_columns: list,
    target_column: str = "load",
    models_to_train: Optional[List[str]] = None,
    multi_model: bool = False
) -> Dict:
    """
    确保模型存在，如不存在则训练并保存

    Args:
        train_df: 训练数据
        feature_columns: 特征列
        target_column: 目标列
        models_to_train: 要训练的模型列表
        multi_model: 是否使用多模型

    Returns:
        模型数据字典
    """
    if multi_model:
        filepath = settings.MODEL_DIR / "models_multi.pkl"
    else:
        filepath = settings.MODEL_DIR / "xgboost_model.pkl"

    if filepath.exists():
        logger.info("检测到已有模型文件，直接加载")
        return load_model(filepath)
    else:
        logger.info("未检测到模型文件，开始训练新模型")

        if multi_model and models_to_train:
            result = train_all_models(train_df, feature_columns, target_column, models_to_train)
            save_model(
                result['models'],
                result['models'].get('confidence_model'),
                result['training_info'],
                multi_model=True
            )
            return {
                'models': result['models'],
                'training_info': result['training_info'],
                'multi_model': True
            }
        else:
            model, confidence_model, training_info = train_model(
                train_df, feature_columns, target_column
            )
            save_model(model, confidence_model, training_info)
            return {
                "model": model,
                "confidence_model": confidence_model,
                "training_info": training_info,
                "multi_model": False
            }


if __name__ == "__main__":
    # 演示模型训练
    logging.basicConfig(level=logging.INFO)

    from src.data.generator import ensure_data_exists
    from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns

    # 加载数据
    raw_df = ensure_data_exists()

    # 预处理
    processed_df, _ = preprocess_data(raw_df)

    # 划分数据集
    train_df, test_df = split_train_test(processed_df)

    # 获取特征列
    feature_cols = get_feature_columns(processed_df)

    # 训练多模型
    result = train_all_models(
        train_df,
        feature_cols,
        models_to_train=['xgboost', 'lightgbm']
    )

    # 保存多模型
    save_model(
        result['models'],
        result['models'].get('confidence_model'),
        result['training_info'],
        multi_model=True
    )

    # 获取模型信息
    info = get_model_info(settings.MODEL_DIR / "models_multi.pkl")
    print("\n模型信息:")
    print(f"  模型类型: {info['model_type']}")
    print(f"  特征数量: {info['n_features']}")
    print(f"  样本数量: {info['n_samples']}")
    print(f"  已训练模型: {result['training_info']['models_trained']}")
