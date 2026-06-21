"""
AI预测分析师 - FastAPI REST API

功能：
- POST /predict：数值预测（输入特征→预测值+置信区间）
- POST /explain：自然语言解释（预测值+SHAP+LLM分析）
- POST /compare：时段对比分析
- GET /insights：自动洞察
- GET /model-info：模型信息+性能指标
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from src.data.generator import ensure_data_exists
from src.data.preprocessor import (
    preprocess_data,
    split_train_test,
    get_feature_columns,
    prepare_prediction_features,
)
from src.model.trainer import ensure_model_exists, get_model_info
from src.model.predictor import predict, predict_batch
from src.model.evaluator import evaluate_model
from src.explainer.shap_analyzer import explain_prediction, explain_global
from src.explainer.report import generate_report
from src.analyst.llm_explainer import generate_explanation_sync
from src.analyst.insight import InsightEngine
from src.analyst.comparator import Comparator, compare_periods

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="AI预测分析师 API",
    description="ML预测 + SHAP可解释 + LLM自然语言分析系统",
    version="1.0.0",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic模型 ====================

class PredictionFeatures(BaseModel):
    """预测特征输入"""

    datetime: str = Field(..., description="日期时间，格式YYYY-MM-DD")
    hour: int = Field(..., ge=0, le=23, description="小时")
    day_of_week: int = Field(..., ge=0, le=6, description="星期几(0=周一)")
    temperature: float = Field(..., description="温度(°C)")
    humidity: float = Field(..., ge=0, le=100, description="湿度(%)")
    is_holiday: bool = Field(default=False, description="是否节假日")
    season: str = Field(..., description="季节(spring/summer/autumn/winter)")
    load_lag_1h: Optional[float] = Field(None, description="前1小时负荷(MW)")
    load_lag_24h: Optional[float] = Field(None, description="前24小时负荷(MW)")
    load_lag_168h: Optional[float] = Field(None, description="前168小时负荷(MW)")


class PredictionResponse(BaseModel):
    """预测响应"""

    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float
    unit: str = "MW"


class SHAPContribution(BaseModel):
    """SHAP贡献"""

    feature: str
    shap_value: float
    feature_value: float
    direction: str


class ExplainResponse(BaseModel):
    """解释响应"""

    prediction: float
    unit: str
    contributions: List[SHAPContribution]
    explanation: str
    report: Dict


class ComparisonRequest(BaseModel):
    """对比请求"""

    features1: PredictionFeatures
    features2: PredictionFeatures


class ComparisonResponse(BaseModel):
    """对比响应"""

    period1_prediction: float
    period2_prediction: float
    prediction_diff: float
    prediction_diff_pct: float
    feature_changes: List[Dict]
    key_drivers: List[str]
    key_reducers: List[str]
    explanation: str


class ModelInfoResponse(BaseModel):
    """模型信息响应"""

    n_features: int
    n_samples: int
    feature_columns: List[str]
    feature_importance: Dict[str, float]
    train_metrics: Dict[str, float]
    model_type: str


# ==================== 全局数据缓存 ====================

_data_cache: Optional[Dict] = None


def get_data() -> Dict:
    """获取缓存的数据"""
    global _data_cache

    if _data_cache is None:
        logger.info("初始化数据...")
        raw_df = ensure_data_exists()
        processed_df, _ = preprocess_data(raw_df)
        train_df, test_df = split_train_test(processed_df)
        feature_cols = get_feature_columns(processed_df)

        ensure_model_exists(train_df, feature_cols)

        test_predictions = predict_batch(test_df)
        eval_report = evaluate_model(
            test_df, test_predictions, "load", "predicted_load", "load", train_df
        )

        _data_cache = {
            "raw_df": raw_df,
            "processed_df": processed_df,
            "train_df": train_df,
            "test_df": test_df,
            "test_predictions": test_predictions,
            "feature_columns": feature_cols,
            "eval_report": eval_report,
        }

    return _data_cache


# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI预测分析师 API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_load(features: PredictionFeatures):
    """
    数值预测

    输入特征，返回预测值及置信区间
    """
    try:
        # 准备特征
        feature_dict = prepare_prediction_features(
            datetime_str=features.datetime,
            hour=features.hour,
            day_of_week=features.day_of_week,
            temperature=features.temperature,
            humidity=features.humidity,
            is_holiday=features.is_holiday,
            season=features.season,
            load_lag_1h=features.load_lag_1h,
            load_lag_24h=features.load_lag_24h,
            load_lag_168h=features.load_lag_168h,
        )

        # 预测
        result = predict(feature_dict)

        return PredictionResponse(
            prediction=result["prediction"],
            lower_bound=result["lower_bound"],
            upper_bound=result["upper_bound"],
            confidence_interval=result["confidence_interval"],
        )

    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
async def explain_load(features: PredictionFeatures):
    """
    自然语言解释

    预测 + SHAP分析 + LLM解释
    """
    try:
        # 准备特征
        feature_dict = prepare_prediction_features(
            datetime_str=features.datetime,
            hour=features.hour,
            day_of_week=features.day_of_week,
            temperature=features.temperature,
            humidity=features.humidity,
            is_holiday=features.is_holiday,
            season=features.season,
            load_lag_1h=features.load_lag_1h,
            load_lag_24h=features.load_lag_24h,
            load_lag_168h=features.load_lag_168h,
        )

        # 预测
        pred_result = predict(feature_dict)

        # SHAP分析
        contributions = explain_prediction(pred_result["prediction"], feature_dict)

        # 生成报告
        report = generate_report(
            pred_result["prediction"],
            feature_dict,
            contributions,
            detail_level="standard"
        )

        # LLM解释
        explanation = generate_explanation_sync(
            prediction=pred_result["prediction"],
            features=feature_dict,
            contributions=contributions,
            detail_level="standard"
        )

        return ExplainResponse(
            prediction=pred_result["prediction"],
            unit="MW",
            contributions=[
                SHAPContribution(
                    feature=c["feature"],
                    shap_value=c["shap_value"],
                    feature_value=c["feature_value"],
                    direction=c["direction"],
                )
                for c in contributions[:10]
            ],
            explanation=explanation,
            report=report,
        )

    except Exception as e:
        logger.error(f"解释生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparisonResponse)
async def compare_load(request: ComparisonRequest):
    """
    时段对比分析

    对比两个时段的预测和SHAP分析
    """
    try:
        # 准备时段1特征
        features1 = prepare_prediction_features(
            datetime_str=request.features1.datetime,
            hour=request.features1.hour,
            day_of_week=request.features1.day_of_week,
            temperature=request.features1.temperature,
            humidity=request.features1.humidity,
            is_holiday=request.features1.is_holiday,
            season=request.features1.season,
            load_lag_1h=request.features1.load_lag_1h,
            load_lag_24h=request.features1.load_lag_24h,
            load_lag_168h=request.features1.load_lag_168h,
        )

        # 准备时段2特征
        features2 = prepare_prediction_features(
            datetime_str=request.features2.datetime,
            hour=request.features2.hour,
            day_of_week=request.features2.day_of_week,
            temperature=request.features2.temperature,
            humidity=request.features2.humidity,
            is_holiday=request.features2.is_holiday,
            season=request.features2.season,
            load_lag_1h=request.features2.load_lag_1h,
            load_lag_24h=request.features2.load_lag_24h,
            load_lag_168h=request.features2.load_lag_168h,
        )

        # 预测
        pred1 = predict(features1)
        pred2 = predict(features2)

        # SHAP分析
        contrib1 = explain_prediction(pred1["prediction"], features1)
        contrib2 = explain_prediction(pred2["prediction"], features2)

        # 对比分析
        result = compare_periods(
            features1, features2,
            pred1["prediction"], pred2["prediction"],
            contrib1, contrib2
        )

        return ComparisonResponse(
            period1_prediction=result.period1_info["prediction"],
            period2_prediction=result.period2_info["prediction"],
            prediction_diff=result.prediction_diff,
            prediction_diff_pct=result.prediction_diff_pct,
            feature_changes=[
                {
                    "feature": c.feature,
                    "value1": c.value1,
                    "value2": c.value2,
                    "shap1": c.shap1,
                    "shap2": c.shap2,
                    "change": c.shap_change,
                    "type": c.impact_type,
                }
                for c in result.feature_changes[:10]
            ],
            key_drivers=result.key_drivers,
            key_reducers=result.key_reducers,
            explanation=result.explanation,
        )

    except Exception as e:
        logger.error(f"对比分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights")
async def get_insights():
    """
    自动洞察

    获取趋势分析、异常检测、关联分析结果
    """
    try:
        data = get_data()
        processed_df = data["processed_df"]
        test_predictions = data["test_predictions"]

        engine = InsightEngine()

        # 趋势检测（使用处理后的数据）
        trends = engine.detect_trends(processed_df.tail(168))  # 最近一周

        # 异常检测（使用测试集预测）
        anomalies = engine.detect_anomalies(test_predictions)

        # 关联分析
        correlations = engine.find_correlations(
            processed_df.sample(min(1000, len(processed_df))),
            "temperature",
            "load",
            max_lag=12
        )

        # 生成摘要
        summary = engine.generate_summary(trends, anomalies, correlations)

        return {
            "trends": [
                {
                    "type": t.trend_type,
                    "duration": t.duration,
                    "change_rate": t.change_rate,
                    "confidence": t.confidence,
                }
                for t in trends[:5]
            ],
            "anomalies": [
                {
                    "timestamp": a.timestamp,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "error_pct": a.error_percentage,
                    "possible_causes": a.possible_causes,
                }
                for a in anomalies[:10]
            ],
            "correlations": [
                {
                    "factor1": c.factor1,
                    "factor2": c.factor2,
                    "correlation": c.correlation,
                    "lag": c.lag,
                    "description": c.description,
                }
                for c in correlations
            ],
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"洞察提取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info_endpoint():
    """
    模型信息

    获取模型信息和性能指标
    """
    try:
        info = get_model_info()

        return ModelInfoResponse(
            n_features=info["n_features"],
            n_samples=info["n_samples"],
            feature_columns=info["feature_columns"],
            feature_importance=info["feature_importance"],
            train_metrics=info["train_metrics"],
            model_type=info["model_type"],
        )

    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance")
async def get_feature_importance():
    """
    全局特征重要性

    获取SHAP分析的特征重要性排序
    """
    try:
        data = get_data()
        processed_df = data["processed_df"]

        importance = explain_global(processed_df.sample(min(500, len(processed_df))))

        return {
            "importance": importance,
            "n_samples": len(processed_df),
        }

    except Exception as e:
        logger.error(f"获取特征重要性失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动应用 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
