"""
AI预测分析师 - 配置文件
使用pydantic-settings进行配置管理
"""

import os
from pathlib import Path
from typing import Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ==================== 路径配置 ====================
    # 项目根目录
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()

    # 数据目录
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # 模型目录
    MODEL_DIR: Path = PROJECT_ROOT / "models"

    # ==================== 数据生成配置 ====================
    # 数据生成的时间范围
    DATA_START_DATE: str = "2022-01-01"
    DATA_END_DATE: str = "2023-12-31"

    # 噪声水平（相对于基础负荷的比例）
    NOISE_LEVEL: float = 0.05

    # ==================== 真实数据配置 ====================
    USE_REAL_DATA: bool = False  # 是否使用真实数据
    REAL_DATA_PATH: Optional[str] = None  # 真实数据文件路径

    # ==================== 模型训练配置 ====================
    # 训练集比例
    TRAIN_RATIO: float = 0.8

    # XGBoost超参数
    XGB_N_ESTIMATORS: int = 200
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.1
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_MIN_CHILD_WEIGHT: int = 3
    XGB_RANDOM_STATE: int = 42

    # LightGBM超参数
    LGBM_N_ESTIMATORS: int = 200
    LGBM_MAX_DEPTH: int = -1
    LGBM_LEARNING_RATE: float = 0.1
    LGBM_NUM_LEAVES: int = 31
    LGBM_SUBSAMPLE: float = 0.8
    LGBM_COLSAMPLE_BYTREE: float = 0.8

    # LSTM超参数
    LSTM_HIDDEN_SIZE: int = 64
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    LSTM_LEARNING_RATE: float = 0.001

    # 随机森林额外树数量（用于置信区间估计）
    RF_N_ESTIMATORS: int = 100

    # 时序交叉验证
    CV_N_SPLITS: int = 5

    # 集成模型配置
    ENSEMBLE_WEIGHTS: Optional[Dict] = None  # None表示自动计算

    # ==================== SHAP配置 ====================
    # 全局分析时使用的样本数量
    SHAP_BACKGROUND_SAMPLES: int = 100

    # ==================== LLM配置 ====================
    # DeepSeek API配置
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # 解释详细程度：brief(简要), standard(标准), detailed(详细)
    DEFAULT_DETAIL_LEVEL: str = "standard"

    # LLM生成配置
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000

    # ==================== API配置 ====================
    # FastAPI配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # CORS配置
    CORS_ORIGINS: list = ["*"]

    # ==================== Streamlit配置 ====================
    STREAMLIT_PORT: int = 8501

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ==================== XAI配置 ====================
    LIME_NUM_FEATURES: int = 10
    PDP_GRID_RESOLUTION: int = 50
    ICE_GRID_RESOLUTION: int = 50
    ICE_SAMPLE_SIZE: int = 50
    SHAP_INTERACTION_ENABLED: bool = True

    # ==================== UI配置 ====================
    DEFAULT_MODEL: str = "xgboost"
    SIDEBAR_WIDTH: int = 300
    CHART_HEIGHT: int = 400

    # ==================== 异常检测配置 ====================
    ANOMALY_THRESHOLD_PCT: float = 15.0  # 异常偏差阈值(%)
    ANOMALY_MAX_RESULTS: int = 20  # 最大异常返回数

    # ==================== 报告配置 ====================
    REPORT_OUTPUT_DIR: Path = PROJECT_ROOT / "reports"

    # ==================== 对话分析配置 ====================
    DIALOGUE_MAX_HISTORY: int = 5  # 对话历史轮数
    DIALOGUE_INTENT_TEMPLATE: str = "standard"  # standard/brief/detailed

    def __init__(self, **kwargs):
        """初始化配置，确保必要目录存在"""
        super().__init__(**kwargs)
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.REPORT_OUTPUT_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
