"""
LSTM模型

基于PyTorch的LSTM时序预测模型。

功能：
- 2层LSTM + 全连接输出
- 支持批量训练和预测
- 模型保存与加载
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import settings

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM预测模型

    2层LSTM + 全连接输出层。
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        初始化LSTM模型

        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度
        """
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_size)

        Returns:
            输出张量 (batch_size, output_size)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # Dropout
        dropped = self.dropout(last_output)

        # 全连接
        output = self.fc(dropped)

        return output


class LSTMPredictor:
    """
    LSTM预测器

    封装LSTM模型的训练、预测、保存和加载功能。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        sequence_length: int = 24,
        device: Optional[str] = None
    ):
        """
        初始化LSTM预测器

        Args:
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批大小
            sequence_length: 序列长度（小时）
            device: 设备 ('cuda', 'cpu', 或 None自动选择)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"LSTM使用设备: {self.device}")

        self.model: Optional[LSTMModel] = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.input_size: int = 32
        self.is_trained: bool = False

    def _prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备序列数据

        Args:
            X: 特征数组 (n_samples, n_features)
            y: 目标数组 (n_samples,)

        Returns:
            (X_sequence, y_sequence) 张量
        """
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            target = y[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        X_seq = np.array(sequences)
        y_seq = np.array(targets)

        # 转换为张量
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)

        return X_tensor, y_tensor

    def _create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        创建数据加载器

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标

        Returns:
            (train_loader, val_loader)
        """
        # 准备序列
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)

        # 训练数据加载器
        train_dataset = TensorDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 验证数据加载器
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            val_dataset = TensorDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        return train_loader, val_loader

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        训练LSTM模型

        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练目标 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            feature_columns: 特征列名列表

        Returns:
            训练历史字典
        """
        logger.info("开始训练LSTM模型...")
        logger.info(f"训练数据: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

        # 保存特征列
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [f"feature_{i}" for i in range(X_train.shape[1])]

        # 确定输入维度
        self.input_size = X_train.shape[1]

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # 创建模型
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # 创建数据加载器
        train_loader, val_loader = self._create_dataloaders(
            X_train_scaled, y_train, X_val_scaled, y_val
        )

        # 训练循环
        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # 验证阶段
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)

            # 日志输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if val_loss else ""
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Train Loss: {train_loss:.4f}{val_str}"
                )

        self.is_trained = True
        logger.info("LSTM模型训练完成")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测

        Args:
            X: 特征数组 (n_samples, n_features)

        Returns:
            预测数组 (n_samples,)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型未训练，请先调用train方法")

        self.model.eval()

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 准备序列
        X_seq, _ = self._prepare_sequences(X_scaled, np.zeros(len(X_scaled)))

        # 预测
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_seq), self.batch_size):
                batch_X = X_seq[i:i + self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())

        return np.array(predictions)

    def save(self, filepath: Path) -> Path:
        """
        保存模型

        Args:
            filepath: 保存路径

        Returns:
            保存的文件路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'feature_columns': self.feature_columns,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained
        }

        torch.save(checkpoint, filepath)
        logger.info(f"LSTM模型已保存至: {filepath}")

        return filepath

    def load(self, filepath: Path) -> 'LSTMPredictor':
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            self
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # 恢复配置
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.sequence_length = checkpoint['sequence_length']
        self.feature_columns = checkpoint['feature_columns']
        self.input_size = checkpoint['input_size']
        self.is_trained = checkpoint['is_trained']

        # 恢复标准化器
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']

        # 恢复模型
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"LSTM模型已从 {filepath} 加载")

        return self


def create_lstm_predictor(config: Optional[Dict] = None) -> LSTMPredictor:
    """
    创建LSTM预测器的便捷函数

    Args:
        config: 配置字典（可选）

    Returns:
        LSTMPredictor实例
    """
    if config is None:
        config = {}

    return LSTMPredictor(
        hidden_size=config.get('hidden_size', settings.LSTM_HIDDEN_SIZE),
        num_layers=config.get('num_layers', settings.LSTM_NUM_LAYERS),
        dropout=config.get('dropout', settings.LSTM_DROPOUT),
        learning_rate=config.get('learning_rate', settings.LSTM_LEARNING_RATE),
        epochs=config.get('epochs', settings.LSTM_EPOCHS),
        batch_size=config.get('batch_size', settings.LSTM_BATCH_SIZE)
    )


if __name__ == "__main__":
    # 演示LSTM训练
    logging.basicConfig(level=logging.INFO)

    from src.data.generator import ensure_data_exists
    from src.data.preprocessor import preprocess_data, split_train_test, get_feature_columns

    # 加载数据
    raw_df = ensure_data_exists()

    # 预处理
    processed_df, _ = preprocess_data(raw_df)

    # 划分数据集
    train_df, test_df = split_train_test(processed_df)

    # 获取特征
    feature_cols = get_feature_columns(processed_df)
    X_train = train_df[feature_cols].values
    y_train = train_df['load'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['load'].values

    # 创建并训练模型
    predictor = LSTMPredictor(epochs=10, batch_size=64)

    history = predictor.train(
        X_train, y_train,
        X_test, y_test,
        feature_columns=feature_cols
    )

    # 预测
    predictions = predictor.predict(X_test)

    # 计算误差
    mae = np.mean(np.abs(y_test[24:] - predictions))
    logger.info(f"测试集MAE: {mae:.2f} MW")

    # 保存模型
    model_path = settings.MODEL_DIR / "lstm_model.pt"
    predictor.save(model_path)
