"""基础模型类"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import pickle


class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
            **kwargs: 模型参数
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """创建模型实例"""
        pass
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            **kwargs: 额外的训练参数
        """
        if self.model is None:
            self.model = self._create_model(**kwargs)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征数据
            
        Returns:
            预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} 不支持概率预测")
    
    def save(self, filepath: Path):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: Path):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            参数字典
        """
        if self.model is None:
            return {}
        return self.model.get_params()
    
    def set_params(self, **params):
        """
        设置模型参数
        
        Args:
            **params: 参数字典
        """
        if self.model is None:
            self.model = self._create_model(**params)
        else:
            self.model.set_params(**params)

