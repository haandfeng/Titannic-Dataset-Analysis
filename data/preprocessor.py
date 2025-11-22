"""数据预处理模块"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple, Optional
from config import Config


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, forced_categorical: Optional[List[str]] = None):
        """
        初始化预处理器
        
        Args:
            forced_categorical: 强制作为类别处理的列名列表
        """
        self.forced_categorical = forced_categorical or Config.FORCED_CATEGORICAL
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.num_medians: Optional[pd.Series] = None
        self.cat_modes: Optional[pd.Series] = None
        self._fitted = False
    
    def _identify_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        识别数值特征和类别特征
        
        Args:
            X: 特征数据
            
        Returns:
            (数值特征列表, 类别特征列表)
        """
        # 强制转换某些列为类别
        X_work = X.copy()
        for col in self.forced_categorical:
            if col in X_work.columns:
                X_work[col] = X_work[col].astype("object")
        
        numeric_features = X_work.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        
        categorical_features = X_work.select_dtypes(
            include=["object", "bool", "category"]
        ).columns.tolist()
        
        return numeric_features, categorical_features
    
    def fit(self, X_train: pd.DataFrame):
        """
        在训练集上拟合预处理器
        
        Args:
            X_train: 训练集特征
        """
        # 识别特征类型
        self.numeric_features, self.categorical_features = self._identify_features(X_train)
        
        # 处理数值特征
        if len(self.numeric_features) > 0:
            X_train_num = X_train[self.numeric_features].copy()
            self.num_medians = X_train_num.median()
            X_train_num_filled = X_train_num.fillna(self.num_medians)
            self.scaler.fit(X_train_num_filled)
        
        # 处理类别特征
        if len(self.categorical_features) > 0:
            X_train_cat = X_train[self.categorical_features].astype("object").copy()
            self.cat_modes = X_train_cat.mode(dropna=True).iloc[0]
            X_train_cat_filled = X_train_cat.fillna(self.cat_modes)
            self.onehot_encoder.fit(X_train_cat_filled)
        
        self._fitted = True
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        转换数据
        
        Args:
            X: 要转换的特征数据
            
        Returns:
            转换后的numpy数组
        """
        if not self._fitted:
            raise ValueError("预处理器尚未拟合，请先调用fit方法")
        
        # 强制转换某些列为类别
        X_work = X.copy()
        for col in self.forced_categorical:
            if col in X_work.columns:
                X_work[col] = X_work[col].astype("object")
        
        # 处理数值特征
        if len(self.numeric_features) > 0:
            X_num = X_work[self.numeric_features].copy()
            X_num_filled = X_num.fillna(self.num_medians)
            X_num_scaled = self.scaler.transform(X_num_filled)
        else:
            X_num_scaled = np.array([]).reshape(len(X), 0)
        
        # 处理类别特征
        if len(self.categorical_features) > 0:
            X_cat = X_work[self.categorical_features].astype("object").copy()
            X_cat_filled = X_cat.fillna(self.cat_modes)
            X_cat_oh = self.onehot_encoder.transform(X_cat_filled)
        else:
            X_cat_oh = np.array([]).reshape(len(X), 0)
        
        # 合并特征
        if X_num_scaled.shape[1] > 0 and X_cat_oh.shape[1] > 0:
            X_final = np.hstack([X_num_scaled, X_cat_oh])
        elif X_num_scaled.shape[1] > 0:
            X_final = X_num_scaled
        elif X_cat_oh.shape[1] > 0:
            X_final = X_cat_oh
        else:
            raise ValueError("没有可用的特征")
        
        return X_final
    
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        拟合并转换训练数据
        
        Args:
            X_train: 训练集特征
            
        Returns:
            转换后的numpy数组
        """
        self.fit(X_train)
        return self.transform(X_train)
    
    def get_feature_info(self) -> dict:
        """
        获取特征信息
        
        Returns:
            包含特征信息的字典
        """
        return {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'n_numeric': len(self.numeric_features),
            'n_categorical': len(self.categorical_features)
        }

