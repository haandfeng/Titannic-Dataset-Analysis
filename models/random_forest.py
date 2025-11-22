"""随机森林模型"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from .base import BaseModel
from config import Config


class RandomForestModel(BaseModel):
    """随机森林分类模型（支持超参数搜索）"""
    
    def __init__(self, use_grid_search: bool = True, **kwargs):
        """
        初始化随机森林模型
        
        Args:
            use_grid_search: 是否使用超参数搜索，默认True
            **kwargs: 随机森林参数或搜索参数
        """
        super().__init__('RandomForest', **kwargs)
        self.use_grid_search = use_grid_search
        self.search_model = None
        self.base_params = Config.MODELS['random_forest']['base_params'].copy()
        self.search_params = Config.MODELS['random_forest']['search_params'].copy()
        self.base_params.update(kwargs)
    
    def _create_model(self, **kwargs):
        """创建随机森林模型实例"""
        params = self.base_params.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        训练模型（可选择是否进行超参数搜索）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            **kwargs: 额外的训练参数
        """
        if self.use_grid_search:
            self._fit_with_search(X_train, y_train)
        else:
            super().fit(X_train, y_train, **kwargs)
    
    def _fit_with_search(self, X_train: np.ndarray, y_train: np.ndarray):
        """使用随机搜索进行超参数优化"""
        base_rf = self._create_model()
        
        # 定义超参数搜索空间
        param_dist = {
            "n_estimators": randint(1, 100),
            "max_depth": [None, 5, 8, 12, 16, 20],
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 6),
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "class_weight": [None, "balanced"]
        }
        
        # 创建搜索对象
        self.search_model = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            **self.search_params
        )
        
        print("开始超参数搜索（RandomizedSearchCV）...")
        self.search_model.fit(X_train, y_train)
        print("搜索结束。")
        print("最优参数：", self.search_model.best_params_)
        print("交叉验证最佳得分：", self.search_model.best_score_)
        
        self.model = self.search_model.best_estimator_
        self.is_trained = True
    
    def get_best_params(self):
        """获取最优参数（仅在使用搜索时有效）"""
        if self.search_model is None:
            return None
        return self.search_model.best_params_

