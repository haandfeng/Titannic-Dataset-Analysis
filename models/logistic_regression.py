"""逻辑回归模型"""
from sklearn.linear_model import LogisticRegression
from .base import BaseModel
from config import Config


class LogisticRegressionModel(BaseModel):
    """逻辑回归分类模型"""
    
    def __init__(self, **kwargs):
        """
        初始化逻辑回归模型
        
        Args:
            **kwargs: 逻辑回归参数
                max_iter: 最大迭代次数，默认1000
                random_state: 随机种子，默认42
        """
        default_params = Config.MODELS['logistic_regression'].copy()
        default_params.update(kwargs)
        super().__init__('LogisticRegression', **default_params)
        self.params = default_params
    
    def _create_model(self, **kwargs):
        """创建逻辑回归模型实例"""
        params = self.params.copy()
        params.update(kwargs)
        return LogisticRegression(**params)

