"""K近邻模型"""
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel
from config import Config


class KNNModel(BaseModel):
    """K近邻分类模型"""
    
    def __init__(self, **kwargs):
        """
        初始化KNN模型
        
        Args:
            **kwargs: KNN参数
                n_neighbors: 邻居数量，默认5
                weights: 权重，默认'uniform'
                metric: 距离度量，默认'minkowski'
                p: 距离参数，默认2（欧氏距离）
        """
        default_params = Config.MODELS['knn'].copy()
        default_params.update(kwargs)
        super().__init__('KNN', **default_params)
        self.params = default_params
    
    def _create_model(self, **kwargs):
        """创建KNN模型实例"""
        params = self.params.copy()
        params.update(kwargs)
        return KNeighborsClassifier(**params)

