"""神经网络模型"""
from sklearn.neural_network import MLPClassifier
from .base import BaseModel
from config import Config


class NeuralNetworkModel(BaseModel):
    """多层感知机分类模型"""
    
    def __init__(self, **kwargs):
        """
        初始化神经网络模型
        
        Args:
            **kwargs: MLP参数
                hidden_layer_sizes: 隐藏层结构，默认(64, 32)
                activation: 激活函数，默认'relu'
                solver: 优化器，默认'adam'
                alpha: L2正则化系数，默认1e-4
                learning_rate_init: 初始学习率，默认1e-3
                max_iter: 最大迭代次数，默认300
                random_state: 随机种子，默认42
                verbose: 是否显示训练过程，默认False
        """
        default_params = Config.MODELS['neural_network'].copy()
        default_params.update(kwargs)
        super().__init__('NeuralNetwork', **default_params)
        self.params = default_params
    
    def _create_model(self, **kwargs):
        """创建MLP模型实例"""
        params = self.params.copy()
        params.update(kwargs)
        return MLPClassifier(**params)

