"""项目配置管理"""
import os
from pathlib import Path

class Config:
    """项目配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 数据路径
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DATA_PATH = PROJECT_ROOT / "train.csv"
    
    # 输出路径
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = OUTPUT_DIR / "models"
    RESULTS_DIR = OUTPUT_DIR / "results"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    
    # 数据列配置
    ID_COL = "PassengerId"
    LABEL_COL = "Survived"
    FORCED_CATEGORICAL = ["Pclass"]
    
    # 数据划分配置
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    STRATIFY = True
    
    # 模型配置
    MODELS = {
        'knn': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski',
            'p': 2
        },
        'logistic_regression': {
            'max_iter': 1000,
            'random_state': 42
        },
        'neural_network': {
            'hidden_layer_sizes': (2, 1),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 1e-4,
            'learning_rate_init': 1e-3,
            'max_iter': 300,
            'random_state': 42,
            'verbose': True
        },
        'random_forest': {
            'base_params': {
                'random_state': 42
            },
            'search_params': {
                'n_iter': 40,
                'cv': 5,
                'scoring': 'accuracy',
                'n_jobs': -1,
                'random_state': 42,
                'verbose': 1
            }
        }
    }
    
    # EDA配置
    EDA_PLOT_ENABLED = True
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.OUTPUT_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.PLOTS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

