"""数据加载模块"""
import pandas as pd
from pathlib import Path
from typing import Optional
from config import Config


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径，如果为None则使用配置中的默认路径
        """
        self.data_path = data_path or Config.TRAIN_DATA_PATH
    
    def load(self) -> pd.DataFrame:
        """
        加载训练数据
        
        Returns:
            包含训练数据的DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        return df
    
    def load_train_test_split(self, test_size: float = None, 
                              random_state: int = None,
                              stratify: bool = None) -> tuple:
        """
        加载数据并划分训练集和测试集
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            stratify: 是否分层抽样
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        df = self.load()
        y = df[Config.LABEL_COL]
        
        # 移除标签列和ID列
        drop_cols = [Config.LABEL_COL]
        if Config.ID_COL in df.columns:
            drop_cols.append(Config.ID_COL)
        X = df.drop(columns=drop_cols)
        
        # 使用配置或传入的参数
        test_size = test_size or Config.TEST_SIZE
        random_state = random_state or Config.RANDOM_STATE
        stratify = y if (stratify if stratify is not None else Config.STRATIFY) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test

