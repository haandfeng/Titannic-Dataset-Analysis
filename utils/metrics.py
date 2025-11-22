"""模型评估工具"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_name: str = "Model"):
        """
        初始化评估器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.results = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            
        Returns:
            评估结果字典
        """
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_proba is not None:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                results['roc_auc'] = None
        
        self.results = results
        return results
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        打印详细的评估报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
        """
        print(f"\n{'='*60}")
        print(f"{self.model_name} 评估结果")
        print(f"{'='*60}")
        
        results = self.evaluate(y_true, y_pred)
        
        print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
        print(f"精确率 (Precision): {results['precision']:.4f}")
        print(f"召回率 (Recall): {results['recall']:.4f}")
        print(f"F1分数 (F1 Score): {results['f1_score']:.4f}")
        
        if results.get('roc_auc') is not None:
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, digits=4))
        
        print("混淆矩阵:")
        print(confusion_matrix(y_true, y_pred))
        print(f"{'='*60}\n")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      save_path: Optional[Path] = None):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            save_path: 保存路径（可选）
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, filepath: Path):
        """
        保存评估结果到文件
        
        Args:
            filepath: 保存路径
        """
        df = pd.DataFrame([self.results])
        df.to_csv(filepath, index=False)

