"""主入口脚本"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data import DataLoader, DataPreprocessor
from eda import EDAAnalyzer
from models import (
    KNNModel,
    LogisticRegressionModel,
    NeuralNetworkModel,
    RandomForestModel
)
from utils import ModelEvaluator
from config import Config


def run_eda():
    """运行探索性数据分析"""
    print("="*60)
    print("运行探索性数据分析")
    print("="*60)
    
    Config.create_directories()
    loader = DataLoader()
    df = loader.load()
    
    analyzer = EDAAnalyzer(save_plots=True)
    analyzer.run_full_analysis(df)


def run_model(model_name: str):
    """运行单个模型"""
    print(f"运行模型: {model_name}")
    
    Config.create_directories()
    
    # 加载数据
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.load_train_test_split()
    
    # 预处理
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 创建模型
    model_map = {
        'knn': KNNModel,
        'logistic': LogisticRegressionModel,
        'neural': NeuralNetworkModel,
        'rf': RandomForestModel
    }
    
    if model_name.lower() not in model_map:
        print(f"未知模型: {model_name}")
        print(f"可用模型: {', '.join(model_map.keys())}")
        return
    
    model_class = model_map[model_name.lower()]
    model = model_class()
    
    # 训练
    print("训练模型...")
    model.fit(X_train_processed, y_train)
    
    # 预测和评估
    y_pred = model.predict(X_test_processed)
    evaluator = ModelEvaluator(model_name)
    evaluator.print_report(y_test, y_pred)
    
    # 保存模型
    model_path = Config.MODELS_DIR / f"{model_name.lower()}.pkl"
    model.save(model_path)
    print(f"模型已保存到: {model_path}")


def run_all_models():
    """运行所有模型"""
    from experiments.run_experiments import run_all_models
    run_all_models()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Titanic数据集分析项目')
    parser.add_argument(
        'command',
        choices=['eda', 'model', 'all'],
        help='要执行的命令: eda (探索性分析), model (单个模型), all (所有模型)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='模型名称 (knn, logistic, neural, rf) - 仅在command=model时使用'
    )
    
    args = parser.parse_args()
    
    if args.command == 'eda':
        run_eda()
    elif args.command == 'model':
        if not args.model:
            print("错误: 使用 --model 参数指定要运行的模型")
            print("可用模型: knn, logistic, neural, rf")
            return
        run_model(args.model)
    elif args.command == 'all':
        run_all_models()


if __name__ == "__main__":
    main()

