"""实验运行脚本"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data import DataLoader, DataPreprocessor
from models import (
    KNNModel,
    LogisticRegressionModel,
    NeuralNetworkModel,
    RandomForestModel
)
from utils import ModelEvaluator
from config import Config


def run_single_model(model_class, model_name: str, X_train, X_test, y_train, y_test, 
                    preprocessor: DataPreprocessor):
    """
    运行单个模型
    
    Args:
        model_class: 模型类
        model_name: 模型名称
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
        preprocessor: 数据预处理器
    """
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name}")
    print(f"{'='*60}")
    
    # 预处理数据
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 创建和训练模型
    model = model_class()
    model.fit(X_train_processed, y_train)
    
    # 预测
    y_pred = model.predict(X_test_processed)
    
    # 评估
    evaluator = ModelEvaluator(model_name)
    evaluator.print_report(y_test, y_pred)
    
    # 尝试获取概率预测
    try:
        y_proba = model.predict_proba(X_test_processed)
        evaluator.plot_roc_curve(y_test, y_proba, 
                               save_path=Config.PLOTS_DIR / f"{model_name}_roc.png")
    except:
        pass
    
    # 保存模型
    model_path = Config.MODELS_DIR / f"{model_name.lower()}.pkl"
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存结果
    results_path = Config.RESULTS_DIR / f"{model_name.lower()}_results.csv"
    evaluator.save_results(results_path)
    print(f"结果已保存到: {results_path}")
    
    return model, evaluator


def run_all_models():
    """运行所有模型"""
    # 创建输出目录
    Config.create_directories()
    
    # 加载数据
    print("加载数据...")
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.load_train_test_split()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 定义要运行的模型
    models_to_run = [
        (KNNModel, "KNN"),
        (LogisticRegressionModel, "LogisticRegression"),
        (NeuralNetworkModel, "NeuralNetwork"),
        (RandomForestModel, "RandomForest")
    ]
    
    results = {}
    
    # 运行每个模型
    for model_class, model_name in models_to_run:
        try:
            model, evaluator = run_single_model(
                model_class, model_name,
                X_train, X_test, y_train, y_test,
                preprocessor
            )
            results[model_name] = evaluator.results
        except Exception as e:
            print(f"运行 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "="*60)
    print("所有模型结果总结")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: Accuracy = {result['accuracy']:.4f}, "
              f"F1 = {result['f1_score']:.4f}")
    
    return results


if __name__ == "__main__":
    run_all_models()

