# 泰坦尼克号数据集分析项目

## 项目简介

这是一个基于泰坦尼克号乘客数据的完整机器学习项目，采用模块化设计，包含数据探索性分析（EDA）和多种分类算法的实现。项目旨在预测乘客在泰坦尼克号沉船事件中的生存情况。

## 项目结构

```
Titannic Dataset Analysis/
├── config/                      # 配置模块
│   ├── __init__.py
│   └── settings.py             # 项目配置管理
├── data/                        # 数据模块
│   ├── __init__.py
│   ├── loader.py               # 数据加载器
│   └── preprocessor.py         # 数据预处理器
├── eda/                         # 探索性数据分析模块
│   ├── __init__.py
│   ├── analysis.py             # EDA分析器
│   └── data_profiler.py        # 数据概览和可视化
├── models/                      # 模型模块
│   ├── __init__.py
│   ├── base.py                 # 基础模型类
│   ├── knn.py                  # K近邻模型
│   ├── logistic_regression.py  # 逻辑回归模型
│   ├── neural_network.py       # 神经网络模型
│   └── random_forest.py        # 随机森林模型
├── utils/                       # 工具模块
│   ├── __init__.py
│   └── metrics.py              # 模型评估工具
├── experiments/                # 实验模块
│   ├── __init__.py
│   └── run_experiments.py      # 实验运行脚本
├── output/                      # 输出目录（自动创建）
│   ├── models/                 # 保存的模型
│   ├── results/                # 评估结果
│   └── plots/                  # 图表
├── train.csv                    # 训练数据集
├── main.py                      # 主入口脚本
├── visualize_data.py            # 数据可视化脚本
├── requirements.txt             # 依赖包列表
├── report.docx                  # 项目报告文档
├── README.md                    # 本文件
├── DATA_VISUALIZATION_GUIDE.md  # 数据可视化指南
└── MIGRATION_GUIDE.md          # 迁移指南
```

## 模块化设计优势

### 1. **配置管理** (`config/`)
- 集中管理所有配置参数
- 易于修改和维护
- 支持不同环境的配置

### 2. **数据模块** (`data/`)
- `DataLoader`: 统一的数据加载接口
- `DataPreprocessor`: 可复用的数据预处理流程
- 避免代码重复

### 3. **EDA模块** (`eda/`)
- `EDAAnalyzer`: 封装所有EDA分析功能
- 支持图表保存
- 可单独运行或集成使用

### 4. **模型模块** (`models/`)
- `BaseModel`: 统一的模型接口
- 每个模型独立实现，易于扩展
- 支持模型保存和加载

### 5. **工具模块** (`utils/`)
- `ModelEvaluator`: 统一的评估接口
- 支持多种评估指标
- 自动生成报告和可视化

## 数据集说明

- **数据来源**: Kaggle 泰坦尼克号数据集
- **目标变量**: `Survived` (0 = 未生还, 1 = 生还)
- **主要特征**:
  - `Pclass`: 乘客舱位等级 (1, 2, 3)
  - `Sex`: 性别
  - `Age`: 年龄
  - `SibSp`: 兄弟姐妹/配偶数量
  - `Parch`: 父母/子女数量
  - `Fare`: 票价
  - `Embarked`: 登船港口
  - `Cabin`: 船舱号

## 环境要求

### Python 版本
Python 3.7+

### 安装依赖
```bash
pip install -r requirements.txt
```

依赖包包括：
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- seaborn >= 0.11.0（用于美观的数据可视化）

## 使用说明

### 方式一：使用主入口脚本（推荐）

#### 1. 运行数据概览和可视化（推荐先运行）
```bash
python main.py profile
```
这会生成美观的数据信息表格、分布图、缺失值可视化、相关性热力图等。

#### 2. 运行探索性数据分析
```bash
python main.py eda
```

#### 3. 运行单个模型
```bash
# K近邻
python main.py model --model knn

# 逻辑回归
python main.py model --model logistic

# 神经网络
python main.py model --model neural

# 随机森林
python main.py model --model rf
```

#### 4. 运行所有模型
```bash
python main.py all
```

### 快速数据可视化

```bash
python main.py eda
```



所有图表会自动保存到 `output/plots/` 目录。

### 方式二：使用实验脚本

直接运行实验脚本，会依次运行所有模型：
```bash
python experiments/run_experiments.py
```

### 方式三：在代码中使用模块

```python
from data import DataLoader, DataPreprocessor
from models import KNNModel
from utils import ModelEvaluator

# 加载数据
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.load_train_test_split()

# 预处理
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 训练模型
model = KNNModel()
model.fit(X_train_processed, y_train)

# 评估
y_pred = model.predict(X_test_processed)
evaluator = ModelEvaluator("KNN")
evaluator.print_report(y_test, y_pred)
```

## 模型说明

### 1. K近邻 (KNN)
- **参数**: K=5, 欧氏距离
- **特点**: 简单直观，适合小数据集

### 2. 逻辑回归 (Logistic Regression)
- **参数**: max_iter=1000
- **特点**: 线性模型，可解释性强

### 3. 神经网络 (Neural Network)
- **结构**: 隐藏层 (64, 32)
- **激活函数**: ReLU
- **优化器**: Adam
- **特点**: 非线性建模能力强

### 4. 随机森林 (Random Forest)
- **特点**: 使用随机搜索进行超参数优化
- **搜索参数**: n_estimators, max_depth, min_samples_split等
- **交叉验证**: 5折

## 数据预处理流程

所有模型使用统一的数据预处理流程：

1. **特征识别**:
   - 自动识别数值特征和类别特征
   - 支持强制指定类别特征（如Pclass）

2. **数值特征处理**:
   - 缺失值填充: 使用训练集的中位数
   - 标准化: Z-score标准化

3. **类别特征处理**:
   - 缺失值填充: 使用训练集的众数
   - 编码: One-Hot编码

4. **数据划分**:
   - 训练集/测试集: 80/20
   - 分层抽样保持类别分布
   - 随机种子: 42

## 评估指标

所有模型使用以下指标进行评估：

- **准确率 (Accuracy)**: 正确预测的比例
- **精确率 (Precision)**: 正例预测的准确性
- **召回率 (Recall)**: 正例的覆盖率
- **F1分数 (F1 Score)**: 精确率和召回率的调和平均
- **ROC-AUC**: ROC曲线下面积（如果支持概率预测）
- **混淆矩阵 (Confusion Matrix)**: 详细的分类结果矩阵
- **分类报告 (Classification Report)**: 包含所有指标的详细报告

## 输出文件

运行实验后，会在 `output/` 目录下生成：

- `models/`: 保存的训练好的模型（.pkl格式）
- `results/`: 评估结果（CSV格式）
- `plots/`: 可视化图表（PNG格式，包括EDA图表和ROC曲线）

## 扩展性

### 添加新模型

1. 在 `models/` 目录下创建新文件
2. 继承 `BaseModel` 类
3. 实现 `_create_model` 方法
4. 在 `models/__init__.py` 中导出
5. 在 `config/settings.py` 中添加配置

示例：
```python
from .base import BaseModel
from sklearn.svm import SVC

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__('SVM', **kwargs)
    
    def _create_model(self, **kwargs):
        return SVC(**kwargs)
```

### 修改配置

直接编辑 `config/settings.py` 文件，修改相应的配置参数。

### 自定义预处理

继承 `DataPreprocessor` 类，重写相应方法。

