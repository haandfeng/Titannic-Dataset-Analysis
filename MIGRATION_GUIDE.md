# 迁移指南：从旧代码到模块化结构

本指南帮助您了解如何从旧的脚本式代码迁移到新的模块化结构。

## 主要变化

### 1. 项目结构变化

**旧结构**:
```
Titannic Dataset Analysis/
├── KNN.py
├── logistics_regression.py
├── neural_network.py
├── Random_forest.py
└── Titannic EDA.py
```

**新结构**:
```
Titannic Dataset Analysis/
├── config/          # 配置管理
├── data/            # 数据模块
├── eda/             # EDA模块
├── models/          # 模型模块
├── utils/           # 工具模块
├── experiments/     # 实验模块
└── main.py          # 主入口
```

### 2. 代码使用方式变化

#### 旧方式：直接运行脚本
```bash
python KNN.py
python logistics_regression.py
```

#### 新方式：使用主入口
```bash
python main.py model --model knn
python main.py model --model logistic
python main.py all
```

### 3. 在代码中使用

#### 旧方式：每个文件独立实现
每个模型文件都包含完整的数据加载和预处理代码。

#### 新方式：模块化使用
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

## 功能对应关系

### EDA分析

**旧方式**:
```python
from Titannic EDA import eda_overall_survival_rate, eda_survival_by_sex
df = load_train("train.csv")
eda_overall_survival_rate(df)
eda_survival_by_sex(df)
```

**新方式**:
```python
from data import DataLoader
from eda import EDAAnalyzer

loader = DataLoader()
df = loader.load()

analyzer = EDAAnalyzer()
analyzer.overall_survival_rate(df)
analyzer.survival_by_sex(df)
# 或运行完整分析
analyzer.run_full_analysis(df)
```

### 模型训练

**旧方式** (KNN.py):
```python
df = pd.read_csv("train.csv")
# ... 大量预处理代码 ...
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_final, y_train)
y_pred = knn.predict(X_test_final)
```

**新方式**:
```python
from data import DataLoader, DataPreprocessor
from models import KNNModel

loader = DataLoader()
X_train, X_test, y_train, y_test = loader.load_train_test_split()

preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = KNNModel(n_neighbors=5)
model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)
```

## 配置管理

### 旧方式：硬编码在代码中
```python
ID_COL = "PassengerId"
LABEL_COL = "Survived"
FORCED_CATEGORICAL = ["Pclass"]
```

### 新方式：集中配置
所有配置在 `config/settings.py` 中管理，可以轻松修改：
```python
from config import Config

# 使用配置
ID_COL = Config.ID_COL
LABEL_COL = Config.LABEL_COL
FORCED_CATEGORICAL = Config.FORCED_CATEGORICAL
```

## 常见问题

### Q: 旧代码中的硬编码路径怎么办？
A: 新结构使用相对路径，数据文件放在项目根目录即可。如需修改，编辑 `config/settings.py`。

### Q: 如何修改模型参数？
A: 有两种方式：
1. 在 `config/settings.py` 中修改默认参数
2. 在创建模型时传入参数：`KNNModel(n_neighbors=10)`

### Q: 如何添加新模型？
A: 参考 `models/knn.py` 的实现，创建新文件并继承 `BaseModel` 类。

### Q: 输出文件在哪里？
A: 所有输出文件在 `output/` 目录下：
- `output/models/`: 保存的模型
- `output/results/`: 评估结果
- `output/plots/`: 图表

## 需要帮助？

查看以下文件了解更多：
- `README.md`: 完整的使用说明
- 各模块的文档字符串

