# 泰坦尼克号数据集分析项目

## 项目简介

这是一个基于泰坦尼克号乘客数据的完整机器学习项目，包含数据探索性分析（EDA）和多种分类算法的实现。项目旨在预测乘客在泰坦尼克号沉船事件中的生存情况。

## 项目结构

```
Titannic Dataset Analysis/
├── train.csv                    # 训练数据集
├── Titannic EDA.py              # 探索性数据分析脚本
├── KNN.py                       # K近邻分类算法
├── logistics_regression.py      # 逻辑回归分类算法
├── neural_network.py            # 神经网络分类算法
├── Random_forest.py             # 随机森林分类算法（含超参数调优）
├── report.docx                  # 项目报告文档
└── README.md                    # 本文件
```

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

## 文件说明

### 1. Titannic EDA.py - 探索性数据分析

该文件提供了全面的数据探索功能，包含以下分析函数：

- `eda_overall_survival_rate()`: 整体生存率统计
- `eda_survival_by_sex()`: 按性别分析生存率
- `eda_survival_by_pclass()`: 按舱位等级分析生存率
- `eda_age_distribution_and_survival()`: 年龄分布与生存率关系
- `eda_family_size_and_isalone()`: 家庭规模与是否独自出行对生存的影响
- `eda_embarked_and_survival()`: 登船港口与生存率关系
- `eda_fare_and_cabin()`: 票价和船舱信息分析

**使用方法**:
```python
python Titannic EDA.py
```

### 2. KNN.py - K近邻分类

使用K近邻算法进行生存预测。

**特点**:
- K值: 5 (可调整)
- 距离度量: 欧氏距离 (Minkowski, p=2)
- 权重: 均匀权重

**运行**:
```python
python KNN.py
```

### 3. logistics_regression.py - 逻辑回归分类

使用逻辑回归进行二分类预测。

**特点**:
- 最大迭代次数: 1000
- 使用标准化特征以提高收敛速度

**运行**:
```python
python logistics_regression.py
```

### 4. neural_network.py - 神经网络分类

使用多层感知机（MLP）进行分类。

**网络结构**:
- 隐藏层: (2, 1) - 两个隐藏层，分别有2个和1个神经元
- 激活函数: ReLU
- 优化器: Adam
- L2正则化: α = 1e-4
- 学习率: 1e-3
- 最大迭代次数: 300

**运行**:
```python
python neural_network.py
```

### 5. Random_forest.py - 随机森林分类

使用随机森林算法，并通过随机搜索进行超参数调优。

**特点**:
- 使用 `RandomizedSearchCV` 进行超参数优化
- 搜索参数包括:
  - `n_estimators`: 树的数量 (1-100)
  - `max_depth`: 最大深度
  - `min_samples_split`: 分裂所需最小样本数
  - `min_samples_leaf`: 叶子节点最小样本数
  - `max_features`: 每次分裂考虑的特征数
  - `class_weight`: 类别权重
- 5折交叉验证
- 搜索迭代次数: 40次

**运行**:
```python
python Random_forest.py
```

## 数据预处理流程

所有模型脚本都遵循统一的数据预处理流程：

1. **特征分离**:
   - 数值特征: `Age`, `SibSp`, `Parch`, `Fare`
   - 类别特征: `Pclass` (强制转换为类别), `Sex`, `Embarked`, `Cabin`, `Name`, `Ticket`

2. **数值特征处理**:
   - 缺失值填充: 使用训练集的中位数填充
   - 标准化: 使用 `StandardScaler` 进行Z-score标准化

3. **类别特征处理**:
   - 缺失值填充: 使用训练集的众数填充
   - 编码: 使用 `OneHotEncoder` 进行独热编码

4. **数据划分**:
   - 训练集/测试集比例: 80/20
   - 使用分层抽样 (`stratify=y`) 保持类别分布
   - 随机种子: 42

## 评估指标

所有模型都使用以下指标进行评估：

- **准确率 (Accuracy)**: 正确预测的比例
- **F1分数 (F1 Score)**: 精确率和召回率的调和平均
- **召回率 (Recall)**: 真正例率
- **混淆矩阵 (Confusion Matrix)**: 详细的分类结果矩阵
- **分类报告 (Classification Report)**: 包含精确率、召回率、F1分数等详细指标

## 环境要求

### Python 版本
Python 3.7+

### 依赖包
```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

或者使用 requirements.txt (如果存在):
```bash
pip install -r requirements.txt
```

## 使用说明

1. **准备数据**: 确保 `train.csv` 文件在项目根目录下

2. **运行EDA**: 首先运行探索性数据分析了解数据特征
   ```bash
   python "Titannic EDA.py"
   ```

3. **运行模型**: 可以单独运行任意一个模型脚本
   ```bash
   python KNN.py
   python logistics_regression.py
   python neural_network.py
   python Random_forest.py
   ```

4. **查看结果**: 每个脚本运行后会输出详细的评估指标和分类报告

## 注意事项

1. **文件路径**: 部分脚本中的文件路径可能需要根据实际情况修改（当前为Windows路径格式）

2. **数据路径**: 如果 `train.csv` 不在当前目录，需要修改各脚本中的文件路径

3. **可视化**: EDA脚本会显示多个图表，需要图形界面支持

4. **计算资源**: 随机森林的超参数搜索可能需要较长时间，建议在性能较好的机器上运行

## 项目特点

- ✅ 完整的数据预处理流程
- ✅ 多种机器学习算法实现
- ✅ 详细的探索性数据分析
- ✅ 统一的评估指标体系
- ✅ 超参数自动调优（随机森林）
- ✅ 规范的代码结构和注释

## 后续改进建议

1. 特征工程: 可以尝试创建更多衍生特征（如标题提取、家庭规模组合等）
2. 模型集成: 可以尝试将多个模型进行集成（投票、堆叠等）
3. 交叉验证: 可以添加更详细的交叉验证分析
4. 可视化增强: 可以添加更多模型性能可视化（ROC曲线、学习曲线等）

## 作者

本项目用于学习和研究机器学习分类问题。

## 许可证

本项目仅供学习和研究使用。

