import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    confusion_matrix
)
from scipy.stats import randint

# 配置：ID 列名和标签列名
ID_COL = "PassengerId"   # 如果没有就删掉或者设为 None
LABEL_COL = "Survived"   # 如果你的列叫 survival，就改成 "survival"

# 哪些“看起来是数字、但实际是类别”的列
FORCED_CATEGORICAL = ["Pclass"]

# ========== 1. 读入数据 ==========
df = pd.read_csv(r"C:\Users\zhy20\Desktop\研一\ECE225\project\titanic\dataset\train.csv")

y = df[LABEL_COL]

drop_cols = [LABEL_COL]
if ID_COL in df.columns:
    drop_cols.append(ID_COL)
X = df.drop(columns=drop_cols)

# ========== 2. 把 Pclass 强制当类别列 ==========
for col in FORCED_CATEGORICAL:
    if col in X.columns:
        X[col] = X[col].astype("object")

# 按 dtype 分数值 / 类别特征
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

print("数值特征:", numeric_features)
print("类别特征:", categorical_features)

# ========== 3. 划分训练 / 测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ========== 4. 数值特征：中位数填充 + 标准化 ==========
X_train_num = X_train[numeric_features].copy()
X_test_num  = X_test[numeric_features].copy()

# 在训练集上计算每列中位数
num_medians = X_train_num.median()

# 用训练集的中位数填充训练 & 测试
X_train_num_filled = X_train_num.fillna(num_medians)
X_test_num_filled  = X_test_num.fillna(num_medians)

# 标准化（对 RF 不是必须，但统一一下）
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_filled)
X_test_num_scaled  = scaler.transform(X_test_num_filled)

# ========== 5. 类别特征：众数填充 + One-Hot ==========
if len(categorical_features) > 0:
    X_train_cat = X_train[categorical_features].astype("object").copy()
    X_test_cat  = X_test[categorical_features].astype("object").copy()

    # 训练集上算每列众数
    cat_modes = X_train_cat.mode(dropna=True).iloc[0]

    # 用众数填充缺失
    X_train_cat_filled = X_train_cat.fillna(cat_modes)
    X_test_cat_filled  = X_test_cat.fillna(cat_modes)

    # 注意：不传 sparse / sparse_output，最后用 .toarray()
    onehot = OneHotEncoder(handle_unknown="ignore")
    X_train_cat_oh = onehot.fit_transform(X_train_cat_filled).toarray()
    X_test_cat_oh  = onehot.transform(X_test_cat_filled).toarray()

    # 数值 + 类别拼在一起
    X_train_final = np.hstack([X_train_num_scaled, X_train_cat_oh])
    X_test_final  = np.hstack([X_test_num_scaled,  X_test_cat_oh])
else:
    X_train_final = X_train_num_scaled
    X_test_final  = X_test_num_scaled

# ========== 6. 定义基础随机森林模型 ==========
base_rf = RandomForestClassifier(
    random_state=42
)

# ========== 7. 定义超参数搜索空间 ==========
param_dist = {
    # 树的数量：越多越稳，但越慢
    "n_estimators": randint(1, 100),

    # 最大深度：防止严重过拟合
    "max_depth": [None, 5, 8, 12, 16, 20],

    # 内部分裂所需最小样本数
    "min_samples_split": randint(2, 10),

    # 叶子节点最小样本数（>1 通常能提升泛化）
    "min_samples_leaf": randint(1, 6),

    # 每次分裂考虑多少特征
    "max_features": ["sqrt", "log2", 0.5, 0.8],

    # 是否平衡类别权重（可能会略微牺牲 accuracy 换 recall）
    "class_weight": [None, "balanced"]
}

# ========== 8. 使用 RandomizedSearchCV 做超参数搜索 ==========
rf_search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_dist,
    n_iter=40,              # 搜索次数，可以调小/调大
    cv=5,                   # 5 折交叉验证
    scoring="accuracy",     # 以 accuracy 作为调参目标
    n_jobs=-1,              # 用所有 CPU 核
    random_state=42,
    verbose=1
)

print("开始超参数搜索（RandomizedSearchCV）...")
rf_search.fit(X_train_final, y_train)
print("搜索结束。")
print("最优参数：", rf_search.best_params_)
print("交叉验证最佳 accuracy：", rf_search.best_score_)

# 取出最优模型
best_rf = rf_search.best_estimator_

# ========== 9. 在测试集上评估最优随机森林 ==========
y_pred = best_rf.predict(X_test_final)

print("\n===== Random Forest (tuned, Pclass one-hot) =====")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)