import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    confusion_matrix
)

# 配置：ID 列名和标签列名
ID_COL = "PassengerId"   # Kaggle 默认
LABEL_COL = "Survived"   # 如果是 'survival' 就改这里

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

# ========== 4. 数值特征：中位数填充 + 标准化（KNN 必须标准化） ==========
X_train_num = X_train[numeric_features].copy()
X_test_num  = X_test[numeric_features].copy()

num_medians = X_train_num.median()
X_train_num_filled = X_train_num.fillna(num_medians)
X_test_num_filled  = X_test_num.fillna(num_medians)

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_filled)
X_test_num_scaled  = scaler.transform(X_test_num_filled)

# ========== 5. 类别特征：众数填充 + One-Hot ==========
if len(categorical_features) > 0:
    X_train_cat = X_train[categorical_features].astype("object").copy()
    X_test_cat  = X_test[categorical_features].astype("object").copy()

    cat_modes = X_train_cat.mode(dropna=True).iloc[0]
    X_train_cat_filled = X_train_cat.fillna(cat_modes)
    X_test_cat_filled  = X_test_cat.fillna(cat_modes)

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_oh = onehot.fit_transform(X_train_cat_filled)
    X_test_cat_oh  = onehot.transform(X_test_cat_filled)

    X_train_final = np.hstack([X_train_num_scaled, X_train_cat_oh])
    X_test_final  = np.hstack([X_test_num_scaled, X_test_cat_oh])
else:
    X_train_final = X_train_num_scaled
    X_test_final  = X_test_num_scaled

# ========== 6. 训练 KNN 模型 ==========
knn = KNeighborsClassifier(
    n_neighbors=5,      # 之后可以改成 3/7/9 对比效果
    weights="uniform",
    metric="minkowski",
    p=2                 # 欧氏距离
)
knn.fit(X_train_final, y_train)

# ========== 7. 评估 ==========
y_pred = knn.predict(X_test_final)

print("===== KNN (Pclass one-hot) =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)