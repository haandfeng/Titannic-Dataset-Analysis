import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    confusion_matrix
)

# 配置：ID 列名和标签列名
ID_COL = "PassengerId"   # 你的文件里是这个
LABEL_COL = "Survived"   # 你之前说叫 survival，如果真叫 survival 就改这里

# 哪些“长得像数字但其实是类别”的列要当类别处理
FORCED_CATEGORICAL = ["Pclass"]

# ========== 1. 读入数据 ==========
df = pd.read_csv(r"C:\Users\zhy20\Desktop\研一\ECE225\project\titanic\dataset\train.csv")

y = df[LABEL_COL]

# 去掉 label 和 ID，只保留特征
drop_cols = [LABEL_COL]
if ID_COL in df.columns:
    drop_cols.append(ID_COL)
X = df.drop(columns=drop_cols)

# ========== 2. 把某些 int 列强行当类别用（比如 Pclass） ==========
for col in FORCED_CATEGORICAL:
    if col in X.columns:
        X[col] = X[col].astype("object")  # 转成 object，后面会走类别特征流程

# 现在再按 dtype 分数值 / 类别特征
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

# 4.1 在训练集上计算中位数
num_medians = X_train_num.median()

# 4.2 用训练集中位数填充训练 & 测试
X_train_num_filled = X_train_num.fillna(num_medians)
X_test_num_filled  = X_test_num.fillna(num_medians)

# 4.3 标准化
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_filled)
X_test_num_scaled  = scaler.transform(X_test_num_filled)

# ========== 5. 类别特征：众数填充 + One-Hot ==========
if len(categorical_features) > 0:
    X_train_cat = X_train[categorical_features].astype("object").copy()
    X_test_cat  = X_test[categorical_features].astype("object").copy()

    # 5.1 在训练集上计算每列众数
    cat_modes = X_train_cat.mode(dropna=True).iloc[0]

    # 5.2 用训练集众数填充训练 & 测试
    X_train_cat_filled = X_train_cat.fillna(cat_modes)
    X_test_cat_filled  = X_test_cat.fillna(cat_modes)

    # 5.3 One-Hot 编码（包括 Pclass）
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_oh = onehot.fit_transform(X_train_cat_filled)
    X_test_cat_oh  = onehot.transform(X_test_cat_filled)

    # 5.4 数值 + 类别 拼在一起
    X_train_final = np.hstack([X_train_num_scaled, X_train_cat_oh])
    X_test_final  = np.hstack([X_test_num_scaled, X_test_cat_oh])
else:
    X_train_final = X_train_num_scaled
    X_test_final  = X_test_num_scaled

# ========== 6. 训练逻辑回归模型 ==========
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_final, y_train)

# ========== 7. 评估 ==========
y_pred = log_reg.predict(X_test_final)

print("===== Logistic Regression (Pclass one-hot) =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)