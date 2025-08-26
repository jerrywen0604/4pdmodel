import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.feature.csv")

label_col = "label"
y = train[label_col].astype(int)
X = train.drop(columns=[label_col])
X_test = test.copy()

# 简单清洗：缺失值填充
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))


X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LightGBM 数据集
dtrain = lgb.Dataset(X_tr, label=y_tr)
dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

# 参数设置
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

# 训练
model = lgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200),
    ],
)
# 验证 AUC
va_pred = model.predict(X_va, num_iteration=model.best_iteration)
auc = roc_auc_score(y_va, va_pred)
print("Validation AUC:", auc)

# 正式预测 test
test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 写出提交文件
sub = pd.DataFrame({"label": test_pred})
sub.to_csv("recommend_trx_simple.csv", index=False)
print("recommend_trx_simple.csv done")