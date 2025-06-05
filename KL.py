import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve,accuracy_score

# 1. 读取数据
df = pd.read_csv("data.csv")

# 2. 提取特征和标签
X = df[['x', 'y', 'z']]
y = df['label']

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分测试集（用于最终评估）
X_temp_train, X_test, y_temp_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. KNN 模型调参
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
knn_grid.fit(X_temp_train, y_temp_train)
knn_best = knn_grid.best_estimator_
knn_pred = knn_best.predict(X_test)

print("\nKNN 最佳参数:", knn_grid.best_params_)
print("\nKNN 分类报告:")
print(classification_report(y_test, knn_pred))
knn_cm = confusion_matrix(y_test, knn_pred)

# 6. 逻辑回归调参
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200]
}
lr_grid = GridSearchCV(LogisticRegression(), lr_param_grid, cv=5)
lr_grid.fit(X_temp_train, y_temp_train)
lr_best = lr_grid.best_estimator_
lr_pred = lr_best.predict(X_test)

print("\n逻辑回归最佳参数:", lr_grid.best_params_)
print("\n逻辑回归分类报告:")
print(classification_report(y_test, lr_pred))
lr_cm = confusion_matrix(y_test, lr_pred)

# 7. 可视化混淆矩阵
sns.set(style="whitegrid")
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(1, 2, 2)
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# 8. ROC 曲线
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_best.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_best.predict_proba(X_test)[:, 1])
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.2f})")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 9. Precision-Recall 曲线
precision_knn, recall_knn, _ = precision_recall_curve(y_test, knn_best.predict_proba(X_test)[:, 1])
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_best.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall_knn, precision_knn, label="KNN")
plt.plot(recall_lr, precision_lr, label="Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 10. 训练集预测
knn_train_pred = knn_best.predict(X_temp_train)
lr_train_pred = lr_best.predict(X_temp_train)

# 11. 打印训练集与测试集的准确率和差值
print("\n==== 训练集 vs 测试集性能对比（判断过拟合） ====")

# KNN
knn_train_acc = accuracy_score(y_temp_train, knn_train_pred)
knn_test_acc = accuracy_score(y_test, knn_pred)
print(f"KNN 训练集准确率: {knn_train_acc:.4f}")
print(f"KNN 测试集准确率:  {knn_test_acc:.4f}")
print(f"KNN 差值（train - test）: {knn_train_acc - knn_test_acc:.4f}")

# Logistic Regression
lr_train_acc = accuracy_score(y_temp_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_pred)
print(f"\nLogistic 回归训练集准确率: {lr_train_acc:.4f}")
print(f"Logistic 回归测试集准确率:  {lr_test_acc:.4f}")
print(f"Logistic 回归差值（train - test）: {lr_train_acc - lr_test_acc:.4f}")

# 12. 简单判断是否可能过拟合/欠拟合
def judge_fit(train_acc, test_acc, model_name):
    diff = train_acc - test_acc
    if train_acc < 0.6 and test_acc < 0.6:
        print(f"{model_name} ➤ 可能欠拟合（训练和测试都差）")
    elif diff > 0.15:
        print(f"{model_name} ➤ 可能过拟合（训练很好，测试差）")
    elif abs(diff) <= 0.1:
        print(f"{model_name} ➤ 泛化能力较好")
    else:
        print(f"{model_name} ➤ 中等程度过拟合或其他问题")

print("\n==== 模型拟合情况分析 ====")
judge_fit(knn_train_acc, knn_test_acc, "KNN")
judge_fit(lr_train_acc, lr_test_acc, "Logistic Regression")
