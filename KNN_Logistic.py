import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# 1. 读取数据
df = pd.read_csv("/home/xingxin/Downloads/Isolation_point/random_xyz_data.csv")  # 👈 替换为你的 CSV 文件

# 2. 提取特征和标签
X = df[['X', 'Y', 'Z']]  # 三轴数据
y = df['label']         # 0（正常）或 1（异常）

# 3. 标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======================
# 模型1：KNN
# ======================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# 打印分类报告
print("\n📊 KNN 分类结果：")
print(classification_report(y_test, knn_pred))

# 计算混淆矩阵
knn_cm = confusion_matrix(y_test, knn_pred)

# ======================
# 模型2：逻辑回归
# ======================
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 打印分类报告
print("\n📊 逻辑回归分类结果：")
print(classification_report(y_test, lr_pred))

# 计算混淆矩阵
lr_cm = confusion_matrix(y_test, lr_pred)

# ======================
# 可视化：混淆矩阵
# ======================

# 设置图形样式
sns.set(style="whitegrid")

# KNN 混淆矩阵
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
plt.title("KNN Confusion Matrix")
plt.xlabel("Pre Label")
plt.ylabel("True Label")

# 逻辑回归 混淆矩阵
plt.subplot(1, 2, 2)
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
plt.title("Logistic Regression  confusion Matrix")
plt.xlabel("Pre Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# ======================
# 可视化：ROC 曲线
# ======================
# 计算 ROC 曲线
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])

# 计算 AUC
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color="blue", label=f"KNN (AUC = {roc_auc_knn:.2f})")
plt.plot(fpr_lr, tpr_lr, color="green", label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("假阳性率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("ROC 曲线")
plt.legend(loc="lower right")
plt.show()

# ======================
# 可视化：Precision-Recall 曲线
# ======================
# 计算 Precision-Recall 曲线
precision_knn, recall_knn, _ = precision_recall_curve(y_test, knn.predict_proba(X_test)[:, 1])
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr.predict_proba(X_test)[:, 1])

# 绘制 Precision-Recall 曲线
plt.figure(figsize=(8, 6))
plt.plot(recall_knn, precision_knn, color="blue", label="KNN")
plt.plot(recall_lr, precision_lr, color="green", label="Logistic Regression")
plt.xlabel(" (Recall)")
plt.ylabel(" (Precision)")
plt.title("Precision-Recall ")
plt.legend(loc="lower left")
plt.show()
