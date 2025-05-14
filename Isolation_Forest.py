import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
# file_path = "/home/xingxin/3d_data_with_outliers.csv"
file_path = "test3.csv"
df = pd.read_csv(file_path)

# 2. 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(df[['x', 'y', 'z']].values)

# 3. 训练Isolation Forest模型
iso_forest = IsolationForest(n_estimators=300, max_samples=128, contamination='auto', random_state=42)
iso_forest.fit(X)

# 4. 获取异常分数并添加到 DataFrame
anomaly_scores = iso_forest.score_samples(X)
df['anomaly_score'] = anomaly_scores  # 关键修改：添加分数列

# 5. 动态阈值判定异常
# threshold = np.quantile(anomaly_scores, 0.05)
# df['is_outlier'] = np.where(anomaly_scores < threshold, -1, 1)

import scipy.stats as stats

# 计算 IQR 方法的阈值
Q1 = np.percentile(anomaly_scores, 25)
Q3 = np.percentile(anomaly_scores, 75)
IQR = Q3 - Q1
threshold_iqr = Q1 - 1.5 * IQR

# 计算数据的偏态（skewness）
skewness = stats.skew(anomaly_scores)

# 分位数阈值动态调整
if skewness < -1:
    quantile_value = 0.01
elif skewness > 1:
    quantile_value = 0.01
else:
    quantile_value = 0.05

threshold_quantile = np.quantile(anomaly_scores, quantile_value)
threshold = min(threshold_iqr, threshold_quantile)

df['is_outlier'] = np.where(anomaly_scores < threshold, -1, 1)

# 6. 输出异常点（包含异常分数）
outliers = df[df['is_outlier'] == -1]
print("异常点信息（包含异常分数）:")
print(outliers[['timestamp','x', 'y', 'z', 'anomaly_score', 'is_outlier']].round(4))  # 保留4位小数

print(f"\n异常点总数: {len(outliers)}")
# 7. 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
normal = df[df['is_outlier'] == 1]
abnormal = df[df['is_outlier'] == -1]
ax.scatter(normal['x'], normal['y'], normal['z'], c='blue', label='Normal', alpha=0.5)
ax.scatter(abnormal['x'], abnormal['y'], abnormal['z'], c='red', label='Anomaly', alpha=0.7)
ax.view_init(elev=20, azim=45)
ax.set_title('3D Anomaly Detection')
ax.legend()
plt.show()

# 8. 直方图
plt.figure(figsize=(8, 6))
plt.hist(anomaly_scores, bins=50, color='gray', edgecolor='black', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title('Anomaly Scores Distribution')
plt.legend()
plt.show()

# 8. 可视化及阈值合理性检验（新增）
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, color='gray', alpha=0.7, label='Scores')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.title(f"Skewness = {skewness:.2f}, Threshold = {threshold:.2f}")
plt.legend()
plt.show()

# 获取平均路径长度 E(h(x)) 和标准化常数 c(n)
max_samples = iso_forest.max_samples_  # 128
c_n = 2 * (np.log(max_samples - 1) + 0.5772)  # 约 10.8428

# 计算原始论文的分数 s(x)
path_lengths = iso_forest.decision_function(X)  # 返回值为 2 * (0.5 - s(x))，需逆推
s_x = 0.5 - path_lengths / 2
original_score = np.power(2, -path_lengths / c_n)  # 原始论文的 s(x)

plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, color='gray', edgecolor='black', alpha=0.7, label='Scores')

# 三条阈值线
plt.axvline(threshold_iqr, color='green', linestyle='--', linewidth=2, label=f'IQR Threshold ({threshold_iqr:.2f})')
plt.axvline(threshold_quantile, color='orange', linestyle='--', linewidth=2, label=f'Quantile Threshold ({threshold_quantile:.2f})')
plt.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Final Threshold ({threshold:.2f})')

plt.title(f"Anomaly Score Distribution (Skewness = {skewness:.2f})")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 输出对比
print(f"Scikit-Learn 分数: {anomaly_scores[0]:.4f}")
print(f"原始论文分数: {original_score[0]:.4f}")
print(f"IQR 下界阈值 (threshold_iqr): {threshold_iqr:.4f}")
print(f"分位数阈值 (threshold_quantile): {threshold_quantile:.4f}")
print(f"最终使用的阈值 (threshold): {threshold:.4f}")
