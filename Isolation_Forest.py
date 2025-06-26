import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# 1. 加载数据
file_path = "mqtt_data.csv"
df = pd.read_csv(file_path)

# 2. 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(df[['x', 'y', 'z']].values)

# 3. 训练 Isolation Forest 模型
iso_forest = IsolationForest(n_estimators=300, max_samples=128, contamination='auto', random_state=42)
iso_forest.fit(X)

# 4. 获取异常分数（负路径长度）
anomaly_scores = iso_forest.score_samples(X)  # 这是负的平均路径长度
df['anomaly_score'] = anomaly_scores

# 5. 动态阈值计算（基于负路径长度）
Q1 = np.percentile(anomaly_scores, 25)
Q3 = np.percentile(anomaly_scores, 75)
IQR = Q3 - Q1
threshold_iqr = Q1 - 1.5 * IQR

skewness = stats.skew(anomaly_scores)
if skewness <= -1:
    quantile_value = 0.05
elif -1 < skewness < -0.5:
    quantile_value = 0.1
elif -0.5 <= skewness < 0:
    quantile_value = 0.05


threshold_quantile = np.quantile(anomaly_scores, quantile_value)



# 选择更严格的阈值
threshold = min(threshold_iqr, threshold_quantile)

# 6. 判定异常点：异常为 -1，正常为 1
df['is_outlier'] = np.where(anomaly_scores < threshold, -1, 1)

# 7. 添加监督学习标签（0=正常，1=异常）
df['label'] = (df['is_outlier'] == -1).astype(int)

# 8. 输出异常点
outliers = df[df['is_outlier'] == -1]
print("异常点信息（包含异常分数）:")
print(outliers[['time', 'x', 'y', 'z', 'anomaly_score', 'is_outlier']].round(4))
print(f"\n异常点总数: {len(outliers)}")

# 9. 3D 可视化
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

# 10. 异常分数直方图（带最终阈值线）
plt.figure(figsize=(8, 6))
plt.hist(anomaly_scores, bins=50, color='gray', edgecolor='black', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title('Anomaly Scores Distribution')
plt.legend()
plt.show()

# 11. 多阈值可视化（IQR、分位数、最终）
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, color='gray', alpha=0.7, label='Scores')
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

# 12. 原始论文分数 s(x) 对比计算（正确方式！）
max_samples = iso_forest.max_samples_
c_n = 2 * (np.log(max_samples - 1) + 0.5772)
avg_path_length = -anomaly_scores  # ← 负号是关键，恢复平均路径长度
original_score = np.power(2, -avg_path_length / c_n)

# 示例打印
print(f"\nScikit-Learn 分数示例: {anomaly_scores[0]:.4f}")
print(f"原始论文分数示例: {original_score[0]:.4f}")
print(f"IQR 下界阈值: {threshold_iqr:.4f}")
print(f"分位数阈值: {threshold_quantile:.4f}")
print(f"最终使用阈值: {threshold:.4f}")

# 13. 保存带标签数据
output_file = "mqtt_data_labeled.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ 带标签的新数据已保存至：{output_file}") 
