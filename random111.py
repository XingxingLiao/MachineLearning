import csv
import random
import argparse

# 设置参数解析器
parser = argparse.ArgumentParser(description="生成符合要求的CSV数据文件。")
parser.add_argument('--filename', type=str, default='data.csv', help='输出CSV文件名')
parser.add_argument('--rows', type=int, default=100000, help='生成数据行数')
args = parser.parse_args()

filename = args.filename
num_rows = args.rows

# 异常值生成函数
def generate_anomaly():
    while True:
        x = round(random.uniform(-0.1, 0.0), 3)
        y = round(random.uniform(-0.6, 0.1), 3)
        z = round(random.uniform(0.5, 2.0), 3)
        if not (-0.05 <= x <= -0.03 and -0.2 <= y <= -0.1 and 1.0 <= z <= 1.3):
            return x, y, z

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z', 'label'])

    for _ in range(num_rows):
        if random.random() < 0.99:
            # 正常数据
            x = round(random.uniform(-0.05,-0.03 ), 3)
            y = round(random.uniform(-0.2, -0.1), 3)
            z = round(random.uniform(1.0, 1.3), 3)
            label = 0
        else:
            # 异常数据，确保不在正常范围内
            x, y, z = generate_anomaly()
            label = 1
        writer.writerow([x, y, z, label])

print(f"CSV文件 '{filename}' 已生成，共 {num_rows} 行数据。")
