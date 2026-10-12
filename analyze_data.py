from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "flower_photos"

class_counts = {}

for class_dir in sorted(data_dir.iterdir()):
    # 跳过 LICENSE.txt 等文件，只处理类别文件夹
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg"))
    class_counts[class_dir.name] = len(images)

total = sum(class_counts.values())

print(f"图片总数：{total}")

for name, count in class_counts.items():
    percentage = count / total * 100
    print(f"{name}：{count} 张，占比 {percentage:.1f}%")
    import pandas as pd

# 把字典整理成两列：类别、数量
df = pd.DataFrame({
    "class": list(class_counts.keys()),
    "count": list(class_counts.values()),
})

# 新增占比列
df["percentage"] = (df["count"] / total * 100).round(2)

print("\n类别统计表：")
print(df)

# 建立结果文件夹
output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存表格
csv_path = output_dir / "class_counts.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n表格已保存到：{csv_path}")
import matplotlib.pyplot as plt

# 创建画布
fig, ax = plt.subplots(figsize=(8, 5))

# 横轴是类别，柱子高度是图片数量
ax.bar(df["class"], df["count"], color="steelblue")

ax.set_title("Flower Images per Class")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")

# 在每根柱子上方标出数量
for i, count in enumerate(df["count"]):
    ax.text(i, count + 10, str(count), ha="center")

ax.set_ylim(0, df["count"].max() * 1.15)
fig.tight_layout()

chart_path = output_dir / "class_counts.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print(f"柱状图已保存到：{chart_path}")
import numpy as np

# 把数量这一列转换成 NumPy 数组
counts = df["count"].to_numpy()

mean_count = np.mean(counts)
min_count = np.min(counts)
max_count = np.max(counts)
ratio = max_count / min_count

print("\n类别数量分析：")
print(f"平均每类：{mean_count:.1f} 张")
print(f"最少的类别：{min_count} 张")
print(f"最多的类别：{max_count} 张")
print(f"最多与最少的数量比：{ratio:.2f}")