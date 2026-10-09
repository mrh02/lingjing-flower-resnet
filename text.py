# 字典：类别名称对应图片数量
class_counts = {
    "daisy": 20,
    "dandelion": 30,
    "roses": 40,
    "sunflowers": 25,
    "tulips": 10,
}


def count_images(counts):
    total = 0

    for name, count in counts.items():
        total += count

    return total


total = count_images(class_counts)

print(f"图片总数：{total}")

for name, count in class_counts.items():
    percentage = count / total * 100
    print(f"{name}：{count} 张，占比 {percentage:.1f}%")
for name, count in class_counts.items():
    if count < 20: 
        print(f"{name} 的图片较少，只有 {count} 张")
        import sys
print("当前 Python 路径：", sys.executable)