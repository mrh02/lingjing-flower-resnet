from pathlib import Path

# 根据当前代码文件的位置，找到项目文件夹
project_dir = Path(__file__).resolve().parent

print("项目位置：", project_dir)

# 逐个查看项目文件夹中的内容
for item in project_dir.iterdir():
    if item.is_dir():
        print("文件夹：", item.name)
    else:
        print("文件：", item.name)