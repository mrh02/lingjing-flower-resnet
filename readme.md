# 花卉图像分类项目

灵境竞赛组 AI 方向第一阶段考核项目。

## 项目目标
使用 Python 完成花卉数据分析，并使用 PyTorch 手写 ResNet，
训练模型并对新图片进行分类预测。

## 当前进度
- 已建立 Python 3.10 独立环境。
- 已安装数据分析库及 PyTorch。
- 已验证 RTX 5060 可以执行 GPU 计算。
- 已完成字典、循环、函数和条件判断练习。
- 已下载并统计五类花卉图片，共 3670 张。
- 已使用 Pandas 保存类别统计表。
- 已使用 Matplotlib 绘制类别数量柱状图。
- 已使用 NumPy 计算平均数量和最大、最小数量比。

## 开发状态
已完成数据统计与可视化、数据划分、手写 ResNet 和首次三轮训练。
首次训练的最佳验证准确率为 57.0%，测试集尚未评估。
已加入单张新图片预测功能，后续继续完善训练和评估。
## 数据准备

数据来源：https://www.tensorflow.org/tutorials/load_data/images

在项目根目录的 PowerShell 终端依次执行：

```powershell
New-Item -ItemType Directory -Force data
curl.exe -L --fail --output data/flower_photos.tgz https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar.exe -xzf data/flower_photos.tgz -C data
```

图片授权和作者信息见数据中的 LICENSE.txt。

## 运行数据分析

```powershell
.\.venv\Scripts\python.exe analyze_data.py
```

结果保存在 outputs 文件夹：
- class_counts.csv：各类别的数量与占比。
- class_counts.png：类别数量柱状图。

## 单张图片实际预测

使用已训练模型预测一张 JPG/PNG 花卉照片：

```powershell
.\.venv\Scripts\python.exe predict.py "C:\Users\moruohan\Desktop\rose.jpg"
```

将图片路径替换为自己的文件路径。默认模型为
`outputs/train_20261012_102328/best_model.pth`，模型权重未提交到 Git。
需要先获得本地训练权重；已有其他权重时可通过参数指定：

```powershell
.\.venv\Scripts\python.exe predict.py "图片路径" --checkpoint "模型文件路径"
```

模型支持雏菊、蒲公英、玫瑰、向日葵、郁金香五类。
预测处理方式与验证一致：RGB、128×128、像素缩放到 0～1。
预测时使用评估模式，不更新参数，不需要提供图片的正确类别。

终端显示预测类别和五类的 softmax 分数；展示图及 JSON 记录保存在
`outputs/predictions/`。分数不是实测准确率，也不能用于确认照片属于
这五类之外的某个类别。额外照片用于实际使用演示，不计入验证准确率。
