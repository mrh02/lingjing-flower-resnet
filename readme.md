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
已完成初步数据统计与可视化，模型搭建和训练尚未完成。
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