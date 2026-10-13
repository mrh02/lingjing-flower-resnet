# 基于手写ResNet的五类花卉图像分类系统

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
已完成数据统计与可视化、固定数据划分、手写 ResNet、十轮训练、
单张图片预测和测试集评估。第 9 轮模型的验证准确率最高，为 70.7%；
该模型的测试准确率为 69.8%（388/556）。后续完善项目说明、个人总结和演示材料。
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

## 训练与接续训练

首次三轮训练：

```powershell
.\.venv\Scripts\python.exe train.py
```

从最初保存的第 3 轮最佳模型接续到第 10 轮：

```powershell
.\.venv\Scripts\python.exe resume_train.py --epochs 10 --lr 0.0003
```

接续训练恢复模型和 Adam 优化器，并覆盖学习率；它复用原有数据划分，
只使用训练集和验证集。结果保存到终端显示的新目录，包含历史记录、
最佳模型 `best_model.pth` 和最后一轮模型 `last_model.pth`。
这里的 `--epochs 10` 是总轮次，默认从第 4 轮开始，新训练 7 轮。
从其他检查点恢复时用 `--checkpoint "模型路径"`；`--dry-run` 只检查恢复。

绘制接续训练曲线（将目录替换为终端显示的目录）：

```powershell
.\.venv\Scripts\python.exe plot_training.py "outputs/本次训练目录"
```

## 单张图片实际预测

使用本次选定的第 9 轮模型预测一张 JPG/PNG 花卉照片：

```powershell
.\.venv\Scripts\python.exe predict.py "C:\Users\moruohan\Desktop\rose.jpg" --checkpoint "outputs/train_20261012_104629_645720_resume/best_model.pth"
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

## 测试集评估与结果分析

固定划分为训练集 2567 张、验证集 547 张、测试集 556 张。
训练与模型选择使用训练集和验证集，模型选定后才评估测试集。
首次三轮使用 Adam，学习率 0.001；第 4～10 轮恢复模型及优化器，
将学习率改为 0.0003，保持数据划分不变。

| 指标 | 结果 |
| --- | --- |
| 首次三轮最佳验证准确率 | 57.0% |
| 十轮中的最佳模型 | 第 9 轮 |
| 最佳验证准确率 | 70.7% |
| 第 10 轮验证准确率 | 68.9% |
| 第 9 轮模型测试准确率 | 69.8%（388/556） |
| 第 9 轮模型测试损失 | 0.8275 |

验证准确率提高 13.7 个百分点。此过程同时增加了训练轮次并降低学习率，
不能把提升单独归因于学习率变化。测试准确率与验证准确率接近，
但一次划分的结果不足以证明模型在所有新照片上都能保持相同表现。

| 类别 | 测试图片数 | 正确数 | 召回率 |
| --- | ---: | ---: | ---: |
| 雏菊 daisy | 96 | 79 | 82.3% |
| 蒲公英 dandelion | 136 | 105 | 77.2% |
| 玫瑰 roses | 97 | 52 | 53.6% |
| 向日葵 sunflowers | 106 | 87 | 82.1% |
| 郁金香 tulips | 121 | 65 | 53.7% |

召回率表示该类别的测试图片中被正确识别的比例。
主要错误包括 23 张郁金香被判为玫瑰、19 张玫瑰被判为蒲公英、
17 张郁金香被判为雏菊。错误原因需结合具体图片分析，不能仅凭统计确定。
模型对玫瑰和郁金香的识别仍需改进。测试集用于最终报告；
后续如继续调整训练，应依据验证集，避免反复根据这份测试结果选择模型。

运行评估：

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint "outputs/train_20261012_104629_645720_resume/best_model.pth"
```

运行目录会因训练时间而不同，复现时请替换为自己保存的模型路径。
评估时使用 `model.eval()` 和 `torch.inference_mode()`，不更新参数。
结果保存在所选模型目录下的 `evaluation/`：

- `test_metrics.json`：整体指标、模型轮次及数据划分指纹。
- `per_class_metrics.csv`：每类图片数量、正确数量和召回率。
- `test_predictions.csv`：每张图片的真实类别、预测类别和模型分数。
- `confusion_matrix.csv` / `confusion_matrix.png`：混淆矩阵，行是真实类别，列是预测类别。

![十轮训练曲线](outputs/train_20261012_104629_645720_resume/training_curves.png)

![测试集混淆矩阵](outputs/train_20261012_104629_645720_resume/evaluation/confusion_matrix.png)

## 错分图片观察

通过 `review_errors.py` 读取已有的测试预测清单，不重新运行模型。
选择错误数量最多的三种“真实类别 → 预测类别”组合，
每种按清单顺序展示前两张，共六张。这是说明性样例，不是随机抽样。

```powershell
.\.venv\Scripts\python.exe review_errors.py "outputs/train_20261012_104629_645720_resume/evaluation"
```

![错分样例](outputs/train_20261012_104629_645720_resume/evaluation/error_examples.png)

对样例的观察：

- 图片包含花朵特写、花园远景和花束，拍摄距离及背景差异较大。
- 一张标为郁金香的图片主要呈现天空和风车，花卉区域很小，
  提示部分样本需要进一步检查目标是否清晰、标签是否适合整图分类。
  仅凭该图不能确定标签错误，也不能确定模型使用了哪些特征。
- 该图片被预测为雏菊，softmax 分数为 82.1%，但预测与数据集标签不符。
  因此，较高的模型分数不能作为预测正确的保证。

以上观察用于说明局限，没有删除测试样本或改动测试标签；
测试准确率仍按全部 556 张图片计算。背景、目标大小等因素对模型的具体影响
需要另行设计验证实验，不能直接作为本次错误的已证实原因。
