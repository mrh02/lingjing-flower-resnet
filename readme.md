# 花卉图像分类：从 5 类到 25 类

本项目用 PyTorch 实现花卉图片分类。先用手写 ResNet 完成 5 类分类，再扩展到 25 类；另训练一个使用预训练 ResNet18 图像特征的对照模型。**25 类手写 ResNet 是项目的主要模型**，对照模型用于比较两种训练方式，不是花卉类别的“标准答案”。

## 模型与结果

| 模型 | 构建方式 | 测试集准确率 |
| --- | --- | ---: |
| 5 类手写 ResNet | 从随机参数开始训练，建立完整流程 | 69.8%（388/556） |
| 25 类手写 ResNet，初版 | 扩大输出类别，从随机参数开始训练 | 61.5%（566/920） |
| **25 类手写 ResNet，当前版** | 调整类别抽样并接续训练，按验证集平均类别召回率选中第 33 轮 | **83.2%（765/920）** |
| 25 类预训练 ResNet18 对照 | 使用已有图像特征，训练 25 类分类层 | 88.3%（812/920） |

当前手写模型在 25 类测试集上的平均类别召回率为 **84.3%**。同一模型在测试集上识别正确玫瑰 63/97 张、康乃馨 4/9 张；类别之间仍会混淆。测试集和部分额外照片在开发过程中已被查看，这些数字不能视为全新盲测成绩。

## 数据与方法

- 前 5 类：雏菊、蒲公英、玫瑰、向日葵、郁金香，来自 [TensorFlow 花卉示例数据](https://www.tensorflow.org/tutorials/load_data/images)，共 3670 张。
- 新增 20 类：包括荷花、百合、兰花、康乃馨等，选自 [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)；具体类别和品种范围见 [flower25_config.py](flower25_config.py)。
- 25 类数据按图片像素去重并排除标签冲突记录后，共 5963 张：训练 4161、验证 882、测试 920。原 5 类保留原划分；新增类别按固定随机种子划分。本项目未采用 Oxford 官方分组。
- 手写模型见 [resnet.py](resnet.py)。图片缩放为 `128×128`，通过卷积和残差块提取特征，最后输出各类别分数。训练时对类别数量不均衡进行抽样；每轮用验证集选择模型。

部分名称仅覆盖选取的来源品种，例如“百合”对应卷丹百合，“兰花”合并三个来源类别，不代表能识别该花卉的所有品种。

## 安装与准备数据

以下命令在项目根目录的 **PowerShell** 中运行。实验环境为 Windows、Python 3.10、PyTorch 2.11.0+cu128；其他设备需安装相适配的 PyTorch 版本。

```powershell
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
New-Item -ItemType Directory -Force data
curl.exe -L --fail --output data/flower_photos.tgz https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar.exe -xzf data/flower_photos.tgz -C data
& .\.venv\Scripts\python.exe .\prepare_data.py
```

准备扩展数据：

```powershell
New-Item -ItemType Directory -Force data/flowers102
curl.exe -L --fail --output data/flowers102/imagelabels.mat https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
curl.exe -L --fail --output data/flowers102/categories.html https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html
curl.exe -L --fail --output data/flowers102/README.txt https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt
curl.exe -L --fail --output data/flowers102/102flowers.tgz https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
& .\.venv\Scripts\python.exe .\prepare_flower25.py
```

## 训练与使用

复现当前的 25 类手写模型，需要连续训练三个阶段。第二、三条命令中的目录，改为上一阶段终端显示的实际训练目录；`--epochs` 表示累计轮数。

```powershell
& .\.venv\Scripts\python.exe .\train_flower25.py --epochs 20 --lr 0.001 --sampler-power 0.5
& .\.venv\Scripts\python.exe .\train_flower25.py --resume "outputs\第一阶段训练目录\last_model.pth" --epochs 30 --lr 0.0003 --sampler-power 0.5
& .\.venv\Scripts\python.exe .\train_flower25.py --resume "outputs\第二阶段训练目录\last_model.pth" --epochs 35 --lr 0.0001 --sampler-power 0.5
```

每次训练会生成独立的 `outputs/train25_时间戳/` 目录，其中 `best_model.pth` 是按验证集选出的模型。训练完成后，将下面的路径换成实际目录和图片路径：

```powershell
& .\.venv\Scripts\python.exe .\predict.py "C:\图片目录\flower.jpg" --checkpoint "outputs\第三阶段训练目录\best_model.pth"
& .\.venv\Scripts\python.exe .\evaluate.py --checkpoint "outputs\第三阶段训练目录\best_model.pth"
```

预测结果和展示图保存在 `outputs/predictions/`。`predict.py` 不指定 `--checkpoint` 时使用旧的 5 类模型，因此使用 25 类模型时必须指定路径。也可以运行 `train.py` 和 `resume_train.py` 复现 5 类基线；预训练对照模型的训练入口是 [train_transfer25.py](train_transfer25.py)。

## 如何理解结果

模型会在已支持的类别中给出一个分类，尚不能可靠判断图片是否属于这 25 类之外。终端中的百分比是**这一张图片在 25 类之间的模型分数**，不是模型的总体正确率；高分仍可能判错。类别样本较少时，单类结果波动尤其大，例如当前测试集的康乃馨只有 9 张。

项目还提供 [explain_prediction.py](explain_prediction.py) 的热力图和遮挡检查，用于观察哪些区域影响预测，但它们不能证明模型只依据某一种可命名的花朵特征。训练过程、错分案例、数据审查与各阶段结果见 [详细实验记录](docs/experiment_log.md)。
