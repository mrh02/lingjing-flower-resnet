from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms

project_dir = Path(__file__).resolve().parent

# 读取分组清单，选出第一张训练图片
df = pd.read_csv(project_dir / "outputs" / "split_manifest.csv")
train_df = df[df["split"] == "train"]
sample = train_df.iloc[0]

image_path = project_dir / sample["path"]

# 统一为 RGB 三通道图片
with Image.open(image_path) as source:
    image = source.convert("RGB")

print("图片类别：", sample["class"])
print("类别编号：", sample["label"])
print("原始尺寸（宽、高）：", image.size)

# 统一图片尺寸，再转换为张量
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

tensor = transform(image)

print("张量形状：", tensor.shape)
print("数据类型：", tensor.dtype)
print("最小像素值：", tensor.min().item())
print("最大像素值：", tensor.max().item())