from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FlowerDataset(Dataset):
    def __init__(self, split, transform):
        self.project_dir = Path(__file__).resolve().parent

        manifest = self.project_dir / "outputs" / "split_manifest.csv"
        df = pd.read_csv(manifest)

        self.rows = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.project_dir / row["path"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image = self.transform(image)
        label = int(row["label"])

        return image, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = FlowerDataset("train", transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("训练图片总数：", len(train_dataset))
    print("一批图片的形状：", images.shape)
    print("一批标签的形状：", labels.shape)
    print("标签数据类型：", labels.dtype)
    print("这一批的类别编号：", labels.tolist())