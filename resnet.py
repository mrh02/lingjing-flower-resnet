import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主路径：两次卷积
        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # 形状相同时，跳跃路径直接传递输入
        self.shortcut = nn.Identity()

        # 形状改变时，用 1×1 卷积匹配尺寸
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        main_output = self.main_path(x)
        shortcut_output = self.shortcut(x)

        return torch.relu(main_output + shortcut_output)

class FlowerResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # 将 RGB 图片转换成 32 通道特征，并缩小尺寸
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3,
                stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)

if __name__ == "__main__":
    from flower_dataset import FlowerDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = FlowerDataset("train", transform)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0,
    )

    images, labels = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FlowerResNet(num_classes=5).to(device)
    model.eval()

    with torch.no_grad():
        scores = model(images.to(device))

    print("运行设备：", device)
    print("输入形状：", images.shape)
    print("输出形状：", scores.shape)
    print("标签形状：", labels.shape)

    parameter_count = sum(p.numel() for p in model.parameters())
    print("模型参数数量：", parameter_count)