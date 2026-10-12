from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from flower_dataset import FlowerDataset
from resnet import FlowerResNet


def run_epoch(model, loader, criterion, device, optimizer=None):
    # 提供优化器就是训练，否则就是验证
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 验证时关闭梯度计算
    with torch.set_grad_enabled(is_training):
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            scores = model(images)
            loss = criterion(scores, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = scores.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            if is_training and step % 50 == 0:
                print(f"  训练批次：{step}/{len(loader)}", flush=True)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    project_dir = Path(__file__).resolve().parent

    # 每次运行使用单独的结果目录
    run_dir = (
        project_dir / "outputs"
        / datetime.now().strftime("train_%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = FlowerDataset("train", train_transform)
    val_dataset = FlowerDataset("val", val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=16,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16,
        shuffle=False, num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FlowerResNet(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001,
    )

    class_names = (
        train_dataset.rows[["label", "class"]]
        .drop_duplicates()
        .sort_values("label")["class"]
        .tolist()
    )

    epochs = 3
    best_val_accuracy = -1.0
    history = []

    print("运行设备：", device)
    print("结果目录：", run_dir)

    for epoch in range(1, epochs + 1):
        print(f"\n开始第 {epoch}/{epochs} 轮", flush=True)

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer,
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device,
        )

        print(
            f"训练损失：{train_loss:.4f}，"
            f"训练准确率：{train_accuracy:.1%}\n"
            f"验证损失：{val_loss:.4f}，"
            f"验证准确率：{val_accuracy:.1%}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        pd.DataFrame(history).to_csv(
            run_dir / "history.csv", index=False,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "class_names": class_names,
                "image_size": 128,
                "best_val_accuracy": best_val_accuracy,
            }, run_dir / "best_model.pth")

            print("已保存目前验证效果最好的模型", flush=True)

    print(f"\n训练完成，最佳验证准确率：{best_val_accuracy:.1%}")