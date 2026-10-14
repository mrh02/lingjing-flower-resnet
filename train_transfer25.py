"""独立的预训练 ResNet18 对照实验；手写 ResNet 仍是项目基线。"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18

from flower_dataset import FlowerDataset


ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def run_epoch(model, loader, criterion, device, classes, optimizer=None):
    training = optimizer is not None
    # 冻结的骨干网络及其 BatchNorm 始终使用预训练统计量。
    model.eval()
    total, loss_sum = 0, 0.0
    counts = torch.zeros(len(classes), dtype=torch.long)
    correct = torch.zeros(len(classes), dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            counts += torch.bincount(labels.cpu(), minlength=len(classes))
            correct += torch.bincount(
                labels[predicted == labels].cpu(), minlength=len(classes))
            loss_sum += loss.item() * len(labels)
            total += len(labels)
    if total == 0 or (not training and (counts == 0).any()):
        raise ValueError("训练或验证图片为空，或验证集缺少类别")
    recalls = correct.float() / counts.clamp(min=1)
    return loss_sum / total, correct.sum().item() / total, recalls.mean().item(), recalls.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-info", type=Path,
                        default=ROOT / "outputs/flower25/dataset_info.json")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "data/pretrained/resnet18-f37072fd.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or not 0 <= args.sampler_power <= 1:
        parser.error("训练轮次与学习率须为正数；抽样指数须在 0～1 之间")
    info_path, weights_path = args.data_info.resolve(), args.weights.resolve()
    manifest_path = info_path.parent / "split_manifest.csv"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != info["split_sha256"]:
        parser.error("数据清单和元数据的校验值不一致")
    classes = info["class_names"]
    if len(classes) != 25:
        parser.error("这项对照实验需要 25 个类别")
    weight_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if not weight_hash.startswith("f37072fd"):
        parser.error("预训练权重校验值与官方文件名不匹配")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_set = FlowerDataset("train", train_transform, manifest_path)
    val_set = FlowerDataset("val", val_transform, manifest_path)
    counts = train_set.rows["label"].value_counts().reindex(range(25), fill_value=0)
    if (counts == 0).any():
        parser.error("训练集缺少类别")
    weights = train_set.rows["label"].map(
        lambda label: float(counts[label]) ** (-args.sampler_power)).tolist()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set),
                                    replacement=True)
    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    print("运行设备：", device, flush=True)
    print(f"预训练权重 SHA256：{weight_hash}", flush=True)
    print(f"类别：25；训练图片：{len(train_set)}；验证图片：{len(val_set)}", flush=True)
    print("只更新最后的分类层；最佳模型按验证集平均类别召回率选择。", flush=True)
    if args.dry_run:
        images, labels = next(iter(val_loader))
        with torch.inference_mode():
            logits = model(images.to(device))
        if logits.shape != (len(labels), 25) or not torch.isfinite(logits).all():
            raise ValueError("模型输出形状或数值异常")
        print("输入形状：", images.shape)
        print("输出形状：", logits.shape)
        print("检查通过，尚未开始训练。")
        return

    run_dir = ROOT / "outputs" / datetime.now().strftime("transfer25_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    shutil.copyfile(manifest_path, run_dir / "split_manifest.csv")
    shutil.copyfile(info_path, run_dir / "dataset_info.json")
    print("结果目录：", run_dir, flush=True)
    best_macro, selected_accuracy, history = -1.0, 0.0, []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            model, train_loader, criterion, device, classes, optimizer)
        val_loss, val_accuracy, val_macro, val_recalls = run_epoch(
            model, val_loader, criterion, device, classes)
        roses_recall = val_recalls[classes.index("roses")]
        dandelion_recall = val_recalls[classes.index("dandelion")]
        carnation_recall = val_recalls[classes.index("carnations")]
        print(f"第 {epoch}/{args.epochs} 轮：训练损失 {train_loss:.4f}、抽样准确率 {train_accuracy:.1%}；"
              f"验证损失 {val_loss:.4f}、准确率 {val_accuracy:.1%}、平均类别召回率 {val_macro:.1%}；"
              f"玫瑰 {roses_recall:.1%}、蒲公英 {dandelion_recall:.1%}、康乃馨 {carnation_recall:.1%}",
              flush=True)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
            "val_macro_recall": val_macro,
            "val_roses_recall": roses_recall,
            "val_dandelion_recall": dandelion_recall,
            "val_carnations_recall": carnation_recall,
        })
        improved = val_macro > best_macro
        if improved:
            best_macro, selected_accuracy = val_macro, val_accuracy
        state = {
            "architecture": "torchvision_resnet18_pretrained_head",
            "model_state_dict": model.state_dict(), "epoch": epoch,
            "class_names": classes, "class_names_zh": info["class_names_zh"],
            "image_size": 224, "normalization_mean": MEAN, "normalization_std": STD,
            "split_sha256": manifest_hash, "pretrained_weights_sha256": weight_hash,
            "pretrained_weights_url": WEIGHTS_URL,
            "sampler_power": args.sampler_power,
            "selection_metric": "val_macro_recall",
            "best_val_macro_recall": best_macro,
            "best_val_accuracy": selected_accuracy,
            "history": list(history),
        }
        torch.save(state, run_dir / "last_model.pth")
        if improved:
            torch.save(state, run_dir / "best_model.pth")
            print("  已保存新的最佳模型", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"最佳验证平均类别召回率：{best_macro:.1%}；验证准确率：{selected_accuracy:.1%}")
    print("最佳模型：", run_dir / "best_model.pth")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"预训练对照失败：{error}")
