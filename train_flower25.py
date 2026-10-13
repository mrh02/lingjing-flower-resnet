"""训练二十五类版本，使用手写 ResNet；保留原五类实验。"""

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

from flower_dataset import FlowerDataset
from resnet import FlowerResNet


ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, total = 0.0, 0
    counts = torch.zeros(num_classes, dtype=torch.long)
    correct_counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            counts += torch.bincount(labels.cpu(), minlength=num_classes)
            correct_counts += torch.bincount(
                labels[predictions == labels].cpu(), minlength=num_classes,
            )
            loss_sum += loss.item() * len(labels)
            total += len(labels)
            if training and step % 50 == 0:
                print(f"  训练批次：{step}/{len(loader)}", flush=True)
    if total == 0 or (not training and (counts == 0).any()):
        raise ValueError("数据集为空或验证集缺少某个类别")
    recall = correct_counts.float() / counts.clamp(min=1)
    return loss_sum / total, correct_counts.sum().item() / total, recall.mean().item()


def main():
    parser = argparse.ArgumentParser(description="训练新增二十类后的花卉模型")
    parser.add_argument("--data-info", type=Path, default=ROOT / "outputs" / "flower25" / "dataset_info.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", type=Path, help="从本版本的 last_model.pth 接续")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和前向计算，不开始训练")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("轮次和学习率必须大于零")
    info_path = args.data_info.resolve()
    info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest_path = info_path.parent / "split_manifest.csv"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != info["split_sha256"]:
        parser.error("分组清单已变化，请重新准备数据")
    classes = info["class_names"]
    if len(classes) != 25 or len(set(classes)) != 25:
        parser.error("当前脚本需要 25 个不同类别")
    manifest = pd.read_csv(manifest_path)
    pairs = manifest[["class", "label"]].drop_duplicates()
    if len(pairs) != 25 or dict(zip(pairs["class"], pairs["label"])) != {
        name: index for index, name in enumerate(classes)
    }:
        parser.error("分组清单的类别编号不匹配")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
    ])
    train_dataset = FlowerDataset("train", train_transform, manifest_path)
    val_dataset = FlowerDataset("val", evaluation_transform, manifest_path)
    counts = train_dataset.rows["label"].value_counts().reindex(range(25), fill_value=0)
    if (counts == 0).any():
        parser.error("训练集缺少某个类别")
    # 每类被抽到的总机会相同；仅训练集重复抽样，验证集仍逐张检查。
    weights = train_dataset.rows["label"].map(lambda label: 1.0 / counts[label]).tolist()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    model = FlowerResNet(num_classes=25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    start_epoch, best_macro, selected_accuracy, history = 1, -1.0, 0.0, []
    previous_best = None
    if args.resume:
        checkpoint_path = args.resume.resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["class_names"] != classes or checkpoint["split_sha256"] != digest:
            parser.error("恢复模型的类别或数据划分不匹配")
        if checkpoint.get("selection_metric") != "val_macro_recall":
            parser.error("请使用二十五类版本的检查点")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint["history"]
        previous_best = checkpoint_path.parent / "best_model.pth"
        best = torch.load(previous_best, map_location="cpu", weights_only=True)
        if best["epoch"] >= start_epoch:
            parser.error("配套最佳模型来自更晚轮次，请从最后一轮模型恢复")
        best_macro, selected_accuracy = best["best_val_macro_recall"], best["best_val_accuracy"]
    if args.epochs < start_epoch:
        parser.error("目标总轮次必须大于已完成轮次")
    print("运行设备：", device)
    print(f"类别数量：25；训练图片：{len(train_dataset)}；验证图片：{len(val_dataset)}")
    print("选模依据：验证集各类别召回率的平均值，每一类权重相同。")
    print("模型参数数量：", sum(p.numel() for p in model.parameters()))
    if args.dry_run:
        images, labels = next(iter(val_loader))
        model.eval()
        with torch.inference_mode():
            logits = model(images.to(device))
        if logits.shape != (len(labels), 25) or not torch.isfinite(logits).all():
            raise ValueError("模型前向计算异常")
        print("输入形状：", images.shape)
        print("输出形状：", logits.shape)
        print("检查通过，尚未开始训练。")
        return

    run_dir = ROOT / "outputs" / datetime.now().strftime("train25_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    shutil.copyfile(manifest_path, run_dir / "split_manifest.csv")
    shutil.copyfile(info_path, run_dir / "dataset_info.json")
    if previous_best:
        shutil.copyfile(previous_best, run_dir / "best_model.pth")
    print("结果目录：", run_dir, flush=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n开始第 {epoch}/{args.epochs} 轮", flush=True)
        train_loss, train_accuracy, _ = run_epoch(model, train_loader, criterion, device, 25, optimizer)
        val_loss, val_accuracy, val_macro = run_epoch(model, val_loader, criterion, device, 25)
        print(f"训练损失：{train_loss:.4f}，训练抽样准确率：{train_accuracy:.1%}\n"
              f"验证损失：{val_loss:.4f}，验证准确率：{val_accuracy:.1%}\n"
              f"验证平均类别召回率：{val_macro:.1%}", flush=True)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy, "val_macro_recall": val_macro,
        })
        improved = val_macro > best_macro
        if improved:
            best_macro, selected_accuracy = val_macro, val_accuracy
        state = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "class_names": classes, "class_names_zh": info["class_names_zh"],
            "image_size": 128, "split_sha256": digest,
            "selection_metric": "val_macro_recall", "best_val_macro_recall": best_macro,
            "best_val_accuracy": selected_accuracy, "history": list(history),
            "training_sampler": "inverse_class_count_with_replacement",
            "initialization": "random" if not args.resume else str(args.resume.resolve()),
        }
        torch.save(state, run_dir / "last_model.pth")
        if improved:
            torch.save(state, run_dir / "best_model.pth")
            print("已保存新的最佳模型（按平均类别召回率选择）", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"\n训练完成，最佳验证平均类别召回率：{best_macro:.1%}")
    print(f"该模型的验证准确率：{selected_accuracy:.1%}")
    print("最佳模型：", run_dir / "best_model.pth")
    print("最后一轮模型：", run_dir / "last_model.pth")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"训练失败：{error}")
