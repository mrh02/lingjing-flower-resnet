"""恢复已保存的模型和优化器，继续训练到指定轮次。"""

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from flower_dataset import FlowerDataset
from resnet import FlowerResNet
from train import run_epoch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "outputs" / "train_20261012_102328" / "best_model.pth"
)


def main():
    parser = argparse.ArgumentParser(description="从保存的模型接续训练")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10, help="训练到的总轮次")
    parser.add_argument("--lr", type=float, default=0.0003, help="接续训练的学习率")
    parser.add_argument("--dry-run", action="store_true", help="只检查恢复，不开始训练")
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error("学习率必须大于 0")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"找不到模型：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    saved_epoch = int(checkpoint["epoch"])
    start_epoch = saved_epoch + 1
    if args.epochs < start_epoch:
        parser.error(f"模型已完成第 {saved_epoch} 轮，目标轮次必须大于它")

    manifest = PROJECT_DIR / "outputs" / "split_manifest.csv"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if checkpoint.get("split_sha256", manifest_hash) != manifest_hash:
        parser.error("分组清单已变化，请使用与该模型相同的数据划分")

    image_size = int(checkpoint["image_size"])
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = FlowerDataset("train", train_transform)
    val_dataset = FlowerDataset("val", val_transform)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        parser.error("训练集和验证集都必须包含图片")

    class_names = checkpoint["class_names"]
    expected_labels = {name: label for label, name in enumerate(class_names)}
    for dataset in (train_dataset, val_dataset):
        pairs = dataset.rows[["class", "label"]].drop_duplicates()
        actual_labels = dict(zip(pairs["class"], pairs["label"]))
        if actual_labels != expected_labels or len(pairs) != len(class_names):
            parser.error("数据清单的类别编号与模型不一致")

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=0,
    )

    model = FlowerResNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 恢复优化器后，明确覆盖它原来保存的学习率。
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    history = checkpoint.get("history")
    if history is None:
        history_path = checkpoint_path.parent / "history.csv"
        if not history_path.is_file():
            parser.error(f"找不到已有训练记录：{history_path}")
        old_history = pd.read_csv(history_path)
        history = old_history[old_history["epoch"] <= saved_epoch].to_dict("records")
    else:
        history = [row for row in history if row["epoch"] <= saved_epoch]

    # 从最后一轮恢复时，也保留该次训练中此前的最佳模型。
    best_path = checkpoint_path.parent / "best_model.pth"
    if checkpoint_path.name == "best_model.pth":
        best_path = checkpoint_path
    if not best_path.is_file():
        parser.error("需要保留与该检查点配套的 best_model.pth")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if int(best_checkpoint["epoch"]) > saved_epoch:
        parser.error("最佳模型来自更晚的轮次，请从它或配套的最后一轮模型恢复")
    best_val_accuracy = float(best_checkpoint["best_val_accuracy"])

    print("运行设备：", device)
    print("恢复模型：", checkpoint_path)
    print(f"继续训练第 {start_epoch}～{args.epochs} 轮")
    print("接续学习率：", optimizer.param_groups[0]["lr"])
    print(f"已有最佳验证准确率：{best_val_accuracy:.1%}")
    if args.dry_run:
        print("恢复检查通过，尚未开始训练")
        return

    run_dir = (
        PROJECT_DIR / "outputs"
        / datetime.now().strftime("train_%Y%m%d_%H%M%S_%f_resume")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, run_dir / "best_model.pth")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print("结果目录：", run_dir, flush=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n开始第 {epoch}/{args.epochs} 轮", flush=True)
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer,
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        print(
            f"训练损失：{train_loss:.4f}，训练准确率：{train_accuracy:.1%}\n"
            f"验证损失：{val_loss:.4f}，验证准确率：{val_accuracy:.1%}",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        improved = val_accuracy > best_val_accuracy
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "class_names": class_names,
            "image_size": image_size,
            "best_val_accuracy": best_val_accuracy,
            "history": list(history),
            "split_sha256": manifest_hash,
            "resume_from": str(checkpoint_path),
        }
        torch.save(state, run_dir / "last_model.pth")
        if improved:
            torch.save(state, run_dir / "best_model.pth")
            print("已保存新的最佳模型", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    print(f"\n训练完成，最佳验证准确率：{best_val_accuracy:.1%}")
    print("最佳模型：", run_dir / "best_model.pth")
    print("最后一轮模型：", run_dir / "last_model.pth")


if __name__ == "__main__":
    main()
