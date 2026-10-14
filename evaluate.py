"""用固定的测试集评估已选定的模型，不训练或更新参数。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from flower_dataset import FlowerDataset
from resnet import FlowerResNet


PROJECT_DIR = Path(__file__).resolve().parent


def evaluate(checkpoint_path, output_dir, device, manifest_override=None):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True,
    )
    classes = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    # 新实验随模型保存独立清单，旧五类模型仍使用原清单。
    manifest_path = manifest_override or (
        checkpoint_path.parent / "split_manifest.csv"
        if (checkpoint_path.parent / "split_manifest.csv").is_file()
        else PROJECT_DIR / "outputs" / "split_manifest.csv"
    )
    split_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if checkpoint.get("split_sha256", split_hash) != split_hash:
        raise ValueError("数据划分清单与训练时不同，请恢复原清单")

    manifest = pd.read_csv(manifest_path)
    if manifest["path"].duplicated().any():
        raise ValueError("划分清单有重复图片路径，请检查数据划分")
    mapping = manifest[["class", "label"]].drop_duplicates()
    expected = {name: index for index, name in enumerate(classes)}
    if len(mapping) != len(classes) or dict(
        zip(mapping["class"], mapping["label"])
    ) != expected:
        raise ValueError("清单的类别编号与模型不一致")

    # 测试时使用模型在验证阶段采用的相同预处理。
    architecture = checkpoint.get("architecture", "handwritten_flower_resnet")
    if architecture == "torchvision_resnet18_pretrained_head":
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["normalization_mean"],
                                 checkpoint["normalization_std"]),
        ])
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif architecture == "handwritten_flower_resnet":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        model = FlowerResNet(num_classes=len(classes))
    else:
        raise ValueError(f"未知模型结构：{architecture}")
    dataset = FlowerDataset("test", transform, manifest_path=manifest_path)
    if len(dataset) == 0:
        raise ValueError("测试集为空")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # BatchNorm 使用保存的统计量。
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    true_labels, predicted_labels, scores = [], [], []

    print("运行设备：", device)
    print("模型轮次：", checkpoint["epoch"])
    print("测试图片总数：", len(dataset))
    # 关闭梯度计算；没有 optimizer，不会更新模型参数。
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            probabilities = torch.softmax(logits, dim=1)
            top_scores, predictions = probabilities.max(dim=1)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            scores.extend(top_scores.cpu().tolist())

    # 行表示真实类别，列表示预测类别。
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    correct = int(np.trace(confusion))
    accuracy = correct / len(dataset)
    per_class = pd.DataFrame({
        "class": classes,
        "test_count": confusion.sum(axis=1),
        "correct_count": np.diag(confusion),
        "recall": np.diag(confusion) / confusion.sum(axis=1).clip(min=1),
    })
    predictions_table = dataset.rows[["path", "class", "label"]].copy()
    predictions_table = predictions_table.rename(columns={"class": "true_class"})
    predictions_table["predicted_class"] = [classes[i] for i in predicted_labels]
    predictions_table["correct"] = np.array(true_labels) == predicted_labels
    predictions_table["softmax_score"] = scores

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": architecture,
        "best_val_accuracy": float(checkpoint["best_val_accuracy"]),
        "split_sha256": split_hash,
        "test_count": len(dataset),
        "correct_count": correct,
        "test_accuracy": accuracy,
        "test_loss": total_loss / len(dataset),
        "test_macro_recall": float(per_class["recall"].mean()),
        "selection_metric": checkpoint.get("selection_metric", "val_accuracy"),
        "class_names": classes,
        "confusion_matrix": confusion.tolist(),
        "matrix_axes": "rows=true, columns=predicted",
        "score_note": "softmax_score 是模型分数，不是实测准确率。",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    predictions_table.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(confusion, index=classes, columns=classes).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_class",
    )

    side = max(7, len(classes) * 0.46 + 2)
    fig, ax = plt.subplots(figsize=(side, side - 1))
    chart = ax.imshow(confusion, cmap="Blues", vmin=0)
    fig.colorbar(chart, ax=ax, label="Number of images")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Test confusion matrix | Accuracy: {accuracy:.1%}")
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center",
                    color="white" if confusion[row, col] > confusion.max() / 2 else "black",
                    fontsize=8 if len(classes) > 10 else 11)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"测试损失：{summary['test_loss']:.4f}")
    print(f"测试准确率：{accuracy:.1%}（{correct}/{len(dataset)}）")
    print(f"各类别召回率的平均值：{summary['test_macro_recall']:.1%}")
    print("各类别的召回率（该类别测试图片中预测正确的比例）：")
    for name, count, class_correct, recall in per_class.itertuples(index=False, name=None):
        print(f"  {name}：{class_correct}/{count}，{recall:.1%}")
    print("评估结果保存到：", output_dir)


def main():
    parser = argparse.ArgumentParser(description="评估固定测试集上的分类表现")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="覆盖模型配套的分组清单")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"找不到模型：{checkpoint_path}")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，可用 --device cpu")
    output_dir = (args.output_dir or checkpoint_path.parent / "evaluation").resolve()
    try:
        evaluate(checkpoint_path, output_dir, torch.device(device_name),
                 args.manifest.resolve() if args.manifest else None)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"评估失败：{error}\n")


if __name__ == "__main__":
    main()
