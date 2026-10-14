"""只检查固定验证集的错分；不更新模型，也不读取测试集。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from flower_dataset import FlowerDataset
from resnet import FlowerResNet


def check(checkpoint_path, output_dir, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = checkpoint["class_names"]
    manifest_path = checkpoint_path.parent / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if checkpoint.get("split_sha256", digest) != digest:
        raise ValueError("模型与数据划分清单不匹配")
    mapping = dict(manifest[["class", "label"]].drop_duplicates().itertuples(index=False, name=None))
    if mapping != {name: i for i, name in enumerate(classes)}:
        raise ValueError("模型与类别编号不匹配")
    if manifest["path"].duplicated().any():
        raise ValueError("划分清单有重复路径")
    architecture = checkpoint.get("architecture", "handwritten_flower_resnet")
    if architecture == "torchvision_resnet18_pretrained_head":
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(int(checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["normalization_mean"],
                                 checkpoint["normalization_std"]),
        ])
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif architecture == "handwritten_flower_resnet":
        transform = transforms.Compose([
            transforms.Resize((int(checkpoint["image_size"]),) * 2),
            transforms.ToTensor(),
        ])
        model = FlowerResNet(num_classes=len(classes))
    else:
        raise ValueError(f"未知模型结构：{architecture}")
    dataset = FlowerDataset("val", transform, manifest_path=manifest_path)
    if not len(dataset):
        raise ValueError("验证集为空")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predicted, top_scores = [], []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
            scores, labels = model(images.to(device)).softmax(dim=1).max(dim=1)
            predicted.extend(labels.cpu().tolist())
            top_scores.extend(scores.cpu().tolist())
    true = dataset.rows["label"].to_numpy(dtype=np.int64)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (true, np.asarray(predicted)), 1)
    counts = confusion.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("验证集缺少类别")
    recalls = np.diag(confusion) / counts
    accuracy = float(np.trace(confusion) / len(dataset))
    macro = float(recalls.mean())
    summary = {
        "checkpoint": str(checkpoint_path), "epoch": int(checkpoint["epoch"]),
        "architecture": architecture,
        "split": "val", "split_sha256": digest, "count": len(dataset),
        "accuracy": accuracy, "macro_recall": macro,
        "roses_to_carnations": int(confusion[classes.index("roses"), classes.index("carnations")]),
        "carnations_to_roses": int(confusion[classes.index("carnations"), classes.index("roses")]),
        "roses_recall": float(recalls[classes.index("roses")]),
        "carnations_recall": float(recalls[classes.index("carnations")]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(confusion, index=classes, columns=classes).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_class",
    )
    predictions_table = dataset.rows[["path", "class", "label"]].copy()
    predictions_table = predictions_table.rename(columns={"class": "true_class"})
    predictions_table["predicted_class"] = [classes[i] for i in predicted]
    predictions_table["correct"] = true == predicted
    predictions_table["softmax_score"] = top_scores
    predictions_table.to_csv(output_dir / "val_predictions.csv", index=False)
    print("验证图片总数：", summary["count"])
    print(f"验证准确率：{accuracy:.1%}；验证平均类别召回率：{macro:.1%}")
    print(f"玫瑰：{confusion[classes.index('roses'), classes.index('roses')]}/{counts[classes.index('roses')]} 正确，召回率 {summary['roses_recall']:.1%}")
    print(f"康乃馨：{confusion[classes.index('carnations'), classes.index('carnations')]}/{counts[classes.index('carnations')]} 正确，召回率 {summary['carnations_recall']:.1%}")
    print(f"玫瑰 → 康乃馨：{summary['roses_to_carnations']} 张")
    print(f"康乃馨 → 玫瑰：{summary['carnations_to_roses']} 张")
    print("验证分析已保存到：", output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="检查已选模型在固定验证集上的错分")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请改用 --device cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = (args.output_dir or checkpoint_path.parent / "validation_analysis").expanduser().resolve()
    try:
        check(checkpoint_path, output_dir, torch.device(name))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"验证集检查失败：{error}\n")


if __name__ == "__main__":
    main()
