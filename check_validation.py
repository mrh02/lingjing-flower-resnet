"""只检查固定验证集的错分；不更新模型，也不读取测试集。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]),) * 2),
        transforms.ToTensor(),
    ])
    dataset = FlowerDataset("val", transform, manifest_path=manifest_path)
    if not len(dataset):
        raise ValueError("验证集为空")
    model = FlowerResNet(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predicted = []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
            predicted.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
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
