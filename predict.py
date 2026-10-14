"""使用已训练的手写或预训练对照模型预测单张图片，并保存展示材料。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18

from resnet import FlowerResNet


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "outputs" / "train_20261012_102328" / "best_model.pth"
)
CHINESE_NAMES = {
    "daisy": "雏菊",
    "dandelion": "蒲公英",
    "roses": "玫瑰",
    "sunflowers": "向日葵",
    "tulips": "郁金香",
}


def predict_image(image_path, checkpoint_path, output_dir, device):
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片：{image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型：{checkpoint_path}")

    # 恢复已训练的参数，类别顺序以模型保存的信息为准。
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True,
    )
    class_names = checkpoint["class_names"]
    chinese_names = checkpoint.get("class_names_zh", CHINESE_NAMES)
    image_size = int(checkpoint["image_size"])
    architecture = checkpoint.get("architecture", "handwritten_flower_resnet")
    if architecture == "torchvision_resnet18_pretrained_head":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["normalization_mean"],
                                 checkpoint["normalization_std"]),
        ])
    elif architecture == "handwritten_flower_resnet":
        model = FlowerResNet(num_classes=len(class_names))
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"不支持的模型结构：{architecture}")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 校正手机照片方向，再按照验证时的方式处理图片。
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    # 增加批次维度：[3, H, W] -> [1, 3, H, W]。
    batch = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    ranking = probabilities.argsort(descending=True).tolist()
    predictions = [
        {
            "class": class_names[index],
            "name_zh": chinese_names.get(class_names[index], class_names[index]),
            "score": float(probabilities[index]),
        }
        for index in ranking
    ]

    result = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": architecture,
        "device": str(device),
        "predicted_class": predictions[0]["class"],
        "predictions": predictions,
        "score_note": f"分数是 {len(class_names)} 个已知类别的 softmax 分数，不是实测准确率。",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    basename = f"{image_path.stem}_{stamp}"
    json_path = output_dir / f"{basename}.json"
    chart_path = output_dir / f"{basename}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(class_names) * 0.28 + 1)))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title(f"Prediction: {predictions[0]['class']}")

    axes[1].barh(
        [item["class"] for item in predictions],
        [item["score"] * 100 for item in predictions],
        color="steelblue",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Softmax score (%)")
    axes[1].set_title(f"Scores among {len(class_names)} supported classes")
    for index, item in enumerate(predictions):
        axes[1].text(
            98, index, f"{item['score']:.1%}",
            ha="right", va="center",
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result, chart_path, json_path


def main():
    parser = argparse.ArgumentParser(description="对一张花卉图片进行分类预测")
    parser.add_argument("image", type=Path, help="图片路径，含空格时加双引号")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="训练得到的模型文件路径",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_DIR / "outputs" / "predictions",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("当前无法使用 CUDA，请检查显卡环境，或选择 --device cpu")

    try:
        result, chart_path, json_path = predict_image(
            args.image.expanduser().resolve(),
            args.checkpoint.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            torch.device(device_name),
        )
    except (OSError, UnidentifiedImageError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(2, f"预测失败：{error}\n")

    top = result["predictions"][0]
    print("运行设备：", result["device"])
    print("模型类型：", "预训练 ResNet18 对照" if result["architecture"] ==
          "torchvision_resnet18_pretrained_head" else "手写 ResNet")
    print(f"预测类别：{top['name_zh']}（{top['class']}）")
    print("各类别的模型分数：")
    for item in result["predictions"]:
        print(f"  {item['name_zh']}：{item['score']:.1%}")
    print(f"分数不代表实测准确率；模型仅支持以上 {len(result['predictions'])} 类花卉。")
    print("预测展示图：", chart_path)
    print("预测结果记录：", json_path)


if __name__ == "__main__":
    main()
