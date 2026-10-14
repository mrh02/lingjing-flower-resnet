"""用 Grad-CAM、选区遮挡及灰度对比检查单张图片的分类依据。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from predict import CHINESE_NAMES, PROJECT_DIR
from resnet import FlowerResNet


def grad_cam(model, batch, target_index=None):
    """对类别原始分数求梯度；不训练模型，不修改参数。"""
    features = []
    handle = model.stage4.register_forward_hook(
        lambda module, inputs, output: features.append(output)
    )
    try:
        with torch.enable_grad():
            logits = model(batch.detach().clone().requires_grad_(True))
            if target_index is None:
                target_index = int(logits[0].argmax())
            activation = features[0]
            gradient = torch.autograd.grad(logits[0, target_index], activation)[0]
            # 每个通道的空间平均梯度作为权重，保留对目标分数的正向贡献。
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            raw_cam = torch.relu((weights * activation).sum(dim=1))[0]
            low, high = raw_cam.min(), raw_cam.max()
            has_contrast = bool((high - low).item() > 1e-12)
            cam = ((raw_cam - low) / (high - low) if has_contrast
                   else torch.zeros_like(raw_cam))
        return logits.detach(), target_index, cam.detach().cpu(), has_contrast
    finally:
        handle.remove()


def make_variants(image, box):
    """选区是人工指定矩形，不是自动花朵分割。灰色填充值固定为 127。"""
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("选区必须满足 0 <= x1 < x2 <= 1、0 <= y1 < y2 <= 1")
    width, height = image.size
    pixel_box = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
    left, top, right, bottom = pixel_box
    if left >= right or top >= bottom:
        raise ValueError("选区太小，至少需要覆盖一个像素")
    original = np.array(image)
    covered = original.copy()
    covered[top:bottom, left:right] = 127
    kept = np.full_like(original, 127)
    kept[top:bottom, left:right] = original[top:bottom, left:right]
    return {
        "original": image,
        "cover_selected_area": Image.fromarray(covered),
        "keep_selected_area": Image.fromarray(kept),
        "grayscale": ImageOps.grayscale(image).convert("RGB"),
    }, pixel_box


def explain_image(image_path, checkpoint_path, output_dir, device, box=None, target=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = checkpoint["class_names"]
    chinese_names = checkpoint.get("class_names_zh", CHINESE_NAMES)
    if target is not None and target not in classes:
        raise ValueError(f"目标类别 {target} 不在模型类别中：{', '.join(classes)}")
    model = FlowerResNet(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]),) * 2),
        transforms.ToTensor(),
    ])
    original_batch = transform(image).unsqueeze(0).to(device)
    logits, target_index, cam, has_contrast = grad_cam(
        model, original_batch, classes.index(target) if target is not None else None,
    )
    # 所有改图实验比较同一个目标类别，不能在每张图上偷偷换比较对象。
    target_class = classes[target_index]
    supplied_box = box is not None
    box = tuple(box) if supplied_box else (0.25, 0.25, 0.75, 0.75)
    variants, pixel_box = make_variants(image, box)
    records = []
    for name, variant in variants.items():
        if name == "original":
            variant_logits = logits
        else:
            with torch.inference_mode():
                variant_logits = model(transform(variant).unsqueeze(0).to(device))
        scores = variant_logits.softmax(dim=1)[0].cpu()
        prediction = int(scores.argmax())
        records.append({
            "variant": name,
            "predicted_class": classes[prediction],
            "predicted_name_zh": chinese_names.get(classes[prediction], classes[prediction]),
            "predicted_score": float(scores[prediction]),
            "target_class": target_class,
            "target_score": float(scores[target_index]),
            "target_logit": float(variant_logits[0, target_index]),
            "scores": {name: float(scores[i]) for i, name in enumerate(classes)},
        })
    baseline = records[0]["target_score"]
    for record in records:
        record["target_score_change_pp"] = (record["target_score"] - baseline) * 100

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    chart_path = output_dir / f"{basename}.png"
    json_path = output_dir / f"{basename}.json"
    # 限制展示尺寸；实际预测仍使用原图经相同预处理得到的输入。
    display = image.copy()
    display.thumbnail((1000, 1000))
    heat = torch.nn.functional.interpolate(
        cam[None, None], size=(display.height, display.width),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    target_name = chinese_names.get(target_class, target_class)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    axes[0, 0].imshow(display)
    axes[0, 0].add_patch(Rectangle(
        (box[0] * display.width, box[1] * display.height),
        (box[2] - box[0]) * display.width, (box[3] - box[1]) * display.height,
        fill=False, edgecolor="cyan", linewidth=2,
    ))
    axes[0, 0].set_title(f"原图：预测为{records[0]['predicted_name_zh']}\n青色框是本次人工选区")
    axes[0, 1].imshow(display)
    if has_contrast:
        axes[0, 1].imshow(heat, cmap="inferno", vmin=0, vmax=1, alpha=0.55)
    axes[0, 1].set_title(f"解释目标：{target_name}\n"
                         + ("亮色表示相对更强的正向贡献" if has_contrast else "未得到有对比度的正向热图"))
    labels = {"cover_selected_area": "遮挡选区",
              "keep_selected_area": "仅保留选区", "grayscale": "转成灰度图"}
    for axis, record in zip((axes[0, 2], axes[1, 0], axes[1, 1]), records[1:]):
        variant = variants[record["variant"]].copy()
        variant.thumbnail((1000, 1000))
        axis.imshow(variant)
        axis.set_title(f"{labels[record['variant']]}\n"
                       f"预测：{record['predicted_name_zh']}（{record['predicted_score']:.1%}）")
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        axis.axis("off")
    bars = axes[1, 2]
    bars.barh(["原图", "遮挡选区", "仅保留选区", "灰度图"],
              [record["target_score"] * 100 for record in records], color="steelblue")
    bars.invert_yaxis()
    bars.set_xlim(0, 100)
    bars.set_xlabel(f"统一比较“{target_name}”的模型分数（%）")
    bars.set_title("四种图片比较同一个类别的分数")
    for i, record in enumerate(records):
        bars.text(99, i, f"{record['target_score']:.1%}", ha="right", va="center")
    fig.suptitle("单张图片的判断依据检查：结果提供线索，不能证明完整判断原因\n"
                 "矩形选区不是花朵分割；模型分数不是准确率。", fontsize=13)
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    result = {
        "image_path": str(image_path), "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]), "device": str(device),
        "image_size_original": list(image.size),
        "target_class": target_class,
        "target_name_zh": chinese_names.get(target_class, target_class),
        "grad_cam_layer": "stage4 (final residual stage output)",
        "grad_cam_native_shape": list(cam.shape),
        "grad_cam_has_contrast": has_contrast,
        "box_mode": "manual_rectangle" if supplied_box else "default_center_rectangle",
        "box_normalized": list(box), "box_pixels": list(pixel_box),
        "mask_fill_rgb": [127, 127, 127], "variants": records,
        "notes": [
            "Grad-CAM 是目标类别原始分数的粗略正向贡献定位，不能证明具体颜色或形状的因果作用。",
            "热图按本张图片归一化，亮度不能作为跨图片或类别的贡献大小比较。",
            "选区是矩形，不是自动花朵分割；选区内外均可能包含花朵和背景。",
            "所有修改图都比较同一个目标类别；softmax 分数不是准确率。",
            "灰度和灰色遮挡改变了输入分布，实验仅提供单张图片的线索。",
            "不训练或更新模型参数，不计入验证或测试成绩。",
        ],
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, chart_path, json_path


def main():
    parser = argparse.ArgumentParser(description="热力图、矩形遮挡和灰度对比")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "explanations")
    parser.add_argument("--target", help="要解释的英文类别名，默认解释原图预测类别")
    parser.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="人工选区左上、右下坐标，按宽高比例 0～1；默认中央矩形")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请选择 --device cpu")
    try:
        result, chart, record_path = explain_image(
            args.image.expanduser().resolve(), args.checkpoint.expanduser().resolve(),
            args.output_dir.expanduser().resolve(), torch.device(device), args.box, args.target,
        )
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(2, f"检查失败：{error}\n")
    print("运行设备：", device)
    print(f"本次统一比较的类别：{result['target_name_zh']}（{result['target_class']}）")
    print("选区：", result["box_mode"], result["box_normalized"])
    names = {"original": "原图", "cover_selected_area": "遮挡选区",
             "keep_selected_area": "仅保留选区", "grayscale": "灰度图"}
    for record in result["variants"]:
        print(f"{names[record['variant']]}：预测 {record['predicted_name_zh']}；"
              f"目标类别分数 {record['target_score']:.1%}，"
              f"较原图变化 {record['target_score_change_pp']:+.1f} 个百分点")
    if not result["grad_cam_has_contrast"]:
        print("本次未得到有对比度的正向热力图，不能据此确定重要区域。")
    print("选区不是自动花朵分割；热图和分数变化提供线索，不能证明完整判断原因。")
    print("检查展示图：", chart)
    print("检查记录：", record_path)


if __name__ == "__main__":
    main()
