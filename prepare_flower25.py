"""组合原五类与 Oxford 新增二十类，保存独立清单，不覆盖五类结果。"""

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

from flower25_config import BASE_CLASSES, EXTRA_CLASSES, CLASS_NAMES, CLASS_NAMES_ZH


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"


def official_categories(cache_dir):
    labels = loadmat(cache_dir / "imagelabels.mat")["labels"].reshape(-1)
    content = (cache_dir / "categories.html").read_text(encoding="utf-8")
    # 官方类别表的示例图片有名称；通过该图片的标签核实类别编号。
    thumbnails = re.findall(r'thumbim_(\d+)\.jpg[^>]*alt="([^"]+)"', content)
    categories = {
        html.unescape(name).strip(): int(labels[int(image_id) - 1])
        for image_id, name in thumbnails
    }
    if len(categories) != 102 or len(set(categories.values())) != 102:
        raise ValueError("官方类别表格式变化，未能核实全部 102 个类别")
    used = [name for _, _, source_names in EXTRA_CLASSES for name in source_names]
    if len(set(used)) != len(used) or not set(used).issubset(categories):
        raise ValueError(f"新增类别的名称缺失或被重复使用：{sorted(set(used) - set(categories))}")
    return labels, categories


def image_digest(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        # 对解码后的像素去重，忽略文件名和 EXIF 等元数据差异。
        size = f"{image.width},{image.height}:".encode("ascii")
        return hashlib.sha256(size + image.tobytes()).hexdigest()


def prepare(cache_dir, output_dir, metadata_only=False):
    labels, categories = official_categories(cache_dir)
    print("核实官方标签：", len(labels), "张图片，102 个来源类别", flush=True)
    for name, chinese, source_names in EXTRA_CLASSES:
        count = sum(int((labels == categories[source]).sum()) for source in source_names)
        print(f"  {chinese} ({name})：{count} 张", flush=True)
    if metadata_only:
        return

    original_path = ROOT / "outputs" / "split_manifest.csv"
    original = pd.read_csv(original_path)
    actual = original[["class", "label"]].drop_duplicates()
    if dict(zip(actual["class"], actual["label"])) != {
        name: index for index, (name, _) in enumerate(BASE_CLASSES)
    } or len(actual) != 5:
        raise ValueError("原五类清单的类别编号不匹配")
    if original["path"].duplicated().any():
        raise ValueError("原清单有重复路径")
    if not set(original["split"]).issubset({"train", "val", "test"}):
        raise ValueError("原清单存在未知的数据分组")

    records, duplicates = [], []
    # 如发现原数据有相同像素，优先保留 test，再 val，再 train，防止跨组重复。
    priority = {"test": 0, "val": 1, "train": 2}
    original = original.assign(_priority=original["split"].map(priority))
    for record in original.sort_values("_priority", kind="stable").to_dict("records"):
        digest = image_digest(ROOT / record["path"])
        record.pop("_priority")
        record.update(source="tensorflow_flower_photos", image_sha256=digest)
        records.append(record)

    print("原五类图片已检查，开始提取新增类别。", flush=True)
    archive_path = cache_dir / "102flowers.tgz"
    wanted, extracted = {}, set()
    for label, (name, _, source_names) in enumerate(EXTRA_CLASSES, start=5):
        source_ids = [categories[source] for source in source_names]
        for image_id in (np.flatnonzero(np.isin(labels, source_ids)) + 1).tolist():
            wanted[f"jpg/image_{image_id:05d}.jpg"] = (image_id, name, label)
    # 顺序读取一次压缩包，避免为每张图片反复解压前面的内容。
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name in wanted:
                if member.name in extracted:
                    raise ValueError(f"压缩包中图片路径重复：{member.name}")
                extracted.add(member.name)
                image_id, name, label = wanted[member.name]
                filename = f"image_{image_id:05d}.jpg"
                class_dir = cache_dir / "selected_images" / name
                class_dir.mkdir(parents=True, exist_ok=True)
                destination = class_dir / f"oxford_{filename}"
                # 只写入显式构造的本地文件路径，不按压缩包中的路径展开。
                if not member.isfile():
                    raise ValueError(f"压缩包成员不是普通图片：{filename}")
                with archive.extractfile(member) as stream:
                    archived_bytes = stream.read()
                if destination.exists():
                    if destination.read_bytes() != archived_bytes:
                        raise ValueError(f"已有图片与官方压缩包不同：{destination}")
                else:
                    destination.write_bytes(archived_bytes)
                relative_path = destination.relative_to(ROOT).as_posix()
                digest = image_digest(destination)
                record = {
                    "path": relative_path, "class": name, "label": label,
                    "source": "oxford_flowers102", "source_label": int(labels[image_id - 1]),
                    "image_sha256": digest,
                }
                records.append(record)
    if extracted != set(wanted):
        raise ValueError("压缩包没有提供全部选中的图片，请检查下载是否完整")

    # 相同像素同标签只保留一张；相同像素不同标签全部排除，不自动猜测正确类别。
    pixel_groups = {}
    for record in records:
        pixel_groups.setdefault(record["image_sha256"], []).append(record)
    cleaned, conflict_groups, conflict_images = [], 0, 0
    original_cross_split_groups = 0
    for group in pixel_groups.values():
        original_group = [row for row in group if row["source"] == "tensorflow_flower_photos"]
        if len({row["split"] for row in original_group}) > 1:
            original_cross_split_groups += 1
        if len({row["class"] for row in group}) > 1:
            conflict_groups += 1
            conflict_images += len(group)
            for row in group:
                duplicates.append({"path": row["path"], "class": row["class"],
                                   "kept_path": "", "reason": "conflicting_labels_excluded"})
            continue
        group.sort(key=lambda row: (priority.get(row.get("split"), 3), row["path"]))
        cleaned.append(group[0])
        for row in group[1:]:
            duplicates.append({"path": row["path"], "class": row["class"],
                               "kept_path": group[0]["path"], "reason": "same_pixels_same_label"})
    records = [row for row in cleaned if row["label"] < 5]
    for label, (name, _, _) in enumerate(EXTRA_CLASSES, start=5):
        class_records = sorted([row for row in cleaned if row["class"] == name],
                               key=lambda row: row["path"])
        if len(class_records) < 10:
            raise ValueError(f"{name} 可用图片过少")
        random.Random(42 + label).shuffle(class_records)
        n_train, n_val = int(len(class_records) * 0.70), int(len(class_records) * 0.15)
        for index, record in enumerate(class_records):
            record["split"] = "train" if index < n_train else (
                "val" if index < n_train + n_val else "test"
            )
        records.extend(class_records)
        print(f"  已准备 {name}：{len(class_records)} 张", flush=True)

    manifest = pd.DataFrame(records)
    if manifest["path"].duplicated().any() or manifest["image_sha256"].duplicated().any():
        raise ValueError("新版清单仍存在重复图片")
    counts = manifest.groupby("class").size().reindex(CLASS_NAMES)
    split_counts = manifest.groupby(["class", "split"]).size().unstack(fill_value=0)
    split_counts = split_counts.reindex(index=CLASS_NAMES, columns=["train", "val", "test"])
    if split_counts.isna().any().any() or (split_counts <= 0).any().any():
        raise ValueError("某个类别的数据分组为空")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    split_counts.to_csv(output_dir / "split_counts.csv", index_label="class", encoding="utf-8-sig")
    class_counts = pd.DataFrame({
        "label": range(25), "class": CLASS_NAMES,
        "name_zh": [CLASS_NAMES_ZH[name] for name in CLASS_NAMES],
        "count": counts.to_numpy(),
    })
    class_counts["percentage"] = (class_counts["count"] / len(manifest) * 100).round(2)
    class_counts.to_csv(output_dir / "class_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicates, columns=["path", "class", "kept_path", "reason"]).to_csv(
        output_dir / "duplicates_removed.csv", index=False,
    )
    metadata = {
        "class_names": CLASS_NAMES, "class_names_zh": CLASS_NAMES_ZH,
        "oxford_source": SOURCE_URL,
        "oxford_category_mapping": [
            {"class": name, "name_zh": chinese,
             "source_categories": {source: categories[source] for source in source_names}}
            for name, chinese, source_names in EXTRA_CLASSES
        ],
        "oxford_labels_sha256": hashlib.sha256((cache_dir / "imagelabels.mat").read_bytes()).hexdigest(),
        "original_manifest_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split_policy": "Preserve original five-class splits after exact-pixel deduplication; new classes 70/15/remainder, seed=42+label. Not Oxford official splits.",
        "deduplication": "Same RGB pixels and dimensions; does not detect all near-duplicates.",
        "original_test_status": "Original five-class test results and example images have already been inspected; combined test is not a completely unseen final benchmark.",
        "images": len(manifest), "exact_duplicates_removed": len(duplicates),
        "conflicting_label_groups_excluded": conflict_groups,
        "conflicting_label_images_excluded": conflict_images,
        "original_cross_split_duplicate_groups": original_cross_split_groups,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    shutil.copyfile(cache_dir / "README.txt", output_dir / "oxford_README.txt")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(CLASS_NAMES, counts.to_numpy(), color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("25-class flower dataset")
    fig.tight_layout()
    fig.savefig(output_dir / "class_counts.png", dpi=150)
    plt.close(fig)
    print("\n25 类图片总数：", len(manifest))
    print("去除完全相同的图片：", len(duplicates))
    print(split_counts.to_string())
    print("各组总数：", manifest["split"].value_counts().to_dict())
    print("新版清单：", manifest_path)


def main():
    parser = argparse.ArgumentParser(description="准备独立的 25 类花卉实验")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "flowers102")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "flower25")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(ROOT / "outputs") or output_dir == ROOT / "outputs":
        parser.error("新版清单应放在 outputs 的独立子目录中")
    try:
        prepare(args.cache_dir.resolve(), output_dir, args.metadata_only)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"准备失败：{error}\n")


if __name__ == "__main__":
    main()
