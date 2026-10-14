"""核对玫瑰、蒲公英和康乃馨图片清单，并生成可人工复核的缩略图。"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from flower25_config import CLASS_NAMES


ROOT = Path(__file__).resolve().parent
TARGET_CLASSES = ("roses", "dandelion", "carnations")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def make_sheet(name, records, output_dir):
    if not records:
        return []
    columns, cell_width, cell_height = 6, 180, 190
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        x = index % columns * cell_width
        y = index // columns * cell_height
        with Image.open(ROOT / record["path"]) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
            photo.thumbnail((cell_width - 12, cell_height - 38))
            sheet.paste(photo, (x + (cell_width - photo.width) // 2, y))
        draw.text((x + 4, y + cell_height - 35), f"{index + 1}: {record['class']}", fill="black")
        draw.text((x + 4, y + cell_height - 19), Path(record["path"]).stem[:22], fill="black")
    sheet.save(output_dir / f"{name}.png")
    return [dict(record, sheet=name, cell=index)
            for index, record in enumerate(records, start=1)]


def audit(manifest_path, predictions_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(manifest_path)
    targets = [row for row in manifest if row["class"] in TARGET_CLASSES]
    issues = []
    actual_digests = defaultdict(list)
    valid_paths = set()
    by_split = Counter((row["class"], row["split"]) for row in targets)
    by_source = Counter((row["class"], row["source"]) for row in targets)
    expected_labels = {name: CLASS_NAMES.index(name) for name in TARGET_CLASSES}

    for row in targets:
        path = row["path"]
        image_path = ROOT / path
        if row["label"] != str(expected_labels[row["class"]]):
            issues.append({"path": path, "issue": "class_label_mismatch"})
        if Path(path).parent.name != row["class"]:
            issues.append({"path": path, "issue": "path_class_mismatch"})
        expected_source = ("oxford_flowers102" if row["class"] == "carnations"
                           else "tensorflow_flower_photos")
        if row["source"] != expected_source:
            issues.append({"path": path, "issue": "unexpected_source"})
        if not image_path.is_file():
            issues.append({"path": path, "issue": "missing_file"})
            continue
        try:
            with Image.open(image_path) as source:
                photo = source.convert("RGB")
                encoded_size = f"{photo.width},{photo.height}:".encode("ascii")
                digest = hashlib.sha256(encoded_size + photo.tobytes()).hexdigest()
        except (OSError, ValueError) as error:
            issues.append({"path": path, "issue": f"decode_error: {error}"})
            continue
        actual_digests[digest].append(path)
        valid_paths.add(path)
        if digest != row["image_sha256"]:
            issues.append({"path": path, "issue": "image_sha256_mismatch"})

    for paths in actual_digests.values():
        if len(paths) > 1:
            for path in paths:
                issues.append({"path": path, "issue": "exact_pixel_duplicate"})

    predictions = read_csv(predictions_path)
    manifest_by_path = {row["path"]: row for row in manifest}
    wrong_roses = [row for row in predictions
                   if row["true_class"] == "roses"
                   and row["predicted_class"] == "dandelion"]
    for row in wrong_roses:
        source = manifest_by_path.get(row["path"])
        if source is None or source["class"] != "roses" or source["split"] != "val":
            issues.append({"path": row["path"], "issue": "prediction_manifest_mismatch"})

    selected = []
    rng = random.Random(42)
    for name in TARGET_CLASSES:
        train_rows = [row.copy() for row in targets
                      if row["class"] == name and row["split"] == "train"
                      and row["path"] in valid_paths]
        sample = rng.sample(train_rows, min(36, len(train_rows)))
        selected.extend(make_sheet(f"train_{name}_sample", sample, output_dir))
        if name == "roses":
            # 全量分页面板供人工检查；这一步不会自动改动任何标签。
            for page_start in range(0, len(train_rows), 36):
                page = train_rows[page_start:page_start + 36]
                page_name = f"train_roses_page_{page_start // 36 + 1:02d}"
                selected.extend(make_sheet(page_name, page, output_dir))
    error_rows = [dict(manifest_by_path[row["path"]]) for row in wrong_roses
                  if row["path"] in manifest_by_path
                  and row["path"] in valid_paths]
    selected.extend(make_sheet("val_roses_predicted_dandelion", error_rows,
                               output_dir))

    write_csv(output_dir / "issues.csv", issues, ["path", "issue"])
    write_csv(output_dir / "sheet_index.csv", selected,
              ["sheet", "cell", "path", "class", "label", "split", "source",
               "image_sha256", "source_label"])
    summary = {
        "checked_images": len(targets),
        "issue_count": len(issues),
        "counts_by_split": {name: {split: by_split[name, split]
                                   for split in ("train", "val", "test")}
                            for name in TARGET_CLASSES},
        "counts_by_source": {name: {source: count for (class_name, source), count
                                    in by_source.items() if class_name == name}
                             for name in TARGET_CLASSES},
        "candidate_val_roses_predicted_dandelion": len(wrong_roses),
        "visual_review": "All 447 train roses have contact pages; 36 random train dandelions, all 35 train carnations, and all 18 validation roses predicted dandelion are shown. Manual review candidates are listed separately; contact sheets cannot prove every label correct.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已核对 {len(targets)} 张目标类别图片，发现 {len(issues)} 条文件/清单问题。")
    print(f"候选版验证集中，玫瑰被判为蒲公英：{len(wrong_roses)} 张。")
    print("审查结果：", output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "outputs/flower25/split_manifest.csv")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "outputs/train25_20261014_152630_011600/validation_analysis/val_predictions.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/flower25/label_audit")
    args = parser.parse_args()
    audit(args.manifest.resolve(), args.predictions.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
