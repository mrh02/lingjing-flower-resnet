"""Add reviewed Commons carnation photos to train only, preserving val/test."""

import argparse
import hashlib
import io
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "outputs" / "flower25"
OUTPUT_DIR = ROOT / "outputs" / "flower25_carnation_v1"
SOURCE_CSV = ROOT / "carnation_extra_sources.csv"
IMAGE_DIR = ROOT / "data" / "carnation_extra" / "selected"
CANDIDATE_DIR = ROOT / "data" / "carnation_extra" / "candidates" / "thumbs"


def pixel_digest(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        prefix = f"{image.width},{image.height}:".encode("ascii")
        return hashlib.sha256(prefix + image.tobytes()).hexdigest()


def difference_hash(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8))
        pixels = np.asarray(image)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int("".join("1" if bit else "0" for bit in bits.flat), 2)


def local_image(source):
    pageid = int(source.pageid)
    image_url = urllib.parse.urlparse(source.image_url)
    if image_url.scheme != "https" or image_url.hostname not in {
        "upload.wikimedia.org", "thumb.wikimedia.org",
    }:
        raise ValueError(f"Unexpected Commons image host: {source.page_url}")
    candidate = CANDIDATE_DIR / f"{pageid}.img"
    if candidate.is_file():
        raw = candidate.read_bytes()
    else:
        request = urllib.request.Request(
            source.image_url,
            headers={"User-Agent": "LingjingFlowerResNet/0.1 (educational; mrh02/lingjing-flower-resnet)"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source.file_sha256:
        raise ValueError(f"Source image changed; review its page before using it: {source.page_url}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        if min(image.size) < 128:
            raise ValueError(f"Source image too small: {source.page_url}")
        suffix = ".png" if image.format == "PNG" else ".jpg" if image.format == "JPEG" else None
    if suffix is None:
        raise ValueError(f"Unsupported image format: {source.page_url}")
    destination = IMAGE_DIR / f"commons_{pageid}{suffix}"
    if destination.is_file() and destination.read_bytes() != raw:
        raise ValueError(f"Existing local image changed: {destination}")
    if not destination.is_file():
        destination.write_bytes(raw)
    return destination


def prepare():
    base_path = BASE_DIR / "split_manifest.csv"
    base_bytes = base_path.read_bytes()
    base_info = json.loads((BASE_DIR / "dataset_info.json").read_text(encoding="utf-8"))
    base_digest = hashlib.sha256(base_bytes).hexdigest()
    if base_digest != base_info["split_sha256"]:
        raise ValueError("Original 25-class manifest changed")
    base = pd.read_csv(base_path)
    sources = pd.read_csv(SOURCE_CSV)
    required = {"pageid", "page_url", "image_url", "license", "file_sha256"}
    if not required.issubset(sources.columns) or sources.pageid.duplicated().any():
        raise ValueError("Invalid source list")
    if len(base) != base_info["images"] or base.image_sha256.duplicated().any():
        raise ValueError("Original manifest contains unexpected rows or duplicates")
    if not set(base.split).issubset({"train", "val", "test"}):
        raise ValueError("Original manifest contains unknown split")

    # Compare with every existing image, including untouched validation and test.
    hashes = []
    for path in base.path:
        image_path = ROOT / path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        hashes.append(difference_hash(image_path))
    pixel_hashes = set(base.image_sha256)
    extra_rows, audit = [], []
    for source in sources.itertuples(index=False):
        path = local_image(source)
        pixel_hash = pixel_digest(path)
        visual_hash = difference_hash(path)
        closest = min((visual_hash ^ other).bit_count() for other in hashes)
        status = "accepted"
        if pixel_hash in pixel_hashes:
            status = "exact_duplicate"
        elif closest <= 4:
            status = "near_duplicate_review_needed"
        audit.append({
            "pageid": int(source.pageid), "page_url": source.page_url,
            "status": status, "nearest_dhash_distance": closest,
        })
        if status != "accepted":
            continue
        hashes.append(visual_hash)
        pixel_hashes.add(pixel_hash)
        extra_rows.append({
            "path": path.relative_to(ROOT).as_posix(), "class": "carnations",
            "label": 10, "split": "train", "source": "wikimedia_commons",
            "image_sha256": pixel_hash, "source_label": float("nan"),
        })
    if not extra_rows:
        raise ValueError("No usable new carnation photos")
    extra = pd.DataFrame(extra_rows, columns=base.columns)
    combined = pd.concat([base, extra], ignore_index=True)
    pd.testing.assert_frame_equal(combined.iloc[:len(base)].reset_index(drop=True), base.reset_index(drop=True))
    for split in ("val", "test"):
        pd.testing.assert_frame_equal(
            combined.loc[combined.split == split].reset_index(drop=True),
            base.loc[base.split == split].reset_index(drop=True),
        )
    if combined.image_sha256.duplicated().any() or combined.path.duplicated().any():
        raise ValueError("New manifest contains duplicate images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "split_manifest.csv"
    combined.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    count_table = combined.groupby(["class", "split"]).size().unstack(fill_value=0)
    count_table = count_table.reindex(index=base_info["class_names"], columns=["train", "val", "test"])
    count_table.to_csv(OUTPUT_DIR / "split_counts.csv", index_label="class", encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(OUTPUT_DIR / "addition_audit.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(SOURCE_CSV, OUTPUT_DIR / "sources.csv")
    info = dict(base_info)
    info.update({
        "split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "base_split_sha256": base_digest,
        "images": len(combined),
        "added_carnation_train_images": len(extra),
        "carnation_train_images": int(count_table.loc["carnations", "train"]),
        "carnation_val_images": int(count_table.loc["carnations", "val"]),
        "carnation_test_images": int(count_table.loc["carnations", "test"]),
        "commons_source_category": "https://commons.wikimedia.org/wiki/Category:Dianthus_caryophyllus",
        "commons_selection": "Manually reviewed photos with per-file attribution in carnation_extra_sources.csv; Commons thumbnails around 500px for 128px training inputs.",
        "split_policy": "Original 25-class train/val/test rows unchanged; vetted Commons carnations appended to train only.",
        "near_duplicate_policy": "64-bit difference hash <=4 excluded pending review; visual similarity audit is imperfect.",
    })
    (OUTPUT_DIR / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"新增康乃馨训练图片：{len(extra)} 张")
    print(f"康乃馨：训练 {count_table.loc['carnations', 'train']}，验证 {count_table.loc['carnations', 'val']}，测试 {count_table.loc['carnations', 'test']}")
    print("候选图片检查：", pd.Series([row["status"] for row in audit]).value_counts().to_dict())
    print("新训练清单：", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Build the independent training manifest")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        parser.print_help()
