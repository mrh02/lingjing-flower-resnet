"""Freeze reviewed rose/carnation additions and an independent external check set."""

import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from prepare_carnation_extra import difference_hash, pixel_digest


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "flower25_carnation_v1"
OUTPUT = ROOT / "outputs" / "flower25_targeted_v2"
SOURCE_ROOT = ROOT / "data" / "targeted_extra"
SELECTED = SOURCE_ROOT / "selected"

# IDs were selected from contact sheets before model predictions were inspected.
# Same-source photographers are kept on only one side of the train/holdout divide.
ROSE_TRAIN = [0, 2, 4, 6, 9, 12, 14, 17, 50, 55, 57, 60, 64, 65, 66,
              72, 77, 81, 82, 84, 85, 92, 98, 102, 105, 108]
ROSE_HOLDOUT = [130, 134, 135, 138, 139, 143, 149, 151, 156, 158,
                164, 170, 175, 180, 188, 200, 230, 235]
CARNATION_TRAIN = [2, 8, 20, 25, 27, 28, 30, 31, 36, 43, 49,
                   51, 53, 56, 61]
CARNATION_HOLDOUT = [24, 29, 33, 58, 62, 63, 64, 65, 66, 72, 73, 74, 75]


def normalize_artist(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", "", value))).strip().casefold()


def selected_rows():
    roses = json.loads((SOURCE_ROOT / "roses_closeups" / "metadata.json").read_text(encoding="utf-8"))
    carnations = json.loads((SOURCE_ROOT / "inat_carnations" / "metadata.json").read_text(encoding="utf-8"))
    for source_name, rows, class_name, train_ids, holdout_ids in [
        ("commons", roses, "roses", ROSE_TRAIN, ROSE_HOLDOUT),
        ("inaturalist", carnations, "carnations", CARNATION_TRAIN, CARNATION_HOLDOUT),
    ]:
        overlap = set(train_ids) & set(holdout_ids)
        if overlap:
            raise ValueError(f"Selection overlaps: {overlap}")
        train_authors = set()
        holdout_authors = set()
        for usage, indexes in (("train", train_ids), ("external_holdout", holdout_ids)):
            for index in indexes:
                item = rows[index]
                author = normalize_artist(item.get("artist", item.get("user", "")))
                if not author:
                    raise ValueError(f"Missing photographer: {source_name} {index}")
                (train_authors if usage == "train" else holdout_authors).add(author)
                identifier = int(item.get("pageid", item.get("photo_id")))
                folder = "roses_closeups" if source_name == "commons" else "inat_carnations"
                source_path = SOURCE_ROOT / folder / "thumbs" / f"{identifier}.img"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                license_name = item["license"].lower()
                if not (license_name.startswith("cc by") or license_name == "cc0"
                        or license_name == "public domain" or license_name in {"cc-by", "cc-by-sa"}):
                    raise ValueError(f"Unexpected image license: {item['license']}")
                with Image.open(source_path) as image:
                    image.verify()
                with Image.open(source_path) as image:
                    if min(image.size) < 128 or image.format not in {"JPEG", "PNG"}:
                        raise ValueError(f"Unusable image {source_path}")
                    ext = ".png" if image.format == "PNG" else ".jpg"
                SELECTED.mkdir(parents=True, exist_ok=True)
                target = SELECTED / f"{source_name}_{identifier}{ext}"
                raw = source_path.read_bytes()
                if target.is_file() and target.read_bytes() != raw:
                    raise ValueError(f"Selected source changed: {target}")
                if not target.is_file():
                    target.write_bytes(raw)
                yield {
                    "class": class_name, "usage": usage, "source": source_name,
                    "source_index": index, "source_id": identifier,
                    "page_url": item["page_url"], "image_url": item["image_url"],
                    "license": item["license"], "artist": author,
                    "attribution": item.get("attribution", item.get("artist", "")),
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "path": target.relative_to(ROOT).as_posix(),
                    "image_sha256": pixel_digest(target),
                    "dhash": difference_hash(target),
                }
        if train_authors & holdout_authors:
            raise ValueError(f"Photographer leaked into holdout: {source_name} {train_authors & holdout_authors}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_manifest_path = BASE / "split_manifest.csv"
    base_info = json.loads((BASE / "dataset_info.json").read_text(encoding="utf-8"))
    base_sha = hashlib.sha256(base_manifest_path.read_bytes()).hexdigest()
    if base_sha != base_info["split_sha256"]:
        raise ValueError("Base training manifest changed")
    base = pd.read_csv(base_manifest_path)
    candidates = list(selected_rows())
    pixel_hashes = set(base.image_sha256)
    visual_hashes = [difference_hash(ROOT / path) for path in base.path]
    accepted = []
    for row in candidates:
        closest = min((row["dhash"] ^ earlier).bit_count() for earlier in visual_hashes)
        row["closest_dhash_distance"] = closest
        row["status"] = "accepted"
        if row["image_sha256"] in pixel_hashes:
            row["status"] = "exact_duplicate"
        elif closest <= 4:
            row["status"] = "near_duplicate_review_needed"
        if row["status"] == "accepted":
            visual_hashes.append(row["dhash"])
            pixel_hashes.add(row["image_sha256"])
            accepted.append(row)
    audit = pd.DataFrame(candidates).drop(columns="dhash")
    audit.to_csv(OUTPUT / "source_audit.csv", index=False, encoding="utf-8-sig")
    train = [row for row in accepted if row["usage"] == "train"]
    holdout = [row for row in accepted if row["usage"] == "external_holdout"]
    if len(train) < 20 or len(holdout) < 20:
        raise ValueError(f"Too few accepted photos: train={len(train)} holdout={len(holdout)}")
    new_rows = [{
        "path": row["path"], "class": row["class"],
        "label": base_info["class_names"].index(row["class"]),
        "split": "train", "source": row["source"],
        "image_sha256": row["image_sha256"], "source_label": float("nan"),
    } for row in train]
    manifest = pd.concat([base, pd.DataFrame(new_rows, columns=base.columns)], ignore_index=True)
    if manifest.image_sha256.duplicated().any() or manifest.path.duplicated().any():
        raise ValueError("Duplicate training images")
    for split in ("val", "test"):
        pd.testing.assert_frame_equal(base[base.split == split].reset_index(drop=True),
                                      manifest[manifest.split == split].reset_index(drop=True))
    manifest_path = OUTPUT / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: row[k] for k in (
        "path", "class", "image_sha256", "source", "source_id", "page_url",
        "license", "artist", "file_sha256"
    )} for row in holdout]).to_csv(OUTPUT / "external_holdout.csv", index=False, encoding="utf-8-sig")
    counts = manifest.groupby(["class", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=base_info["class_names"], columns=["train", "val", "test"])
    counts.to_csv(OUTPUT / "split_counts.csv", index_label="class", encoding="utf-8-sig")
    info = dict(base_info)
    info.update({
        "split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "base_split_sha256": json.loads((ROOT / "outputs" / "flower25" / "dataset_info.json").read_text(encoding="utf-8"))["split_sha256"],
        "parent_split_sha256": base_sha,
        "images": len(manifest),
        "added_targeted_training_images": len(train),
        "external_holdout_images": len(holdout),
        "external_holdout_sha256": hashlib.sha256((OUTPUT / "external_holdout.csv").read_bytes()).hexdigest(),
        "split_policy": "Original validation/test unchanged; reviewed rose/carnation images added only to train; separate photographer-disjoint external holdout frozen before model inspection.",
        "near_duplicate_policy": "64-bit difference hash <=4 excluded across original data and all new photos; this is a screening method, not a guarantee.",
    })
    (OUTPUT / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Accepted training:", pd.Series([r["class"] for r in train]).value_counts().to_dict())
    print("Frozen external holdout:", pd.Series([r["class"] for r in holdout]).value_counts().to_dict())
    print("Rejected:", audit.status.value_counts().to_dict())
    print("Training manifest:", manifest_path)


if __name__ == "__main__":
    main()
