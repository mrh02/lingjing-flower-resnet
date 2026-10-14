"""从人工复核清单生成独立的训练集清洗实验，不改原始清单。"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source_dir, review_path, output_dir):
    source_manifest = source_dir / "split_manifest.csv"
    source_info = json.loads((source_dir / "dataset_info.json").read_text(encoding="utf-8"))
    if source_info["split_sha256"] != sha256(source_manifest):
        raise ValueError("原始划分清单与元数据校验值不一致")
    original = pd.read_csv(source_manifest)
    reviews = pd.read_csv(review_path)
    if reviews["path"].duplicated().any():
        raise ValueError("人工复核清单存在重复路径")
    known_status = {"exclude_train_experiment", "keep_train", "val_flag_no_change"}
    if not set(reviews["status"]).issubset(known_status):
        raise ValueError("人工复核清单存在未知决定")
    if not set(reviews["path"]).issubset(set(original["path"])):
        raise ValueError("人工复核清单含有原始数据清单之外的图片")
    excluded = reviews[reviews["status"] == "exclude_train_experiment"]
    if excluded.empty:
        raise ValueError("没有指定要在训练实验中排除的图片")
    matched = original[original["path"].isin(excluded["path"])]
    if len(matched) != len(excluded) or not (
        (matched["class"] == "roses") & (matched["split"] == "train")
    ).all():
        raise ValueError("只允许排除训练集里的玫瑰图片")

    cleaned = original[~original["path"].isin(excluded["path"])].copy()
    fixed = ["val", "test"]
    if not original[original["split"].isin(fixed)].reset_index(drop=True).equals(
        cleaned[cleaned["split"].isin(fixed)].reset_index(drop=True)
    ):
        raise ValueError("验证集或测试集被修改")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_manifest = output_dir / "split_manifest.csv"
    cleaned.to_csv(cleaned_manifest, index=False, encoding="utf-8-sig")
    info = dict(source_info)
    info["split_sha256"] = sha256(cleaned_manifest)
    info["images"] = len(cleaned)
    info["clean_train_experiment"] = {
        "source_manifest_sha256": sha256(source_manifest),
        "review_csv_sha256": sha256(review_path),
        "excluded_train_roses": len(excluded),
        "train_before": int((original["split"] == "train").sum()),
        "train_after": int((cleaned["split"] == "train").sum()),
        "val_unchanged": int((cleaned["split"] == "val").sum()),
        "test_unchanged": int((cleaned["split"] == "test").sum()),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    removed = matched.merge(
        excluded[["path", "observation", "status"]], on="path", validate="one_to_one")
    removed.to_csv(output_dir / "removed_train_rows.csv", index=False,
                   encoding="utf-8-sig")
    cleaned.groupby(["class", "split"]).size().unstack(fill_value=0).to_csv(
        output_dir / "split_counts.csv", encoding="utf-8-sig")
    print(f"仅训练集排除玫瑰图片：{len(excluded)} 张")
    print(f"训练图片：{info['clean_train_experiment']['train_before']} → "
          f"{info['clean_train_experiment']['train_after']} 张")
    print(f"验证集保持 {info['clean_train_experiment']['val_unchanged']} 张；"
          f"测试集保持 {info['clean_train_experiment']['test_unchanged']} 张")
    print("独立数据清单：", cleaned_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "outputs/flower25")
    parser.add_argument("--review", type=Path,
                        default=ROOT / "outputs/flower25/label_audit/review_candidates.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/flower25_clean_train_v1")
    args = parser.parse_args()
    prepare(args.source_dir.resolve(), args.review.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"准备训练清单失败：{error}")
