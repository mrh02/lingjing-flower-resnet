from pathlib import Path
import random
import pandas as pd

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "flower_photos"
output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 固定随机种子，让重复运行得到相同的划分
rng = random.Random(42)

class_dirs = sorted(
    folder for folder in data_dir.iterdir()
    if folder.is_dir()
)

records = []

for label, class_dir in enumerate(class_dirs):
    images = sorted(class_dir.glob("*.jpg"))
    rng.shuffle(images)

    n_train = int(len(images) * 0.70)
    n_val = int(len(images) * 0.15)

    groups = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, group in groups.items():
        for image_path in group:
            records.append({
                "path": image_path.relative_to(project_dir).as_posix(),
                "class": class_dir.name,
                "label": label,
                "split": split,
            })

df = pd.DataFrame(records)

manifest_path = output_dir / "split_manifest.csv"
df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

summary = (
    df.groupby(["class", "split"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["train", "val", "test"])
)

summary.to_csv(output_dir / "split_counts.csv", encoding="utf-8-sig")

print("各类别的划分数量：")
print(summary)
print("\n各组总数：")
print(df["split"].value_counts())
print(f"\n划分清单已保存到：{manifest_path}")