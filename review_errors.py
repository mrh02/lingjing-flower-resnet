"""从已有测试预测清单中选取错分样例，不再次运行模型。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent


def review_errors(results_dir):
    predictions = pd.read_csv(results_dir / "test_predictions.csv")
    # 保留真实类别与预测类别不同的图片。
    errors = predictions[
        predictions["true_class"] != predictions["predicted_class"]
    ]
    if errors.empty:
        print("清单中没有错分图片。")
        return

    # 按错误数量排序；并列时按类别名排序，保证选取规则固定。
    pairs = (
        errors.groupby(["true_class", "predicted_class"])
        .size().reset_index(name="error_count")
        .sort_values(
            ["error_count", "true_class", "predicted_class"],
            ascending=[False, True, True],
        ).head(3)
    )
    samples = []
    for pair in pairs.itertuples(index=False):
        matching = errors[
            (errors["true_class"] == pair.true_class)
            & (errors["predicted_class"] == pair.predicted_class)
        ].head(2).copy()
        matching["pair_error_count"] = pair.error_count
        samples.append(matching)
    selected = pd.concat(samples, ignore_index=True)

    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 4 * len(samples)),
                             squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for row_index, sample_group in enumerate(samples):
        for col_index, sample in enumerate(sample_group.itertuples(index=False)):
            axis = axes[row_index, col_index]
            with Image.open(PROJECT_DIR / sample.path) as source:
                axis.imshow(source.convert("RGB"))
            axis.set_title(
                f"True: {sample.true_class} | Predicted: {sample.predicted_class}\n"
                f"Softmax score: {sample.softmax_score:.1%}", fontsize=11,
            )

    fig.suptitle("Examples from the three most frequent error types", fontsize=15)
    fig.text(0.5, 0.015,
             "First two images per error type in CSV order; examples are not a random sample.\n"
             "Softmax scores do not measure accuracy.",
             ha="center", fontsize=9)
    # 图片宽高比不同，预留行间距，避免两行标题盖住上一行图片。
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.075, top=0.90,
                        hspace=0.35, wspace=0.06)
    chart_path = results_dir / "error_examples.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    selected.to_csv(results_dir / "error_examples.csv", index=False)

    print("总错分图片数：", len(errors))
    print("选择规则：数量最多的三种错误，每种取清单中的前两张。")
    print("样例用于解释错误，不代表全部测试图片的分布。")
    for pair in pairs.itertuples(index=False):
        print(f"  {pair.true_class} → {pair.predicted_class}：{pair.error_count} 张")
    print("错分图片对照图：", chart_path)
    print("所选图片清单：", results_dir / "error_examples.csv")


def main():
    parser = argparse.ArgumentParser(description="查看已有测试结果中的错分图片")
    parser.add_argument("results_dir", type=Path, help="包含 test_predictions.csv 的目录")
    args = parser.parse_args()
    try:
        review_errors(args.results_dir.expanduser().resolve())
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"生成失败：{error}\n")


if __name__ == "__main__":
    main()
