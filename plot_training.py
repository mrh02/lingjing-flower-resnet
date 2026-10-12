from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent

# 可以传入本次训练目录；不传参数时查看最初三轮。
parser = argparse.ArgumentParser(description="绘制指定训练目录中的曲线")
parser.add_argument(
    "run_dir", nargs="?", type=Path,
    default=project_dir / "outputs" / "train_20261012_102328",
)
args = parser.parse_args()
run_dir = args.run_dir
if not run_dir.is_absolute():
    run_dir = project_dir / run_dir
if not (run_dir / "history.csv").is_file():
    parser.error(f"找不到训练记录：{run_dir / 'history.csv'}")
df = pd.read_csv(run_dir / "history.csv")

# 一张图中放两个子图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：损失
axes[0].plot(
    df["epoch"], df["train_loss"], marker="o", label="Train"
)
axes[0].plot(
    df["epoch"], df["val_loss"], marker="o", label="Validation"
)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# 右图：准确率，转换成百分数
axes[1].plot(
    df["epoch"], df["train_accuracy"] * 100,
    marker="o", label="Train",
)
axes[1].plot(
    df["epoch"], df["val_accuracy"] * 100,
    marker="o", label="Validation",
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 100)

for ax in axes:
    ax.set_xticks(df["epoch"])
    ax.legend()
    ax.grid(alpha=0.3)

fig.tight_layout()

chart_path = run_dir / "training_curves.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("训练曲线已保存到：", chart_path)
