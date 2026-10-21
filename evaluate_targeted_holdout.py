"""Compare handwritten 25-class checkpoints on a frozen external two-class set."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from resnet import FlowerResNet


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "flower25_targeted_v2"


def evaluate(checkpoint_path, rows, device, transform, classes):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("class_names") != classes or checkpoint.get("selection_metric") != "val_macro_recall":
        raise ValueError(f"Checkpoint class order/type mismatch: {checkpoint_path}")
    model = FlowerResNet(num_classes=25).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions = []
    with torch.inference_mode():
        for row in rows.to_dict("records"):
            with Image.open(ROOT / row["path"]) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            scores = torch.softmax(model(tensor)[0], dim=0)
            top = int(scores.argmax().item())
            true = classes.index(row["class"])
            predictions.append({
                "path": row["path"], "true_class": classes[true],
                "predicted_class": classes[top], "correct": top == true,
                "true_class_score": round(float(scores[true]), 6),
                "predicted_score": round(float(scores[top]), 6),
            })
    pred = pd.DataFrame(predictions)
    counts = pred.groupby("true_class")["correct"].agg(["sum", "count"])
    report = {
        "checkpoint": str(checkpoint_path), "epoch": checkpoint.get("epoch"),
        "holdout_sha256": hashlib.sha256((DATA / "external_holdout.csv").read_bytes()).hexdigest(),
        "correct": int(pred.correct.sum()), "total": len(pred),
        "accuracy": float(pred.correct.mean()),
        "per_class": {name: {"correct": int(row["sum"]), "total": int(row["count"]),
                             "recall": float(row["sum"] / row["count"])}
                      for name, row in counts.iterrows()},
    }
    return report, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    info = json.loads((DATA / "dataset_info.json").read_text(encoding="utf-8"))
    holdout_path = DATA / "external_holdout.csv"
    if hashlib.sha256(holdout_path.read_bytes()).hexdigest() != info["external_holdout_sha256"]:
        raise ValueError("Frozen holdout changed")
    rows = pd.read_csv(holdout_path)
    if set(rows["class"]) != {"roses", "carnations"} or rows.path.duplicated().any():
        raise ValueError("Unexpected holdout composition")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    results = []
    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.resolve()
        report, predictions = evaluate(checkpoint_path, rows, device, transform, info["class_names"])
        results.append(report)
        predictions.to_csv(DATA / f"external_predictions_{checkpoint_path.parent.name}.csv", index=False, encoding="utf-8-sig")
        print(checkpoint_path.parent.name, f"{report['correct']}/{report['total']} = {report['accuracy']:.1%}",
              {name: f"{value['correct']}/{value['total']}" for name, value in report["per_class"].items()}, flush=True)
    (DATA / "external_holdout_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
