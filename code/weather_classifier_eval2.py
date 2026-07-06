# weather_classifier_eval.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipForImageClassification
from sklearn.metrics import classification_report, confusion_matrix


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_dataset(clear_dir: Path, heavy_rain_dir: Path) -> List[Tuple[Path, str]]:
    items = []

    for p in list_images(clear_dir):
        items.append((p, "clear"))

    for p in list_images(heavy_rain_dir):
        items.append((p, "heavy_rain"))

    return items


@torch.no_grad()
def predict_batch(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    pred_ids = probs.argmax(dim=-1).cpu().tolist()
    pred_labels = [model.config.id2label[i] for i in pred_ids]
    pred_confs = probs.max(dim=-1).values.cpu().tolist()

    return pred_labels, pred_confs, probs.cpu()


def map_weather_to_binary(label: str) -> str:
    """
    モデルの天候ラベルを、今回のCARLA評価用に clear / heavy_rain 側へ寄せる。
    モデルのid2labelを見て必要に応じて調整してください。
    """
    s = label.lower()

    if "rain" in s:
        return "heavy_rain"

    if "clear" in s or "sun" in s or "shine" in s:
        return "clear"

    # cloudy/overcast は clear ではないが、豪雨でもない。
    # 2値評価ではいったん clear 側に倒すか、unknownとして除外する。
    if "cloud" in s or "overcast" in s:
        return "clear"

    # foggy/hazy は豪雨ではないが、悪条件ではある。
    # Alpamayo用途なら bad_weather に入れてもよい。
    if "fog" in s or "hazy" in s:
        return "heavy_rain"

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-dir", type=Path, required=True)
    parser.add_argument("--heavy-rain-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="prithivMLmods/Weather-Image-Classification",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=Path("weather_classifier_results.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model: {args.model_name}")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = SiglipForImageClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    print("id2label:")
    print(model.config.id2label)

    items = build_dataset(args.clear_dir, args.heavy_rain_dir)
    if not items:
        raise RuntimeError("No images found. Please check input directories.")

    rows = []
    y_true = []
    y_pred = []

    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    for start in tqdm(range(0, len(items), args.batch_size)):
        batch_items = items[start:start + args.batch_size]
        paths = [x[0] for x in batch_items]
        true_binary = [x[1] for x in batch_items]
        images = [load_image(p) for p in paths]

        pred_labels, pred_confs, probs = predict_batch(
            model=model,
            processor=processor,
            images=images,
            device=device,
        )

        for i, path in enumerate(paths):
            pred_binary = map_weather_to_binary(pred_labels[i])

            row = {
                "path": str(path),
                "true_binary": true_binary[i],
                "pred_weather_label": pred_labels[i],
                "pred_binary": pred_binary,
                "confidence": float(pred_confs[i]),
                "correct_binary": true_binary[i] == pred_binary,
            }

            for j, label in enumerate(labels):
                safe_label = (
                    label.replace("/", "_")
                    .replace(" ", "_")
                    .replace("-", "_")
                    .lower()
                )
                row[f"prob_{safe_label}"] = float(probs[i, j].item())

            rows.append(row)

            if pred_binary != "unknown":
                y_true.append(true_binary[i])
                y_pred.append(pred_binary)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\nSaved:", args.out_csv)

    if y_true:
        print("\n=== binary classification report ===")
        print(classification_report(y_true, y_pred, labels=["clear", "heavy_rain"], digits=4))

        print("\n=== binary confusion matrix ===")
        cm = confusion_matrix(y_true, y_pred, labels=["clear", "heavy_rain"])
        print(pd.DataFrame(
            cm,
            index=["true_clear", "true_heavy_rain"],
            columns=["pred_clear", "pred_heavy_rain"],
        ))

    print("\n=== sample predictions ===")
    print(df[[
        "path",
        "true_binary",
        "pred_weather_label",
        "pred_binary",
        "confidence",
        "correct_binary",
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
