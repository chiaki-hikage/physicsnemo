# zero_shot_weather_eval.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# まずはこの2値で評価する
CLASS_PROMPTS: Dict[str, List[str]] = {
    "heavy_rain": [
        "a driving scene in heavy rain",
        "a road scene with intense rain and low visibility",
        "a car driving in very rainy weather",
        "a wet road scene during heavy rainfall",
    ],
    "clear": [
        "a clear weather driving scene",
        "a sunny road scene with good visibility",
        "a dry road scene in clear weather",
        "a car driving in normal clear weather",
    ],
}


def list_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


@torch.no_grad()
def compute_text_features(
    model: AutoModel,
    processor: AutoProcessor,
    prompts_by_class: Dict[str, List[str]],
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    """
    各クラスに複数promptを用意し、prompt embeddingを平均してclass embeddingにする。
    """
    class_names = list(prompts_by_class.keys())
    class_embs = []

    for cls in class_names:
        prompts = prompts_by_class[cls]
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

        if hasattr(model, "get_text_features"):
            text_features = model.get_text_features(**inputs)
        else:
            outputs = model(**inputs)
            text_features = outputs.text_embeds

        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        class_feature = text_features.mean(dim=0, keepdim=True)
        class_feature = torch.nn.functional.normalize(class_feature, dim=-1)
        class_embs.append(class_feature)

    class_embs = torch.cat(class_embs, dim=0)
    return class_names, class_embs


@torch.no_grad()
def predict_batch(
    model: AutoModel,
    processor: AutoProcessor,
    images: List[Image.Image],
    class_names: List[str],
    text_features: torch.Tensor,
    device: torch.device,
) -> Tuple[List[str], List[float], torch.Tensor]:
    inputs = processor(images=images, return_tensors="pt").to(device)

    if hasattr(model, "get_image_features"):
        image_features = model.get_image_features(**inputs)
    else:
        outputs = model(**inputs)
        image_features = outputs.image_embeds

    image_features = torch.nn.functional.normalize(image_features, dim=-1)

    # cosine similarity
    logits = image_features @ text_features.T
    probs = torch.softmax(logits, dim=-1)

    pred_idx = probs.argmax(dim=-1)
    pred_labels = [class_names[i] for i in pred_idx.cpu().tolist()]
    pred_confs = probs.max(dim=-1).values.cpu().tolist()

    return pred_labels, pred_confs, probs.cpu()


def build_dataset(clear_dir: Path, heavy_rain_dir: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []

    for p in list_images(clear_dir):
        items.append((p, "clear"))

    for p in list_images(heavy_rain_dir):
        items.append((p, "heavy_rain"))

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-dir", type=Path, required=True)
    parser.add_argument("--heavy-rain-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help=(
            "例: openai/clip-vit-base-patch32, "
            "openai/clip-vit-large-patch14, "
            "google/siglip-base-patch16-224"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=Path("zero_shot_weather_results.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model: {args.model_name}")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    class_names, text_features = compute_text_features(
        model=model,
        processor=processor,
        prompts_by_class=CLASS_PROMPTS,
        device=device,
    )

    items = build_dataset(
        clear_dir=args.clear_dir,
        heavy_rain_dir=args.heavy_rain_dir,
    )

    if not items:
        raise RuntimeError("No images found. Please check --clear-dir and --heavy-rain-dir.")

    rows = []
    y_true = []
    y_pred = []

    for start in tqdm(range(0, len(items), args.batch_size)):
        batch_items = items[start:start + args.batch_size]
        paths = [x[0] for x in batch_items]
        labels = [x[1] for x in batch_items]
        images = [load_image(p) for p in paths]

        pred_labels, pred_confs, probs = predict_batch(
            model=model,
            processor=processor,
            images=images,
            class_names=class_names,
            text_features=text_features,
            device=device,
        )

        for i, path in enumerate(paths):
            prob_dict = {
                f"prob_{cls}": float(probs[i, j].item())
                for j, cls in enumerate(class_names)
            }

            row = {
                "path": str(path),
                "true_label": labels[i],
                "pred_label": pred_labels[i],
                "confidence": pred_confs[i],
                "correct": labels[i] == pred_labels[i],
                **prob_dict,
            }
            rows.append(row)

            y_true.append(labels[i])
            y_pred.append(pred_labels[i])

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\n=== classification report ===")
    print(classification_report(y_true, y_pred, labels=class_names, digits=4))

    print("\n=== confusion matrix ===")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print(pd.DataFrame(cm, index=[f"true_{x}" for x in class_names], columns=[f"pred_{x}" for x in class_names]))

    print(f"\nSaved results to: {args.out_csv}")

    # 誤分類例を表示
    wrong = df[df["correct"] == False]
    if len(wrong) > 0:
        print("\n=== misclassified examples ===")
        print(wrong[["path", "true_label", "pred_label", "confidence", "prob_heavy_rain", "prob_clear"]].head(20).to_string(index=False))
    else:
        print("\nNo misclassified examples.")


if __name__ == "__main__":
    main()
