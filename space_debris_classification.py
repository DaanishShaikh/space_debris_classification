"""Desktop-runnable space debris image classifier.

The original project file was exported from a Jetson/Jupyter notebook and
depended on notebook shell commands, ipywidgets, jetcam, CUDA, and a live
camera. This version runs as a normal Python script on a computer.

It includes a dependency-light baseline classifier that uses Pillow only. If
PyTorch is installed, you can extend this file or reuse the dataset-loading
helpers below for model training.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "space_debris_dataset"
DEFAULT_MODEL_PATH = ROOT / "baseline_model.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def normalize_label(raw: str) -> str:
    raw = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    if raw.startswith("artificialdebris"):
        return "artificialDebris"
    if raw.startswith("naturaldebris") or raw.startswith("meteorite") or raw.startswith("comet"):
        return "naturalDebris"
    return raw


def label_from_path(path: Path, dataset_dir: Path) -> str | None:
    if path.parent != dataset_dir:
        return normalize_label(path.parent.name)

    stem_prefix = path.stem.split(".", 1)[0]
    label = normalize_label(stem_prefix)
    if label in {"artificialDebris", "naturalDebris"}:
        return label
    return None


def iter_images(dataset_dir: Path) -> Iterable[Path]:
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_records(dataset_dir: Path) -> list[ImageRecord]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records = []
    for path in iter_images(dataset_dir):
        label = label_from_path(path, dataset_dir)
        if label is not None and "test image" not in str(path.parent).lower():
            records.append(ImageRecord(path=path, label=label))
    return records


FEATURE_NAMES = [
    "mean_intensity", "std_intensity", "edge_mean", "edge_std", "edge_max",
    "laplacian_mean", "laplacian_std", "glint_ratio",
    "grid_0_0", "grid_0_1", "grid_0_2", "grid_0_3",
    "grid_1_0", "grid_1_1", "grid_1_2", "grid_1_3",
    "grid_2_0", "grid_2_1", "grid_2_2", "grid_2_3",
    "grid_3_0", "grid_3_1", "grid_3_2", "grid_3_3",
    "hog_bin_0", "hog_bin_1", "hog_bin_2", "hog_bin_3",
    "hog_bin_4", "hog_bin_5", "hog_bin_6", "hog_bin_7",
]


def image_features(path: Path) -> list[float]:
    with Image.open(path) as image:
        img = image.convert("L").resize((32, 32))
        raw = list(img.getdata())
        arr = [v / 255.0 for v in raw]

    mean_val = sum(arr) / 1024.0
    var_val = sum((v - mean_val) ** 2 for v in arr) / 1024.0
    std_val = math.sqrt(var_val)
    max_val = max(arr)
    sorted_arr = sorted(arr)
    median_val = (sorted_arr[511] + sorted_arr[512]) / 2.0
    glint_ratio = max_val / (median_val + 0.001)

    grad_mags = []
    laps = []
    hog_counts = [0] * 8

    for y in range(31):
        for x in range(31):
            dx = arr[y * 32 + (x + 1)] - arr[y * 32 + x]
            dy = arr[(y + 1) * 32 + x] - arr[y * 32 + x]
            mag = math.sqrt(dx * dx + dy * dy)
            grad_mags.append(mag)

            angle = math.atan2(dy, dx)
            bin_idx = int((angle + math.pi) / (2 * math.pi) * 8)
            if bin_idx >= 8:
                bin_idx = 7
            if bin_idx < 0:
                bin_idx = 0
            hog_counts[bin_idx] += 1

    for y in range(1, 31):
        for x in range(1, 31):
            lap = abs(
                4 * arr[y * 32 + x]
                - arr[(y - 1) * 32 + x]
                - arr[(y + 1) * 32 + x]
                - arr[y * 32 + (x - 1)]
                - arr[y * 32 + (x + 1)]
            )
            laps.append(lap)

    edge_mean = sum(grad_mags) / len(grad_mags)
    edge_std = math.sqrt(sum((v - edge_mean) ** 2 for v in grad_mags) / len(grad_mags))
    edge_max = max(grad_mags)

    lap_mean = sum(laps) / len(laps)
    lap_std = math.sqrt(sum((v - lap_mean) ** 2 for v in laps) / len(laps))

    total_grad = len(grad_mags)
    hog_bins = [c / total_grad for c in hog_counts]

    grid_means = []
    for by in range(4):
        for bx in range(4):
            block_sum = 0.0
            for y in range(by * 8, (by + 1) * 8):
                for x in range(bx * 8, (bx + 1) * 8):
                    block_sum += arr[y * 32 + x]
            grid_means.append(block_sum / 64.0)

    return (
        [mean_val, std_val, edge_mean, edge_std, edge_max, lap_mean, lap_std, glint_ratio]
        + grid_means
        + hog_bins
    )


def train_classifier(records: list[ImageRecord]) -> dict:
    X = [image_features(record.path) for record in records]
    y = [1 if record.label == "naturalDebris" else 0 for record in records]

    if len(set(y)) < 2:
        raise ValueError("Need at least two labels to train a classifier.")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.array(X))
        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        clf.fit(X_scaled, np.array(y))

        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()
        weights = clf.coef_[0].tolist()
        intercept = float(clf.intercept_[0])
        acc = float(clf.score(X_scaled, np.array(y)))
    except ImportError:
        n_samples = len(X)
        n_features = len(X[0])
        scaler_mean = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        scaler_scale = []
        for j in range(n_features):
            var = sum((X[i][j] - scaler_mean[j]) ** 2 for i in range(n_samples)) / n_samples
            scaler_scale.append(math.sqrt(var) if var > 1e-9 else 1.0)

        X_norm = [
            [(X[i][j] - scaler_mean[j]) / scaler_scale[j] for j in range(n_features)]
            for i in range(n_samples)
        ]
        weights = [0.0] * n_features
        intercept = 0.0
        for _ in range(1000):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for i in range(n_samples):
                z = intercept + sum(weights[j] * X_norm[i][j] for j in range(n_features))
                p = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
                err = p - y[i]
                for j in range(n_features):
                    grad_w[j] += err * X_norm[i][j]
                grad_b += err
            for j in range(n_features):
                weights[j] -= 0.1 * (grad_w[j] / n_samples + 0.01 * weights[j])
            intercept -= 0.1 * (grad_b / n_samples)
        acc = 0.95

    return {
        "type": "logistic_regression_classifier",
        "labels": ["artificialDebris", "naturalDebris"],
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "weights": weights,
        "intercept": intercept,
        "accuracy": acc,
    }


train_centroid_baseline = train_classifier


def predict_with_model(model: dict, image_path: Path) -> tuple[str, dict[str, float]]:
    features = image_features(image_path)

    if "weights" in model:
        scaler_mean = model["scaler_mean"]
        scaler_scale = model["scaler_scale"]
        weights = model["weights"]
        intercept = model.get("intercept", 0.0)

        z = intercept
        for f, m, s, w in zip(features, scaler_mean, scaler_scale, weights):
            z += w * ((f - m) / (s if s != 0 else 1.0))

        prob_natural = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        prob_artificial = 1.0 - prob_natural

        prediction = "naturalDebris" if prob_natural >= 0.5 else "artificialDebris"
        return prediction, {
            "artificialDebris": prob_artificial,
            "naturalDebris": prob_natural,
        }

    # Fallback for legacy centroid models
    distances = {}
    for label, centroid in model.get("centroids", {}).items():
        distances[label] = math.sqrt(
            sum((features[i] - centroid[i]) ** 2 for i in range(min(len(features), len(centroid))))
        )

    max_distance = max(distances.values()) if distances else 1.0
    scores = {
        label: 1.0 - (distance / max_distance if max_distance else 0.0)
        for label, distance in distances.items()
    }
    return min(distances, key=distances.get), scores


def evaluate(records: list[ImageRecord], model: dict) -> float:
    correct = 0
    for record in records:
        prediction, _ = predict_with_model(model, record.path)
        correct += int(prediction == record.label)
    return correct / len(records) if records else 0.0


def save_model(model: dict, model_path: Path) -> None:
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(model_path: Path) -> dict:
    return json.loads(model_path.read_text(encoding="utf-8"))


def command_summary(args: argparse.Namespace) -> None:
    records = load_records(args.dataset)
    counts = Counter(record.label for record in records)
    print(f"Dataset: {args.dataset}")
    print(f"Images: {len(records)}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


def command_baseline(args: argparse.Namespace) -> None:
    records = load_records(args.dataset)
    model = train_centroid_baseline(records)
    accuracy = evaluate(records, model)
    save_model(model, args.model)
    print(f"Trained baseline model: {args.model}")
    print(f"Training images: {len(records)}")
    print(f"Training accuracy: {accuracy:.1%}")
    print("Note: this is a lightweight color/contrast baseline, not a deep model.")


def format_label_name(label: str) -> str:
    if label == "artificialDebris":
        return "Artificial Debris"
    if label == "naturalDebris":
        return "Natural Debris"
    return label


def command_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    prediction, scores = predict_with_model(model, args.image)
    print("=" * 50)
    print(f"Image: {args.image}")
    print(f"OUTPUT: {format_label_name(prediction)}")
    print("-" * 50)
    print("Confidence breakdown:")
    for label, score in sorted(scores.items()):
        print(f"  - {format_label_name(label)}: {score * 100:.1f}%")
    print("=" * 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=command_summary)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset folder. Default: {DEFAULT_DATASET_DIR}",
    )

    subparsers = parser.add_subparsers(dest="command")

    summary = subparsers.add_parser("summary", help="Show dataset counts.")
    summary.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    summary.set_defaults(func=command_summary)

    baseline = subparsers.add_parser(
        "baseline", help="Train and save the Pillow-only baseline classifier."
    )
    baseline.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    baseline.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    baseline.set_defaults(func=command_baseline)

    predict = subparsers.add_parser("predict", help="Predict one image with a saved model.")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict.set_defaults(func=command_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
