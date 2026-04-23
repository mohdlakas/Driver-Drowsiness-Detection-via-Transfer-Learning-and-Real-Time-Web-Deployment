import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from .config import CLASS_TO_INDEX, METRICS_DIR, MODELS_DIR, PLOTS_DIR, SPLITS_DIR, SUPPORTED_MODELS
from .data_utils import create_dataloader
from .modeling import create_transfer_model
from .plotting import plot_confusion_matrix
from .utils import ensure_dir


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc="Test", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained eye-state model.")
    parser.add_argument("--test-csv", type=Path, default=SPLITS_DIR / "test.csv")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=SUPPORTED_MODELS,
        help="Backbone architecture used during training.",
    )
    parser.add_argument(
        "--augment-data",
        action="store_true",
        help="Evaluate the augmentation-trained checkpoint by default path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to outputs/models/<model>_best.pt or <model>_aug_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Path to save scalar metrics JSON. Defaults to outputs/metrics/<model>_test_metrics.json or <model>_aug_test_metrics.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to save full report. Defaults to outputs/metrics/<model>_classification_report.txt or <model>_aug_classification_report.txt",
    )
    parser.add_argument(
        "--cm-path",
        type=Path,
        default=None,
        help="Path to save confusion matrix plot. Defaults to outputs/plots/<model>_confusion_matrix.png or <model>_aug_confusion_matrix.png",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_output_paths(args):
    suffix = "_aug" if args.augment_data else ""
    checkpoint_path = args.checkpoint or (MODELS_DIR / f"{args.model}{suffix}_best.pt")
    metrics_json_path = args.metrics_json or (METRICS_DIR / f"{args.model}{suffix}_test_metrics.json")
    report_path = args.report_path or (METRICS_DIR / f"{args.model}{suffix}_classification_report.txt")
    cm_path = args.cm_path or (PLOTS_DIR / f"{args.model}{suffix}_confusion_matrix.png")
    return checkpoint_path, metrics_json_path, report_path, cm_path


def main():
    args = parse_args()
    checkpoint_path, metrics_json_path, report_path, cm_path = resolve_output_paths(args)

    if not args.test_csv.exists():
        raise FileNotFoundError(f"Missing split file: {args.test_csv}. Run prepare_data first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}. Run train first.")

    ensure_dir(metrics_json_path.parent)
    ensure_dir(report_path.parent)
    ensure_dir(cm_path.parent)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Data augmentation run: {args.augment_data}")

    test_loader = create_dataloader(
        csv_path=str(args.test_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        shuffle=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    trained_model_name = checkpoint.get("model_name", args.model)
    if trained_model_name != args.model:
        raise ValueError(
            f"Checkpoint model mismatch: checkpoint has '{trained_model_name}', "
            f"but --model '{args.model}' was requested."
        )
    trained_with_augmentation = bool(checkpoint.get("augment_data", False))
    if trained_with_augmentation != args.augment_data:
        raise ValueError(
            "Checkpoint augmentation mismatch: "
            f"checkpoint has augment_data={trained_with_augmentation}, "
            f"but --augment-data was set to {args.augment_data}."
        )

    model = create_transfer_model(
        model_name=args.model,
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = run_inference(model=model, loader=test_loader, device=device)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=CLASS_TO_INDEX["closed"],
        zero_division=0,
    )

    labels = ["closed", "open"]
    cm = confusion_matrix(y_true, y_pred, labels=[CLASS_TO_INDEX["closed"], CLASS_TO_INDEX["open"]])
    report_text = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

    metrics = {
        "model": args.model,
        "augment_data": args.augment_data,
        "accuracy": float(accuracy),
        "precision_closed": float(precision),
        "recall_closed": float(recall),
        "f1_closed": float(f1),
        "support_closed": int(report_dict["closed"]["support"]),
        "support_open": int(report_dict["open"]["support"]),
    }

    with open(metrics_json_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    plot_confusion_matrix(cm=cm, labels=labels, output_path=str(cm_path))

    print("Evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
