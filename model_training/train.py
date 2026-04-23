import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from .config import METRICS_DIR, MODELS_DIR, PLOTS_DIR, SPLITS_DIR, SUPPORTED_MODELS
from .data_utils import create_dataloader
from .modeling import create_transfer_model
from .plotting import plot_training_curves
from .utils import ensure_dir, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train transfer-learning model for eye state.")
    parser.add_argument("--train-csv", type=Path, default=SPLITS_DIR / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=SPLITS_DIR / "val.csv")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=SUPPORTED_MODELS,
        help="Backbone architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--finetune", action="store_true", help="Train all layers instead of head only.")
    parser.add_argument(
        "--augment-data",
        action="store_true",
        help="Enable training-time data augmentation for the train split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to outputs/models/<model>_best.pt or <model>_aug_best.pt",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="History CSV path. Defaults to outputs/metrics/<model>_history.csv or <model>_aug_history.csv",
    )
    parser.add_argument(
        "--curves-path",
        type=Path,
        default=None,
        help="Curves image path. Defaults to outputs/plots/<model>_training_curves.png or <model>_aug_training_curves.png",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_output_paths(args):
    suffix = "_aug" if args.augment_data else ""
    checkpoint_path = args.checkpoint or (MODELS_DIR / f"{args.model}{suffix}_best.pt")
    history_csv_path = args.history_csv or (METRICS_DIR / f"{args.model}{suffix}_history.csv")
    curves_path = args.curves_path or (PLOTS_DIR / f"{args.model}{suffix}_training_curves.png")
    return checkpoint_path, history_csv_path, curves_path


def main():
    args = parse_args()
    set_seed(args.seed)

    for csv_path in [args.train_csv, args.val_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}. Run prepare_data first.")

    checkpoint_path, history_csv_path, curves_path = resolve_output_paths(args)

    ensure_dir(checkpoint_path.parent)
    ensure_dir(history_csv_path.parent)
    ensure_dir(curves_path.parent)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Data augmentation: {args.augment_data}")

    train_loader = create_dataloader(
        csv_path=str(args.train_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        augment=args.augment_data,
        shuffle=True,
    )
    val_loader = create_dataloader(
        csv_path=str(args.val_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        augment=False,
        shuffle=False,
    )

    model = create_transfer_model(
        model_name=args.model,
        num_classes=2,
        pretrained=True,
        freeze_backbone=not args.finetune,
    ).to(device)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters were found.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "finetune": args.finetune,
                "model_name": args.model,
                "augment_data": args.augment_data,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    plot_training_curves(history, str(curves_path))
    print(f"Training curves saved to {curves_path}")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
