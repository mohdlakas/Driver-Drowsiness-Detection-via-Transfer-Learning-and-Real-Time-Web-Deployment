import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import METRICS_DIR, PLOTS_DIR, SUPPORTED_MODELS
from .utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across trained models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=SUPPORTED_MODELS,
        help="Models to compare.",
    )
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--output-csv", type=Path, default=METRICS_DIR / "compare_summary.csv")
    parser.add_argument(
        "--augmentation-report-csv",
        type=Path,
        default=METRICS_DIR / "augmentation_pairwise_report.csv",
    )
    parser.add_argument("--plot-path", type=Path, default=PLOTS_DIR / "model_comparison.png")
    parser.add_argument("--augmentation-plot-path", type=Path, default=PLOTS_DIR / "augmentation_delta.png")
    parser.add_argument("--sort-by", type=str, default="f1_closed", choices=["accuracy", "f1_closed"])
    return parser.parse_args()


def load_metric_file(metrics_path: Path):
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_metric_path(metrics_dir: Path, model_name: str, augmented: bool) -> Path | None:
    if augmented:
        candidates = [metrics_dir / f"{model_name}_aug_test_metrics.json"]
    else:
        candidates = [
            metrics_dir / f"{model_name}_test_metrics.json",
            metrics_dir / f"{model_name}_base_test_metrics.json",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_rows(metrics_dir: Path, models: list[str]):
    rows = []
    missing = []

    for model_name in models:
        for variant, augmented in [("base", False), ("augmented", True)]:
            metrics_path = resolve_metric_path(metrics_dir=metrics_dir, model_name=model_name, augmented=augmented)
            if metrics_path is None:
                missing.append(f"{model_name}:{variant}")
                continue

            metrics = load_metric_file(metrics_path)
            rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "augment_data": bool(augmented),
                    "accuracy": float(metrics.get("accuracy", float("nan"))),
                    "precision_closed": float(metrics.get("precision_closed", float("nan"))),
                    "recall_closed": float(metrics.get("recall_closed", float("nan"))),
                    "f1_closed": float(metrics.get("f1_closed", float("nan"))),
                    "metrics_path": str(metrics_path),
                }
            )
    return rows, missing


def build_pairwise_augmentation_report(df: pd.DataFrame) -> pd.DataFrame:
    pair_rows = []
    for model_name in sorted(df["model"].unique()):
        sub = df[df["model"] == model_name]
        if not {"base", "augmented"}.issubset(set(sub["variant"].tolist())):
            continue

        base_row = sub[sub["variant"] == "base"].iloc[0]
        aug_row = sub[sub["variant"] == "augmented"].iloc[0]

        pair_rows.append(
            {
                "model": model_name,
                "accuracy_base": float(base_row["accuracy"]),
                "accuracy_augmented": float(aug_row["accuracy"]),
                "accuracy_delta": float(aug_row["accuracy"] - base_row["accuracy"]),
                "f1_closed_base": float(base_row["f1_closed"]),
                "f1_closed_augmented": float(aug_row["f1_closed"]),
                "f1_closed_delta": float(aug_row["f1_closed"] - base_row["f1_closed"]),
                "precision_closed_base": float(base_row["precision_closed"]),
                "precision_closed_augmented": float(aug_row["precision_closed"]),
                "precision_closed_delta": float(aug_row["precision_closed"] - base_row["precision_closed"]),
                "recall_closed_base": float(base_row["recall_closed"]),
                "recall_closed_augmented": float(aug_row["recall_closed"]),
                "recall_closed_delta": float(aug_row["recall_closed"] - base_row["recall_closed"]),
            }
        )

    if not pair_rows:
        return pd.DataFrame()

    return pd.DataFrame(pair_rows).sort_values(by="f1_closed_delta", ascending=False).reset_index(drop=True)


def plot_all_runs(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    labels = [f"{row.model}\n{row.variant}" for row in df.itertuples(index=False)]
    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], df["accuracy"], width=width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], df["f1_closed"], width=width, label="F1 (closed)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Base vs Augmented)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_augmentation_delta(pair_df: pd.DataFrame, output_path: Path) -> None:
    if pair_df.empty:
        return

    ensure_dir(output_path.parent)

    x = range(len(pair_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], pair_df["accuracy_delta"], width=width, label="Accuracy Delta")
    ax.bar([i + width / 2 for i in x], pair_df["f1_closed_delta"], width=width, label="F1 (closed) Delta")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pair_df["model"].tolist())
    ax.set_ylabel("Delta (Augmented - Base)")
    ax.set_title("Augmentation Gain by Model")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    rows, missing = collect_rows(metrics_dir=args.metrics_dir, models=args.models)
    if not rows:
        raise FileNotFoundError(
            "No evaluation metrics found. Run evaluation first, e.g.:\n"
            "python -m src.evaluate --model resnet18\n"
            "python -m src.evaluate --model resnet18 --augment-data\n"
            "python -m src.evaluate --model mobilenet_v3_small\n"
            "python -m src.evaluate --model mobilenet_v3_small --augment-data\n"
            "python -m src.evaluate --model efficientnet_b0\n"
            "python -m src.evaluate --model efficientnet_b0 --augment-data"
        )

    df = pd.DataFrame(rows).sort_values(by=args.sort_by, ascending=False).reset_index(drop=True)

    ensure_dir(args.output_csv.parent)
    df.to_csv(args.output_csv, index=False)
    plot_all_runs(df=df, output_path=args.plot_path)

    pair_df = build_pairwise_augmentation_report(df=df)
    ensure_dir(args.augmentation_report_csv.parent)
    if pair_df.empty:
        empty_cols = [
            "model",
            "accuracy_base",
            "accuracy_augmented",
            "accuracy_delta",
            "f1_closed_base",
            "f1_closed_augmented",
            "f1_closed_delta",
            "precision_closed_base",
            "precision_closed_augmented",
            "precision_closed_delta",
            "recall_closed_base",
            "recall_closed_augmented",
            "recall_closed_delta",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(args.augmentation_report_csv, index=False)
    else:
        pair_df.to_csv(args.augmentation_report_csv, index=False)
        plot_augmentation_delta(pair_df=pair_df, output_path=args.augmentation_plot_path)

    print("All runs summary:")
    print(df[["model", "variant", "accuracy", "f1_closed", "precision_closed", "recall_closed"]].to_string(index=False))
    print(f"Saved comparison CSV to: {args.output_csv}")
    print(f"Saved comparison plot to: {args.plot_path}")
    print(f"Saved augmentation pairwise CSV to: {args.augmentation_report_csv}")
    if not pair_df.empty:
        print(f"Saved augmentation delta plot to: {args.augmentation_plot_path}")
    else:
        print("Augmentation pairwise report is empty (need both base and augmented metrics per model).")

    if missing:
        print("Missing metric variants:")
        for item in sorted(set(missing)):
            print(f"- {item}")


if __name__ == "__main__":
    main()
