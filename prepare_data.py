import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import CLASS_TO_INDEX, IMAGE_EXTENSIONS, PROCESSED_DIR, RAW_DIR
from .utils import ensure_dir

SUBJECT_PATTERN = re.compile(r"^s\d+$", re.IGNORECASE)


def normalize_tokens(stem: str) -> list[str]:
    normalized = re.sub(r"[\s\-]+", "_", stem.lower())
    return [token for token in normalized.split("_") if token]


def infer_subject_id(image_path: Path, tokens: list[str]) -> str | None:
    for token in tokens:
        if SUBJECT_PATTERN.fullmatch(token):
            return token.lower()

    for part in image_path.parts:
        candidate = re.sub(r"[\s\-]+", "_", part.lower())
        if SUBJECT_PATTERN.fullmatch(candidate):
            return candidate

    return None


def infer_eye_label(image_path: Path, tokens: list[str]) -> str | None:
    eye_state = None

    # MRL filename convention commonly has eye state as the 5th token: 0=closed, 1=open.
    if len(tokens) >= 5 and tokens[4] in {"0", "1"}:
        eye_state = tokens[4]
    else:
        numeric_tokens = [token for token in tokens if token.isdigit()]
        if len(numeric_tokens) >= 5 and numeric_tokens[4] in {"0", "1"}:
            eye_state = numeric_tokens[4]

    path_lower = str(image_path).lower()
    if eye_state is None:
        if any(keyword in path_lower for keyword in ("closed", "close", "sleepy", "drowsy")):
            eye_state = "0"
        elif any(keyword in path_lower for keyword in ("open", "awake", "alert")):
            eye_state = "1"

    if eye_state == "0":
        return "closed"
    if eye_state == "1":
        return "open"
    return None


def collect_metadata(raw_dir: Path) -> pd.DataFrame:
    image_paths = [
        path
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise RuntimeError(f"No images found in {raw_dir}. Run download first.")

    rows = []
    skipped = 0
    for image_path in image_paths:
        tokens = normalize_tokens(image_path.stem)
        label = infer_eye_label(image_path, tokens)
        subject_id = infer_subject_id(image_path, tokens)

        if label is None:
            skipped += 1
            continue

        rows.append(
            {
                "filepath": str(image_path.resolve()),
                "label": label,
                "label_idx": CLASS_TO_INDEX[label],
                "subject_id": subject_id,
            }
        )

    if not rows:
        raise RuntimeError("No labeled images could be parsed from dataset.")

    df = pd.DataFrame(rows)
    print(f"Parsed images: {len(df)}")
    print(f"Skipped images (could not infer label): {skipped}")
    return df


def has_all_classes(df: pd.DataFrame) -> bool:
    return set(df["label"].unique()) == set(CLASS_TO_INDEX.keys())


def random_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_col = df["label"] if df["label"].nunique() > 1 else None

    try:
        train_val, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify_col,
        )
    except ValueError:
        train_val, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_state,
            stratify=None,
        )

    val_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify_col_tv = train_val["label"] if train_val["label"].nunique() > 1 else None
    try:
        train_df, val_df = train_test_split(
            train_val,
            test_size=val_adjusted,
            random_state=random_state,
            stratify=stratify_col_tv,
        )
    except ValueError:
        train_df, val_df = train_test_split(
            train_val,
            test_size=val_adjusted,
            random_state=random_state,
            stratify=None,
        )
    return train_df, val_df, test_df


def subject_group_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_val_idx, test_idx = next(splitter_1.split(df, groups=df["subject_id"]))
    train_val = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    val_adjusted = val_ratio / (train_ratio + val_ratio)
    splitter_2 = GroupShuffleSplit(n_splits=1, test_size=val_adjusted, random_state=random_state)
    train_idx, val_idx = next(splitter_2.split(train_val, groups=train_val["subject_id"]))
    train_df = train_val.iloc[train_idx]
    val_df = train_val.iloc[val_idx]

    return train_df, val_df, test_df


def split_dataset(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    use_subject_split = df["subject_id"].notna().all() and df["subject_id"].nunique() >= 3

    if use_subject_split:
        try:
            train_df, val_df, test_df = subject_group_split(
                df=df,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_state=random_state,
            )
            if has_all_classes(train_df) and has_all_classes(val_df) and has_all_classes(test_df):
                return train_df, val_df, test_df, True
            print("Subject split produced missing classes in at least one split. Falling back.")
        except ValueError:
            print("Subject split failed due to grouping constraints. Falling back.")

    train_df, val_df, test_df = random_split(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    return train_df, val_df, test_df, False


def write_split_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sample(frac=1.0, random_state=42).reset_index(drop=True).to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MRL eye dataset for training.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw image directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Processed directory for metadata and split CSV files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must sum to 1.0")

    ensure_dir(args.output_dir)
    splits_dir = ensure_dir(args.output_dir / "splits")

    metadata_df = collect_metadata(raw_dir=args.raw_dir)
    metadata_path = args.output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    train_df, val_df, test_df, used_subject_split = split_dataset(
        df=metadata_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    write_split_csv(train_df, splits_dir / "train.csv")
    write_split_csv(val_df, splits_dir / "val.csv")
    write_split_csv(test_df, splits_dir / "test.csv")

    summary = pd.DataFrame(
        [
            {"split": "train", "count": len(train_df)},
            {"split": "val", "count": len(val_df)},
            {"split": "test", "count": len(test_df)},
        ]
    )
    summary.to_csv(args.output_dir / "split_summary.csv", index=False)

    print(f"Metadata saved to: {metadata_path}")
    print(f"Train split: {len(train_df)}")
    print(f"Val split: {len(val_df)}")
    print(f"Test split: {len(test_df)}")
    print(f"Used subject-level split: {used_subject_split}")


if __name__ == "__main__":
    main()
