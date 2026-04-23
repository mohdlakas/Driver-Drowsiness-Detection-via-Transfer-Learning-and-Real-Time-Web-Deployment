from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD


class EyeStateDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            raise ValueError(f"Empty split file: {csv_path}")

        required_columns = {"filepath", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        unknown_labels = set(self.df["label"].unique()) - set(CLASS_TO_INDEX.keys())
        if unknown_labels:
            raise ValueError(f"Unknown labels found in {csv_path}: {unknown_labels}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row["filepath"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = CLASS_TO_INDEX[row["label"]]
        return image, label


def build_transforms():
    transform_steps = [transforms.Resize((224, 224))]
    return transforms.Compose(
        transform_steps
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transforms(augment: bool = False):
    transform_steps = [transforms.Resize((224, 224))]
    if augment:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_steps)


def create_dataloader(
    csv_path: str,
    batch_size: int,
    num_workers: int,
    train: bool = False,
    augment: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    transform = build_train_transforms(augment=augment) if train else build_transforms()
    dataset = EyeStateDataset(csv_path=csv_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
