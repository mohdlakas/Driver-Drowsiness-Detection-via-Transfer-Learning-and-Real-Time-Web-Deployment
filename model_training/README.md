# Model Training (Eye-Closure Proxy for Drowsiness)

This project trains a simple transfer-learning classifier on the **MRL Eye Dataset** from Kaggle.

Task:
- `closed` eyes -> proxy for **drowsy**
- `open` eyes -> proxy for **alert**

Important scope note:
- This is a **simplified proxy task** based on eye closure only.
- It is **not** a full real-world drowsiness understanding system (it does not model yawning, head pose, context, or temporal behavior).

## Project Structure

```
model_training/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/         # Kaggle download/unzip target
│   └── processed/   # metadata + train/val/test split CSVs
├── outputs/
│   ├── models/      # best model checkpoint
│   ├── metrics/     # history, metrics, classification report
│   └── plots/       # loss/accuracy curves, confusion matrix
└── src/
    ├── download_data.py
    ├── prepare_data.py
    ├── train.py
    ├── evaluate.py
    ├── data_utils.py
    ├── modeling.py
    ├── plotting.py
    ├── config.py
    └── utils.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies (pinned in `requirements.txt`):

```bash
cd "/mnt/d/Desktop/model_training"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The requirements are pinned for CUDA `cu126` wheels.
- On a CUDA 12.7 machine, `cu126` is usually the closest compatible PyTorch wheel target.

## Kaggle API Setup

Use one of these authentication options:

1. Environment variables:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

2. Or `~/.kaggle/kaggle.json`:
- Download your API token from Kaggle account settings.
- Place it at `~/.kaggle/kaggle.json`.

Example command:

```bash
mkdir -p ~/.kaggle
cp "/mnt/d/Desktop/Bashar/driver_drowsiness/kaggle.json" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Run Pipeline

From the `model_training/` directory:

1. Download MRL Eye dataset from Kaggle:
```bash
python -m src.download_data
```

2. Prepare metadata and 70/15/15 splits:
```bash
python -m src.prepare_data
```

3. Train transfer-learning model (`resnet18`):
```bash
# Baseline (no augmentation)
python -m src.train --model resnet18
python -m src.train --model mobilenet_v3_small
python -m src.train --model efficientnet_b0

# Augmented training
python -m src.train --model resnet18 --augment-data
python -m src.train --model mobilenet_v3_small --augment-data
python -m src.train --model efficientnet_b0 --augment-data
```

4. Evaluate on test set:
```bash
# Baseline checkpoints
python -m src.evaluate --model resnet18
python -m src.evaluate --model mobilenet_v3_small
python -m src.evaluate --model efficientnet_b0

# Augmented checkpoints
python -m src.evaluate --model resnet18 --augment-data
python -m src.evaluate --model mobilenet_v3_small --augment-data
python -m src.evaluate --model efficientnet_b0 --augment-data
```

5. Compare evaluated models (automatically includes base vs augmented if both exist):
```bash
python -m src.compare
```

## Dataset and Splitting Logic

- Expected labels are mapped to two classes:
  - `open`
  - `closed`
- Label inference priority:
  1. MRL filename convention (`0`/`1` eye-state token)
  2. Folder/filename keywords (`open/awake` vs `closed/sleepy`)
- Subject split:
  - If subject IDs are detected (e.g., `s0001`) for all rows, splitting uses grouped subject split to reduce leakage.
  - Otherwise falls back to a standard random split.
- Split ratios:
  - train `70%`
  - val `15%`
  - test `15%`

## Model

- Backbones (transfer learning):
  - `resnet18`
  - `mobilenet_v3_small`
  - `efficientnet_b0`
- Final layer replaced with 2-class head
- Input preprocessing:
  - resize to `224x224`
  - convert to RGB (3 channels)
  - normalize with ImageNet mean/std
- Optional training augmentation (`--augment-data`):
  - random horizontal flip
  - random rotation
  - color jitter

## Saved Artifacts

- Best checkpoint:
  - `outputs/models/<model>_best.pt`
  - `outputs/models/<model>_aug_best.pt` (when `--augment-data` is used)
- Training history:
  - `outputs/metrics/<model>_history.csv`
  - `outputs/metrics/<model>_aug_history.csv`
- Curves:
  - `outputs/plots/<model>_training_curves.png`
  - `outputs/plots/<model>_aug_training_curves.png`
- Test metrics:
  - `outputs/metrics/<model>_test_metrics.json`
  - `outputs/metrics/<model>_aug_test_metrics.json`
  - `outputs/metrics/<model>_classification_report.txt`
  - `outputs/metrics/<model>_aug_classification_report.txt`
- Confusion matrix:
  - `outputs/plots/<model>_confusion_matrix.png`
  - `outputs/plots/<model>_aug_confusion_matrix.png`
- Comparison outputs:
  - `outputs/metrics/compare_summary.csv`
  - `outputs/metrics/augmentation_pairwise_report.csv`
  - `outputs/plots/model_comparison.png`
  - `outputs/plots/augmentation_delta.png`

## Optional Useful Flags

```bash
python -m src.download_data --dataset imadeddinedjerarda/mrl-eye-dataset
python -m src.train --model resnet18 --epochs 12 --batch-size 32 --lr 1e-4
python -m src.train --model resnet18 --augment-data --epochs 12 --batch-size 32 --lr 1e-4
python -m src.evaluate --model resnet18 --checkpoint outputs/models/resnet18_best.pt
python -m src.evaluate --model resnet18 --augment-data
python -m src.compare --sort-by accuracy
```
