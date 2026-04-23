from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CLASS_NAMES = ["Drowsy", "Alert"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = BASE_DIR.parent / "model_training" / "outputs" / "models" / "resnet18_aug_best.pt"
ENV_MODEL_PATH = os.getenv("DROWSINESS_MODEL_PATH")
MODEL_CANDIDATES = [Path(ENV_MODEL_PATH)] if ENV_MODEL_PATH else []
MODEL_CANDIDATES.extend(
    [
        DEFAULT_MODEL_PATH,
        BASE_DIR / "drowsiness_model.pth",
        BASE_DIR.parent / "model_training" / "drowsiness_model.pth",
    ]
)
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def resolve_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(f"- {p}" for p in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model file not found. Searched:\n{searched}")


def build_model(model_path: Path) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model


MODEL: nn.Module | None = None
MODEL_PATH: Path | None = None
MODEL_ERROR: str | None = None
try:
    MODEL_PATH = resolve_model_path()
    MODEL = build_model(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    MODEL_ERROR = str(exc)

app = FastAPI(title="Driver Drowsiness API", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    if MODEL is None:
        return {
            "status": "error",
            "device": str(DEVICE),
            "model_path": str(MODEL_PATH) if MODEL_PATH else "",
            "message": MODEL_ERROR or "Model failed to load.",
        }
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "message": "Model loaded",
    }


@app.post("/api/predict")
async def predict(frame: UploadFile = File(...)) -> dict[str, float | int | str]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR or "Model is unavailable.")

    try:
        payload = await frame.read()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload.") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    started = perf_counter()
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    latency_ms = (perf_counter() - started) * 1000.0

    pred_idx = int(max(range(len(probs)), key=probs.__getitem__))
    return {
        "prediction_index": pred_idx,
        "prediction_label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "drowsy_prob": float(probs[0]),
        "non_drowsy_prob": float(probs[1]),
        "latency_ms": float(latency_ms),
        "device": str(DEVICE),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
