# Driver Drowsiness Web App

FastAPI backend + browser frontend for realtime driver drowsiness inference.

## Requirements

- Python 3.10+
- `pip`
- Trained model file (default):
  - `model_training/outputs/models/resnet18_aug_best.pt`

The app searches for the model in:

1. `model_training/outputs/models/resnet18_aug_best.pt` (default target)
2. `web-app/drowsiness_model.pth`
3. `model_training/drowsiness_model.pth`

Optional override (any custom checkpoint path):

```bash
export DROWSINESS_MODEL_PATH="/full/path/to/model.pt"
```

## Run (Linux/macOS/WSL)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000` (web app)
- `http://127.0.0.1:8000/api/health` (backend/model status)

## Run (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
