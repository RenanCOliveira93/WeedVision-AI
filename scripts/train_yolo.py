"""Fine-tuning YOLO11 (Ultralytics, AGPL-3.0) a partir de pesos COCO. Uso: train_yolo.py <n|s> [epochs]."""
import json, sys, time
from pathlib import Path

from ultralytics import YOLO, settings

settings.update({"sync": False})
size = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
root = Path(__file__).resolve().parents[1]
t0 = time.time()
m = YOLO(str(root / f"models/pretrained/yolo11{size}.pt"))
m.train(data=str(root / "data/processed/sudars_weed/data.yaml"), epochs=epochs, imgsz=640, batch=16, device="mps",
        seed=0, deterministic=False, workers=4, patience=15, project=str(root / "runs"), name=f"yolo11{size}_weed",
        exist_ok=True, plots=True, verbose=False)
(root / f"runs/yolo11{size}_weed/train_time.json").write_text(json.dumps({"seconds": time.time() - t0, "epochs_requested": epochs, "device": "mps"}))
