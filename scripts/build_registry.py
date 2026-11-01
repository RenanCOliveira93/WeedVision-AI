"""Gera models/registry.json a partir dos artefatos reais (pesos, ONNX, runs/eval). Nenhuma métrica é digitada."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ev = {p.stem: json.loads(p.read_text()) for p in (root / "runs/eval").glob("*.json")}
NOTE = "Treinado em Sudars 2020 (plantas individuais; vasos em estufa e campo aberto); não validado para café brasileiro; falhou zero-shot no milho (Nigéria)."
spec = [
    ("yolo11n", "YOLO11n (Ultralytics)", "pytorch+onnx", "runs/yolo11n_weed/weights/best.pt", "models/onnx/yolo11n_weed.onnx", "AGPL-3.0 (pesos e código Ultralytics)", "yolo11n"),
    ("yolo11s", "YOLO11s (Ultralytics)", "pytorch+onnx", "runs/yolo11s_weed/weights/best.pt", "models/onnx/yolo11s_weed.onnx", "AGPL-3.0 (pesos e código Ultralytics)", "yolo11s"),
    ("frcnn_mnv3", "Faster R-CNN MobileNetV3 v1 (lr 0,01, sem warmup)", "pytorch", "runs/frcnn_mnv3_weed/model.pt", None, "BSD-3-Clause (torchvision); pesos COCO iniciais", "frcnn_mnv3"),
    ("frcnn_mnv3_v2", "Faster R-CNN MobileNetV3 v2 (lr 0,005, warmup)", "pytorch", "runs/frcnn_mnv3_v2_weed/model.pt", None, "BSD-3-Clause (torchvision); pesos COCO iniciais", "frcnn_mnv3_v2"),
]
reg = []
for mid, name, fmt, pt, onnx, lic, evk in spec:
    if not (root / pt).exists():
        continue
    m = {"id": mid, "name": name, "task": "detecção (caixas)", "format": fmt, "weights": pt, "onnx": onnx if onnx and (root / onnx).exists() else None,
         "classes": ["weed"], "license": lic, "trained_on": "sudars_weed/train (CC BY 4.0)", "domain_note": NOTE,
         "metrics": {k.split("__")[1]: {"map50": v["map50"], "map50_95": v["map50_95"], "precision": v["prf"]["weed"]["precision"], "recall": v["prf"]["weed"]["recall"]}
                     for k, v in ev.items() if k.startswith(evk + "__")}}
    reg.append(m)
(root / "models/registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
print([m["id"] for m in reg])
