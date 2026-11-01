"""Avalia um modelo ONNX com o runtime próprio (sem ultralytics) e o módulo de métricas próprio. Uso: eval_onnx.py <onnx> <nome> <split> [dataset_dir]."""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from weedvision import metrics as M
from weedvision.onnx_infer import OnnxDetector

path, name, split = sys.argv[1:4]
ds = Path(sys.argv[4]) if len(sys.argv) > 4 else root / "data/processed/sudars_weed"
import os
prov = os.environ.get("ORT_PROVIDERS")
det = OnnxDetector(path, providers=prov.split(",") if prov else None)
gts, preds, shapes = [], [], []
for p in sorted((ds / "images" / split).iterdir()):
    img = np.asarray(Image.open(p).convert("RGB")); h, w = img.shape[:2]
    d = det(img, conf=0.001, iou=0.6)
    preds.append(np.array([[x.cls, x.score, *x.xyxy] for x in d]).reshape(-1, 6)); shapes.append((h, w))
    lab = ds / "labels" / split / f"{p.stem}.txt"
    g = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
    gts.append(np.c_[g[:, 0], (g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h, (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h] if len(g) else g)
res = M.evaluate(gts, preds, ["weed"], conf=0.25); res["false_negatives"] = M.false_negative_analysis(gts, preds, shapes, conf=0.25)
res.update({"model": name, "weights": path, "split": split, "dataset": ds.name, "runtime": f"onnxruntime {prov or 'CPUExecutionProvider'}, letterbox quadrado 640"})
(root / "runs/eval").mkdir(exist_ok=True); (root / f"runs/eval/{name}__{split}.json").write_text(json.dumps(res, indent=1))
print(name, split, "mAP50=%.3f mAP50-95=%.3f" % (res["map50"], res["map50_95"]))
