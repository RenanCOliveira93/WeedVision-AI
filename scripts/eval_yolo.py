"""Avalia pesos YOLO num split com o módulo de métricas próprio. Uso: eval_yolo.py <pesos.pt> <nome> <split> [dataset_dir]."""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from weedvision import metrics as M

weights, name, split = sys.argv[1:4]
ds = Path(sys.argv[4]) if len(sys.argv) > 4 else root / "data/processed/sudars_weed"
imgs = sorted((ds / "images" / split).iterdir())
model = YOLO(weights)
gts, preds, shapes = [], [], []
for i in range(0, len(imgs), 16):
    for r in model.predict([str(p) for p in imgs[i:i + 16]], conf=0.001, iou=0.6, max_det=100, imgsz=640, device="mps", verbose=False):
        h, w = r.orig_shape
        b = r.boxes
        preds.append(np.c_[b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy()] if len(b) else np.zeros((0, 6)))
        shapes.append((h, w))
for p, (h, w) in zip(imgs, shapes):
    lab = ds / "labels" / split / f"{p.stem}.txt"
    g = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
    gts.append(np.c_[g[:, 0], (g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h, (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h] if len(g) else g)
res = M.evaluate(gts, preds, ["weed"], conf=0.25)
res["false_negatives"] = M.false_negative_analysis(gts, preds, shapes, conf=0.25)
crop = json.loads((ds / "crop_boxes.json").read_text()).get(split, {}) if (ds / "crop_boxes.json").exists() else {}
if crop:  # FP de weed cujo centro cai dentro de uma caixa `crop` anotada
    fp = on_crop = 0
    for p, g, pr in zip(imgs, gts, preds):
        pr = pr[pr[:, 1] >= 0.25]
        ious = M.iou_matrix(pr[:, 2:6], g[:, 1:5]) if len(g) else np.zeros((len(pr), 0))
        for k, row in enumerate(pr):
            if ious.shape[1] and ious[k].max() >= 0.5:
                continue
            fp += 1
            cx, cy = (row[2] + row[4]) / 2, (row[3] + row[5]) / 2
            on_crop += any(a <= cx <= c and b <= cy <= d for a, b, c, d in crop.get(p.name, []))
    res["fp_on_crop"] = {"fp_total": fp, "fp_com_centro_em_crop": on_crop}
import pickle; (root / "runs/preds").mkdir(parents=True, exist_ok=True); pickle.dump({"gts": gts, "preds": preds, "shapes": shapes}, open(root / f"runs/preds/{name}__{split}.pkl", "wb"))
res.update({"model": name, "weights": str(weights), "split": split, "dataset": str(ds.name)})
out = root / "runs/eval"; out.mkdir(parents=True, exist_ok=True)
(out / f"{name}__{split}.json").write_text(json.dumps(res, indent=1))
print(name, split, "mAP50=%.3f mAP50-95=%.3f" % (res["map50"], res["map50_95"]), "P/R@0.25=%.3f/%.3f" % (res["prf"]["weed"]["precision"], res["prf"]["weed"]["recall"]), res.get("fp_on_crop", ""))
