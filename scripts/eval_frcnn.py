"""Reavalia um Faster R-CNN salvo (sem retreinar). Uso: eval_frcnn.py <tag: ''|_v2>"""
import json, os, pickle, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from weedvision import metrics as M
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
dev = torch.device("mps")
m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, min_size=640, max_size=1138, box_detections_per_img=100, box_score_thresh=0.001)
m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
m.load_state_dict(torch.load(root / f"runs/frcnn_mnv3{TAG}_weed/model.pt", map_location="cpu")); m.eval().to(dev)
for name, split, d in [(f"frcnn_mnv3{TAG}", s, root / "data/processed/sudars_weed") for s in ("val", "test", "test_origin")] + [(f"frcnn_mnv3{TAG}_on_maize", "test", root / "data/processed/maize_test")]:
    gts, preds, shapes = [], [], []
    for p in sorted((d / "images" / split).iterdir()):
        im = TF.to_tensor(Image.open(p).convert("RGB")); h, w = im.shape[1:]
        with torch.no_grad(): r = m([im.to(dev)])[0]
        preds.append(np.c_[np.zeros(len(r["boxes"])), r["scores"].cpu().numpy(), r["boxes"].cpu().numpy()] if len(r["boxes"]) else np.zeros((0, 6))); shapes.append((h, w))
        lab = d / "labels" / split / f"{p.stem}.txt"; g = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        gts.append(np.c_[g[:, 0], (g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h, (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h] if len(g) else g)
    pickle.dump({"gts": gts, "preds": preds, "shapes": shapes}, open(root / f"runs/preds/{name}__{split}.pkl", "wb"))
    res = M.evaluate(gts, preds, ["weed"], conf=0.25); res["false_negatives"] = M.false_negative_analysis(gts, preds, shapes, conf=0.25)
    res.update({"model": name, "split": split, "dataset": d.name, "weights": f"runs/frcnn_mnv3{TAG}_weed/model.pt"})
    (root / f"runs/eval/{name}__{split}.json").write_text(json.dumps(res, indent=1)); print(name, split, "mAP50=%.3f" % res["map50"], flush=True)
