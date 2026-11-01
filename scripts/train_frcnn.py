"""Faster R-CNN MobileNetV3-Large FPN (torchvision, BSD-3) com pesos COCO_V1, 1 classe + fundo. Avalia com o módulo próprio."""
import json, os, random, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights as W, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from weedvision import metrics as M

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
TAG = sys.argv[3] if len(sys.argv) > 3 else ""          # "" = v1 (lr 0.01, sem warmup); "_v2" = lr menor + warmup
WARMUP = TAG != ""
ds = root / "data/processed/sudars_weed"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def load(split, root_=ds):
    out = []
    for p in sorted((root_ / "images" / split).iterdir()):
        lab = root_ / "labels" / split / f"{p.stem}.txt"
        g = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        im = Image.open(p).convert("RGB"); w, h = im.size
        b = np.c_[(g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h, (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h] if len(g) else np.zeros((0, 4))
        out.append((p, b.astype(np.float32)))
    return out


model = fasterrcnn_mobilenet_v3_large_fpn(weights=W.COCO_V1, min_size=640, max_size=1138, box_detections_per_img=100, box_score_thresh=0.001)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
model.to(dev)
train = [t for t in load("train") if len(t[1])]
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100) if WARMUP else None
t0, log = time.time(), []
for ep in range(EPOCHS):
    model.train(); random.shuffle(train); tot = 0
    for i in range(0, len(train), 4):
        imgs, tg = [], []
        for p, b in train[i:i + 4]:
            im = TF.to_tensor(Image.open(p).convert("RGB")); bb = torch.tensor(b)
            if random.random() < 0.5:
                im = im.flip(-1); w = im.shape[-1]; bb = torch.stack([w - bb[:, 2], bb[:, 1], w - bb[:, 0], bb[:, 3]], 1)
            imgs.append(im.to(dev)); tg.append({"boxes": bb.to(dev), "labels": torch.ones(len(bb), dtype=torch.int64, device=dev)})
        loss = sum(model(imgs, tg).values()); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if warm is not None and ep == 0: warm.step()
    sched.step(); log.append(tot / (len(train) / 4)); print(f"epoch {ep + 1}/{EPOCHS} loss {log[-1]:.4f} t={time.time() - t0:.0f}s", flush=True)
train_s = time.time() - t0
out = root / f"runs/frcnn_mnv3{TAG}_weed"; out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
(out / "train_time.json").write_text(json.dumps({"seconds": train_s, "epochs": EPOCHS, "lr": LR, "warmup": WARMUP, "device": str(dev), "loss": log}))


@torch.no_grad()
def predict(items):
    model.eval(); preds, shapes, gts = [], [], []
    for p, b in items:
        im = TF.to_tensor(Image.open(p).convert("RGB")); r = model([im.to(dev)])[0]
        preds.append(np.c_[np.zeros(len(r["boxes"])), r["scores"].cpu().numpy(), r["boxes"].cpu().numpy()] if len(r["boxes"]) else np.zeros((0, 6)))
        shapes.append(tuple(im.shape[1:])); gts.append(np.c_[np.zeros(len(b)), b] if len(b) else np.zeros((0, 5)))
    return gts, preds, shapes


for name, split, d in [(f"frcnn_mnv3{TAG}", "val", ds), (f"frcnn_mnv3{TAG}", "test", ds), (f"frcnn_mnv3{TAG}", "test_origin", ds), (f"frcnn_mnv3{TAG}_on_maize", "test", root / "data/processed/maize_test")]:
    gts, preds, shapes = predict(load(split, d))
    import pickle; (root / "runs/preds").mkdir(parents=True, exist_ok=True); pickle.dump({"gts": gts, "preds": preds, "shapes": shapes}, open(root / f"runs/preds/{name}__{split}.pkl", "wb"))
    res = M.evaluate(gts, preds, ["weed"], conf=0.25); res["false_negatives"] = M.false_negative_analysis(gts, preds, shapes, conf=0.25)
    res.update({"model": name, "split": split, "dataset": d.name, "weights": str(out / "model.pt")})
    (root / "runs/eval").mkdir(exist_ok=True); (root / f"runs/eval/{name}__{split}.json").write_text(json.dumps(res, indent=1))
    print(name, split, "mAP50=%.3f mAP50-95=%.3f" % (res["map50"], res["map50_95"]), flush=True)
