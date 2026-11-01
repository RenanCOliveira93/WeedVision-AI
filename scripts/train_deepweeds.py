"""Baseline de CLASSIFICAÇÃO DeepWeeds (MobileNetV3-Large, ImageNet). Compara split aleatório vs por data. Uso: train_deepweeds.py [epochs]."""
import json, sys, time, random
from pathlib import Path

import numpy as np, torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights as W, mobilenet_v3_large

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from weedvision import deepweeds as D

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rows = D.read_labels(root / "data/raw/deepweeds/labels.csv")
X = np.stack([np.asarray(Image.open(root / "data/raw/deepweeds/ex" / r["file"]).convert("RGB").resize((224, 224))) for r in rows])
y = np.array([r["label"] for r in rows]); files = [r["file"] for r in rows]
mean = torch.tensor(W.IMAGENET1K_V2.transforms().mean).view(1, 3, 1, 1).to(dev); std = torch.tensor(W.IMAGENET1K_V2.transforms().std).view(1, 3, 1, 1).to(dev)


def batch(idx, aug):
    x = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).float().div(255).to(dev)
    if aug:
        if random.random() < 0.5: x = x.flip(-1)
        if random.random() < 0.5: x = x.flip(-2)
        x = x * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1, device=dev))  # variação de iluminação
    return (x.clamp(0, 1) - mean) / std


def run(kind, seed=0):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    sp = (D.split_random if kind == "random" else D.split_date)(rows, seed=seed)
    idx = {s: np.array([i for i, f in enumerate(files) if sp[f] == s]) for s in ("train", "val", "test")}
    has_val = len(idx["val"]) > 0
    m = mobilenet_v3_large(weights=W.IMAGENET1K_V2); m.classifier[3] = torch.nn.Linear(1280, len(D.CLASSES)); m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    t0 = time.time()

    @torch.no_grad()
    def pred(ix):
        m.eval(); return np.concatenate([m(batch(ix[i:i + 128], False)).argmax(1).cpu().numpy() for i in range(0, len(ix), 128)])

    for ep in range(EPOCHS):
        m.train(); tr = np.random.permutation(idx["train"])
        for i in range(0, len(tr), 64):
            b = tr[i:i + 64]; loss = torch.nn.functional.cross_entropy(m(batch(b, True)), torch.from_numpy(y[b]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        va = float((pred(idx["val"]) == y[idx["val"]]).mean()) if has_val else None
        print(kind, "epoch", ep + 1, "val_acc", va, "t=%.0fs" % (time.time() - t0), flush=True)
    # protocolo igual nos dois splits: modelo da ÚLTIMA época (sem seleção por validação)
    out = {"split": kind, "seed": seed, "leakage": D.leakage_report(rows, sp), "epochs": EPOCHS, "seconds": time.time() - t0, "protocolo": "última época, sem seleção por validação"}
    for s in (("val",) if has_val else ()) + ("test",):
        p, t = pred(idx[s]), y[idx[s]]
        cm = np.zeros((9, 9), int)
        for a, b_ in zip(t, p): cm[a, b_] += 1
        rec = np.diag(cm) / np.maximum(cm.sum(1), 1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
        out[s] = {"n": int(len(t)), "accuracy": float((p == t).mean()), "macro_f1": float(f1.mean()), "recall_por_classe": dict(zip(D.CLASSES, rec.round(4).tolist())),
                  "confusion_matrix(linhas=real,colunas=predito)": cm.tolist()}
    (root / "runs/deepweeds").mkdir(parents=True, exist_ok=True)
    (root / f"runs/deepweeds/cls_{kind}{'' if seed == 0 else f'_s{seed}'}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(kind, seed, "TEST acc %.4f macroF1 %.4f" % (out["test"]["accuracy"], out["test"]["macro_f1"]), flush=True)


for spec in (sys.argv[2:] or ["random", "date"]):
    k, _, sd = spec.partition(":")
    run(k, int(sd or 0))
