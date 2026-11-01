"""IC 95% (bootstrap por imagem) e diferença pareada entre modelos, a partir de runs/preds/*.pkl. Uso: run_bootstrap.py <modelo_base> <outro> [...]."""
import json, pickle, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from weedvision import metrics as M

models = sys.argv[1:]
out = {}
for split in ("val", "test", "test_origin"):
    data = {m: pickle.load(open(root / f"runs/preds/{m}__{split}.pkl", "rb")) for m in models if (root / f"runs/preds/{m}__{split}.pkl").exists()}
    if len(data) < 1: continue
    first = next(iter(data.values()))
    out[split] = {"n_images": len(first["gts"]), **M.bootstrap_map(first["gts"], {m: d["preds"] for m, d in data.items()}, n_boot=200)}
    for m, r in out[split].items():
        if isinstance(r, dict) and "map50" in r: print(split, m, "mAP50 %.3f [%.3f, %.3f]" % (r["map50"]["media"], *r["map50"]["ic95"]), "| mAP50-95 [%.3f, %.3f]" % tuple(r["map50_95"]["ic95"]), *( ["diff50 %.3f [%.3f, %.3f]" % (r[k]["map50"]["media"], *r[k]["map50"]["ic95"]) for k in r if k.startswith("diferenca")] ))
(root / "runs/bootstrap.json").write_text(json.dumps(out, indent=1))
