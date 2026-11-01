"""Drift de aparência: treino Sudars vs val/test/test_origin/maize (amostra determinística de até 150 imagens por conjunto)."""
import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from weedvision import drift as D

sets = {"train": root / "data/processed/sudars_weed/images/train", "val": root / "data/processed/sudars_weed/images/val",
        "test": root / "data/processed/sudars_weed/images/test", "test_origin": root / "data/processed/sudars_weed/images/test_origin",
        "maize_nigeria": root / "data/processed/maize_test/images/test"}
stats = {}
for k, d in sets.items():
    files = sorted(d.iterdir()); files = files[:: max(1, len(files) // 150)][:150]
    stats[k] = D.collect(files); print(k, len(files), flush=True)
import numpy as np
rng = np.random.default_rng(0)
perm = rng.permutation(len(stats["train"]["brilho"])); h = len(perm) // 2
piso = D.compare({f: v[perm[:h]] for f, v in stats["train"].items()}, {f: v[perm[h:]] for f, v in stats["train"].items()})
print("piso de ruído (metades aleatórias do treino):", {f: round(v["psi"], 2) for f, v in piso.items()})
out = {"piso_de_ruido_metades_aleatorias_do_treino": piso, "referencia": "train", "n_por_conjunto": {k: len(v["brilho"]) for k, v in stats.items()},
       "comparacoes": {k: D.compare(stats["train"], v) for k, v in stats.items() if k != "train"}}
(root / "runs").mkdir(exist_ok=True); (root / "runs/drift.json").write_text(json.dumps(out, indent=1))
for k, c in out["comparacoes"].items():
    print(k, {f: round(v["psi"], 2) for f, v in c.items()})
