"""Gera os notebooks (nbformat). O texto interpretativo vem de scripts/notebook_text.py, escrito DEPOIS de ver as saídas executadas."""
import sys
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
try:
    from notebook_text import TEXT
except ImportError:
    TEXT = {}

SETUP = '''import sys, json, glob, pickle
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib
matplotlib.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt
'''

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)
def note(key): return [md(TEXT[key])] if key in TEXT else [md(f"> _(interpretação `{key}` pendente)_")]

nb1 = [md("# 01 · Dados, sessões e vazamento\nDados reais (Sudars, milho Nigéria, DeepWeeds). Tudo aqui é recalculado a partir dos arquivos em `data/`."), code(SETUP),
 md("## 1. Sudars: tamanho dos objetos e classes"),
 code('''man = json.load(open(ROOT/"data/processed/sudars_yolo/manifest.json"))
items, info = man["items"], man["info"]
print(json.dumps(info["stats"], indent=1, ensure_ascii=False)); print("descartes:", info["descartes"])
areas = []
for f in glob.glob(str(ROOT/"data/processed/sudars_weed/labels/train/*.txt")):
    for l in open(f): areas.append(float(l.split()[3])*float(l.split()[4]))
areas = np.array(areas)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].hist(areas*100, bins=np.logspace(-3, 0.5, 40)); ax[0].set_xscale("log"); ax[0].set_xlabel("área da caixa (% da imagem)"); ax[0].set_ylabel("caixas (treino)")
ax[1].bar(["train","val","test","test_origin"], [info["stats"][k]["weed"] for k in ("train","val","test","test_origin")], label="weed")
ax[1].bar(["train","val","test","test_origin"], [info["stats"][k]["crop"] for k in ("train","val","test","test_origin")], label="crop", color="C3"); ax[1].legend(); ax[1].set_ylabel("caixas")
plt.tight_layout(); plt.show()
print("mediana %.3f%% | p90 %.3f%% | <0,5%%: %.0f%%" % (np.median(areas)*100, np.percentile(areas, 90)*100, 100*(areas<.005).mean()))'''),
 *note("n1_sudars"),
 md("## 2. A suposição de sessão: vizinhos de ID são mais parecidos?\nCorrelação de miniaturas (32×18, cinza) entre imagens com IDs consecutivos vs. pares aleatórios."),
 code('''import re, cv2
base = ROOT/"data/raw/sudars/ex/Ronin_OPEN_DB/raw images"
num = sorted((int(p.stem), p) for p in base.glob("*.jpg") if re.fullmatch(r"\\d+", p.stem))
thumb = lambda p: cv2.resize(cv2.imread(str(p), 0), (32, 18), interpolation=cv2.INTER_AREA).astype(float).ravel()
T = np.array([thumb(p) for _, p in num]); T = (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
ids = np.array([i for i, _ in num])
nb = np.array([(T[k]*T[k+1]).mean() for k in range(len(T)-1) if ids[k+1]-ids[k] <= 5])
rng = np.random.default_rng(0); a, b = rng.integers(0, len(T), 3000), rng.integers(0, len(T), 3000)
rd = np.array([(T[i]*T[j]).mean() for i, j in zip(a, b) if i != j])
plt.figure(figsize=(6, 3)); plt.hist(rd, 40, alpha=.6, label="pares aleatórios", density=True); plt.hist(nb, 40, alpha=.6, label="IDs vizinhos (Δ≤5)", density=True); plt.legend(); plt.xlabel("correlação"); plt.show()
print("mediana vizinhos %.2f vs aleatórios %.2f (n=%d vs %d)" % (np.median(nb), np.median(rd), len(nb), len(rd)))'''),
 *note("n1_sessoes"),
 md("## 3. DeepWeeds: por que split por data, e por que não dá para ter validação"),
 code('''from weedvision import deepweeds as D
from collections import defaultdict
rows = D.read_labels(ROOT/"data/raw/deepweeds/labels.csv")
days = defaultdict(set)
for r in rows: days[D.CLASSES[r["label"]]].add(r["date"])
print("dias distintos por classe:", {k: len(v) for k, v in days.items()}, "| negativos: %.0f%%" % (100*sum(r["label"]==8 for r in rows)/len(rows)))
for kind, fn in (("aleatório", D.split_random), ("por data", D.split_date)):
    sp = fn(rows); print(kind, D.leakage_report(rows, sp))'''),
 *note("n1_deepweeds")]

nb2 = [md("# 02 · Métricas de detecção e resultados"), code(SETUP),
 md("## 1. As métricas num exemplo que dá para conferir à mão\nDuas verdades, três predições (0,9 acerta · 0,8 erra · 0,7 acerta)."),
 code('''from weedvision import metrics as M
g = [np.array([[0,0,0,10,10],[0,50,50,60,60]])]
p = [np.array([[0,.9,0,0,10,10],[0,.8,100,100,110,110],[0,.7,50,50,60,60]])]
rec, prec, _ = M.pr_curve(g, p, 0); print("revocação", rec, "precisão", prec.round(3))
print("AP50 =", round(M.average_precision(g, p, 0), 4), "(esperado (51·1 + 50·2/3)/101 = %.4f)" % ((51+50*2/3)/101))'''),
 md("## 2. Conferência com pycocotools (implementação independente)"),
 code('''import importlib.util, sys
sys.path.insert(0, str(ROOT/"tests"))
from test_metrics import _random_dataset, _coco_ap
for seed in range(3):
    gts, preds, shapes = _random_dataset(seed)
    ref = _coco_ap(gts, preds, 2, shapes)
    print(seed, [(round(M.average_precision(gts, preds, c, .5), 4), round(ref[c][0], 4)) for c in range(2)], "(próprio, pycocotools)")'''),
 *note("n2_conferencia"),
 md("## 3. Resultados dos modelos (lidos de `runs/eval`)"),
 code('''import pandas as pd
rows = []
for f in sorted((ROOT/"runs/eval").glob("*.json")):
    r = json.load(open(f)); w = r["prf"]["weed"]
    rows.append({"modelo": r["model"], "conjunto": r["split"]+" ("+r["dataset"]+")", "imgs": r["n_images"], "mAP50": r["map50"], "mAP50-95": r["map50_95"], "P@.25": w["precision"], "R@.25": w["recall"], "F1@.25": w["f1"]})
df = pd.DataFrame(rows).round(3); pd.set_option("display.width", 200); print(df.to_string(index=False))'''),
 *note("n2_tabela"),
 md("## 4. Curvas PR e matriz de confusão (YOLO11n)"),
 code('''fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for k, s in enumerate(("val", "test", "test_origin")):
    for m in ("yolo11n", "yolo11s", "frcnn_mnv3", "frcnn_mnv3_v2"):
        f = ROOT/f"runs/eval/{m}__{s}.json"
        if f.exists():
            c = json.load(open(f))["pr_curves"]["weed"]; ax[k].plot(c["recall"], c["precision"], label=m)
    ax[k].set_title(s); ax[k].set_xlabel("revocação"); ax[k].set_ylabel("precisão"); ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1.02); ax[k].legend(fontsize=8)
plt.tight_layout(); plt.show()
cm = np.array(json.load(open(ROOT/"runs/eval/yolo11n__test.json"))["confusion_matrix"])
print("confusão YOLO11n/test (linhas=predito, colunas=real; último índice = fundo):\\n", cm)'''),
 *note("n2_curvas"),
 md("## 5. Falsos negativos: por tamanho e por motivo"),
 code('''for m in ("yolo11n", "frcnn_mnv3_v2"):
    for s in ("test", "test_origin"):
        f = ROOT/f"runs/eval/{m}__{s}.json"
        if not f.exists(): continue
        fn = json.load(open(f))["false_negatives"]; print(f"\\n{m} / {s}: {fn['fn_total']}/{fn['n_gt']} FN | motivos {fn['motivos']}")
        for k, v in fn["por_tamanho"].items(): print(f"   {k:34s} n={v['n']:4d} FN={v['fn']:4d} taxa={v['taxa_fn']:.2f}" if v["taxa_fn"] is not None else f"   {k} n=0")'''),
 *note("n2_fn"),
 md("## 6. Incerteza: bootstrap por imagem (IC 95%) e diferença pareada"),
 code('''b = json.load(open(ROOT/"runs/bootstrap.json"))
for s, d in b.items():
    print(f"\\n{s} (n={d['n_images']} imagens)")
    for m, r in d.items():
        if isinstance(r, dict) and "map50" in r:
            line = f"  {m:14s} mAP50 {r['map50']['media']:.3f} IC95 [{r['map50']['ic95'][0]:.3f}, {r['map50']['ic95'][1]:.3f}]"
            for k, v in r.items():
                if k.startswith("diferenca"): line += f" | Δ vs base {v['map50']['media']:+.3f} [{v['map50']['ic95'][0]:+.3f}, {v['map50']['ic95'][1]:+.3f}]"
            print(line)'''),
 *note("n2_bootstrap")]

nb3 = [md("# 03 · Domínio, drift, borda e georreferenciamento"), code(SETUP),
 md("## 1. Generalização zero-shot para o milho (Nigéria)"),
 code('''for m in ("yolo11n", "yolo11s", "frcnn_mnv3", "frcnn_mnv3_v2"):
    f = ROOT/f"runs/eval/{m}_on_maize__test.json"
    if f.exists():
        r = json.load(open(f)); print(m, "mAP50=%.3f mAP50-95=%.3f recall@.25=%.3f" % (r["map50"], r["map50_95"], r["prf"]["weed"]["recall"]), "| FP:", r["prf"]["weed"]["fp"])
import glob
from PIL import Image
from ultralytics import YOLO
mod = YOLO(str(ROOT/"runs/yolo11n_weed/weights/best.pt"))
fs = sorted(glob.glob(str(ROOT/"data/processed/maize_test/images/test/*.jpg")))[::50][:10]
for conf in (0.001, 0.05, 0.25):
    rs = mod.predict(fs, conf=conf, imgsz=640, device="cpu", verbose=False)
    print("conf>=%.3f: %.1f caixas/imagem" % (conf, np.mean([len(r.boxes) for r in rs])), "| conf máx: %s" % (max([float(r.boxes.conf.max()) for r in rs if len(r.boxes)], default=None)))'''),
 *note("n3_maize"),
 md("## 2. Data drift de aparência (PSI e KS), com piso de ruído"),
 code('''d = json.load(open(ROOT/"runs/drift.json"))
import pandas as pd
tab = {"piso (metades do treino)": {f: v["psi"] for f, v in d["piso_de_ruido_metades_aleatorias_do_treino"].items()}}
for k, c in d["comparacoes"].items(): tab[k] = {f: v["psi"] for f, v in c.items()}
print(pd.DataFrame(tab).T.round(2).to_string()); print("n por conjunto:", d["n_por_conjunto"])
print("\\nmédia de brilho:", {k: round(c["brilho"]["media_cur"], 1) for k, c in d["comparacoes"].items()}, "| treino:", round(next(iter(d["comparacoes"].values()))["brilho"]["media_ref"], 1))'''),
 *note("n3_drift"),
 md("## 3. Paridade PyTorch × ONNX (mesmo módulo de métricas)"),
 code('''for m, o in (("yolo11n", "yolo11n_onnx"), ("yolo11s", "yolo11s_onnx")):
    for s in ("val", "test"):
        a, b = json.load(open(ROOT/f"runs/eval/{m}__{s}.json")), json.load(open(ROOT/f"runs/eval/{o}__{s}.json"))
        print(f"{m:8s} {s:5s} mAP50 PT {a['map50']:.3f} vs ONNX {b['map50']:.3f} | mAP50-95 PT {a['map50_95']:.3f} vs ONNX {b['map50_95']:.3f}")'''),
 *note("n3_onnx"),
 md("## 4. Benchmark de inferência"),
 code('''bm = json.load(open(ROOT/"runs/benchmark.json")); print(bm["hardware"]); print(bm["nota"])
import pandas as pd
print(pd.DataFrame(bm["resultados"])[["nome", "mean_ms", "p50_ms", "p95_ms", "fps", "rss_mb", "n"]].round(1).to_string(index=False)); print("falhas:", bm["falhas"])'''),
 *note("n3_bench"),
 md("## 5. Georreferenciamento: validação **sintética** e sensibilidade a erro de pose"),
 code('''from weedvision import georef as G
K = G.Intrinsics(1000, 1000, 640, 360, 1280, 720); pose = G.Pose(-23.0, -46.0, 1.5, yaw_deg=40, pitch_deg=50)
en = np.array([[0.5, 2.0], [-1.0, 3.5]]); print("ida e volta, erro máx (m):", np.abs(G.pixels_to_enu(G.enu_to_pixels(en, K, pose), K, pose) - en).max())
for pitch in (80, 60, 40, 25, 15):
    s = G.ground_error_sensitivity([[640, 600]], K, G.Pose(-23, -46, 1.5, 0, pitch))
    print(f"pitch {pitch:2d}° → ponto a {np.hypot(*G.pixels_to_enu([[640,600]], K, G.Pose(-23,-46,1.5,0,pitch))[0]):5.1f} m | erro por 1° de pitch: {s['pitch_erro_m']:.2f} m | por 10 cm de altura: {s['altura_erro_m']:.2f} m")
try: G.pixels_to_enu([[640, 100]], K, G.Pose(-23, -46, 1.5, 0, 5))
except G.GeorefError as e: print("recusado:", e)'''),
 *note("n3_georef"),
 md("## 6. Augmentation realista"),
 code('''from weedvision import augment as AU
import cv2, random
random.seed(1); np.random.seed(1)
f = sorted((ROOT/"sample_data").glob("sudars_test_1*.jpg"))[0]; img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
lab = np.loadtxt(ROOT/"data/processed/sudars_weed/labels/test"/(f.stem.replace("sudars_test_","")+".txt"), ndmin=2); h, w = img.shape[:2]
boxes = [((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h) for _, x, y, bw, bh in lab]
aug = AU.build(); fig, ax = plt.subplots(2, 3, figsize=(13, 5))
for a_, k in zip(ax.ravel(), range(6)):
    im, bb, _ = (img, boxes, None) if k == 0 else AU.apply(aug, img, boxes, [0]*len(boxes))
    im = im.copy()
    for x1, y1, x2, y2 in bb: cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
    a_.imshow(im); a_.axis("off"); a_.set_title("original" if k == 0 else f"augment {k}")
plt.tight_layout(); plt.show()'''),
 *note("n3_aug")]

for name, cells in (("01_dados_sessoes_vazamento", nb1), ("02_metricas_resultados", nb2), ("03_dominio_drift_borda_georef", nb3)):
    nb = nbf.v4.new_notebook(); nb.cells = cells; nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (root / "notebooks").mkdir(exist_ok=True); nbf.write(nb, root / f"notebooks/{name}.ipynb"); print("ok", name)
