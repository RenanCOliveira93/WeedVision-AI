# 11 · Reprodução

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt && pip install -e .
```

## Dados (internet; não versionados em `data/`)
1. **Sudars:** baixar `Ronin_OPEN_DB.zip` (Mendeley Data, DOI 10.17632/nj4vtk4tt6) para `data/raw/sudars/` e extrair em `data/raw/sudars/ex/` (esperado: `Ronin_OPEN_DB/raw images` e `.../annotations`).
2. **Milho:** os dois Parquet do espelho HF em `data/raw/maize/` (licença conferida na fonte Mendeley).
3. **DeepWeeds:** `images.zip` (Zenodo 7939060; MD5 `b7b30f96d466fba86016aa5a26606e0f`) extraído em `data/raw/deepweeds/ex/` e `labels.csv` do repositório GitHub em `data/raw/deepweeds/`.
4. Pesos iniciais: `models/pretrained/yolo11n.pt`, `yolo11s.pt` (Ultralytics) e pesos COCO do torchvision (baixados automaticamente).

## Pipeline
```bash
python -m weedvision.data_prep                     # Sudars → YOLO/COCO + manifesto (data/processed/sudars_yolo)
python - <<'PY'                                    # derivados de 1 classe e milho
from pathlib import Path
from weedvision.data_prep import weed_only, build_maize
weed_only(Path("data/processed/sudars_yolo"), Path("data/processed/sudars_weed"))
build_maize(Path("data/raw/maize"), Path("data/processed/maize_test"))
PY
python scripts/train_yolo.py n 40 && python scripts/train_yolo.py s 40
python scripts/train_frcnn.py 12 && python scripts/train_frcnn.py 12 0.005 _v2
python scripts/train_deepweeds.py 6 random date date:1 date:2 random:1
# exportar ONNX (ver scripts) e avaliar
for m in n s; do for s in val test test_origin; do python scripts/eval_yolo.py runs/yolo11${m}_weed/weights/best.pt yolo11$m $s; done; done
python scripts/run_bootstrap.py yolo11n yolo11s frcnn_mnv3_v2
python scripts/run_drift.py
python scripts/benchmark.py        # sem outros processos pesados em paralelo!
python scripts/build_registry.py && python scripts/fill_docs.py
```
Exportação ONNX: `YOLO(best.pt).export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)` → copiar para `models/onnx/yolo11{n,s}_weed.onnx`.

## App
```bash
python -m uvicorn backend.app.main:app --port 8000      # http://127.0.0.1:8000/docs  (OpenAPI)
cd frontend && npm install && npm run dev                # http://localhost:5173
```
## Testes
```bash
pytest -q                          # unitários + API (sem navegador)
cd frontend && npm test && npx tsc --noEmit
cd frontend && npm run build && cd .. && pytest tests/e2e -m e2e   # Playwright + webcam falsa
```
## Notebooks
`python scripts/build_notebooks.py && python scripts/run_notebooks.py` (executa em ordem; usa `runs/` e `data/` reais).

> **Determinismo:** as avaliações reproduzem os mesmos números ao reexecutar; o **treino** (MPS, `deterministic=False`) **não** é bit a bit reprodutível. Docker: arquivos em `infra/`, **não executados**.
