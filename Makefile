PY=.venv/bin/python
.PHONY: setup data train eval bench registry api web test e2e notebooks
setup:        ## venv + dependências de treino
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements-train.txt && .venv/bin/pip install -e .
data:         ## Sudars VOC→YOLO/COCO (+ derivados) — exige os dados brutos em data/raw (ver docs/pt-BR/02_dados.md)
	$(PY) -m weedvision.data_prep
test:
	$(PY) -m pytest -q
train:
	$(PY) scripts/train_yolo.py n 40 && $(PY) scripts/train_yolo.py s 40
eval:
	$(PY) scripts/eval_yolo.py runs/yolo11n_weed/weights/best.pt yolo11n test
bench:
	$(PY) scripts/benchmark.py
registry:
	$(PY) scripts/build_registry.py
api:
	$(PY) -m uvicorn backend.app.main:app --port 8000
web:
	cd frontend && npm install && npm run dev
