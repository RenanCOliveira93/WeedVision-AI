# Project 04 · WeedVision AI

[🇧🇷 Português](README.md) · 🇺🇸 English

A **computer vision and edge AI** platform for **weed detection** in RGB field images: annotation validation, training/fine-tuning, evaluation with our own metrics, ONNX export with an edge runtime, a FastAPI backend and a React dashboard.

> Documentation in `docs/pt-BR/` and the notebooks are in Portuguese; this README summarises the project.

> ### ⚠️ What this project is NOT
> - **Detecting a weed is not a spraying recommendation.** Application requires agronomic criteria and field validation. **No actuator control.**
> - **No coffee data.** No model was validated for Brazilian coffee; testing on another dataset (maize, Nigeria) gave **mAP50 = 0**.
> - **Segmentation was not trained.** Classification (DeepWeeds), detection (Sudars) and segmentation are separate tasks; **no image-level label was turned into a box**.
> - **Georeferencing only with calibration + pose;** validated with synthetic data only.

## Results (measured)
One-class (`weed`) detection trained on Sudars et al. 2020 (CC BY 4.0). mAP from our own module, cross-checked against `pycocotools`; one seed per model.

| Model | Test (unseen sessions) mAP50 / mAP50-95 | Held-out origin `IMG_*` | Maize (Nigeria, zero-shot) |
|---|---|---|---|
| YOLO11n | 0.725 / 0.488 | 0.291 / 0.127 | **0.000** |
| YOLO11s | 0.727 / 0.512 | 0.217 / 0.092 | **0.000** |
| Faster R-CNN MobileNetV3 (v1) | 0.612 / 0.326 | 0.186 / 0.070 | **0.000** |
| Faster R-CNN MobileNetV3 (v2, one retune) | 0.605 / 0.317 | 0.179 / 0.063 | **0.000** |

- YOLO11n and 11s are **statistically indistinguishable** in-domain (paired bootstrap); Faster R-CNN is lower with this recipe (rising training loss **undiagnosed**).
- **Generalisation is the problem:** ≈ 0.7 on unseen sessions, 0.2–0.3 on another origin, 0 on another dataset. In the held-out group, **145 of 534** YOLO11n false positives fall on the **crop**.
- **DeepWeeds (classification, not detection):** random split 91.8–94.3% accuracy vs **46.5–61.2%** with a by-date split (3 draws of days): the random split **leaks** same-day frames. The test sets differ.
- **Edge (Apple M4 Max, not edge hardware):** YOLO11n on ONNX Runtime CPU 13.1 ms (≈ 77 FPS); CoreML 6.6 ms; PyTorch CPU 21.6 ms. ONNX within 0.006 mAP50 of PyTorch.

## Data (verified at source)
Sudars 2020 (detection; CC BY 4.0) · Nigerian maize (domain test only; CC BY 4.0) · DeepWeeds (classification baseline; **inconsistent** licence statements) · CropAndWeed / CWFID not used (non-commercial / unclear).
Sudars caveats: **sessions are inferred** from file IDs, `crop` boxes almost only in the held-out group (hence one class), tiny objects (≈ 0.1% of the image), no negative images, greenhouse + field mixed.

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt && pip install -e .
python -m uvicorn backend.app.main:app --port 8000     # OpenAPI at /docs
cd frontend && npm install && npm run dev               # http://localhost:5173
pytest -q
```
Raw data is **not** versioned; see [docs/pt-BR/11_reproducao.md](docs/pt-BR/11_reproducao.md) for the full pipeline.

## Verification status
| Item | Status |
|---|---|
| 40 pytest, 5 vitest, 5 Playwright E2E | ✅ run |
| 3 notebooks | ✅ executed on real data |
| **Docker / docker-compose** | ❌ **written, never executed** (no Docker on this machine) |
| Real webcam | ❌ Chromium fake device only |
| Real edge hardware, quantisation, power | ❌ not measured |
| Field calibration / georeferencing | ❌ synthetic only |
| MLflow | ❌ not integrated |
| Code licence | GPL-3.0 (`LICENSE`); YOLO11 weights are **AGPL-3.0** (doc 10): review compatibility before redistributing |

## Repository, versioning and running on another machine

Published from the original development machine (macOS, Podman). This has **not** been run on GitHub Codespaces or Docker Desktop; if you run it there, treat it as a first verification and compare with the tests section.

- **Versioned:** `weedvision/`, `backend/`, `frontend/`, `scripts/`, notebooks, tests (40), `docs/pt-BR/`, `models/onnx/` (trained ONNX weights, ≈ 48 MB, derived from Ultralytics YOLO11, **AGPL-3.0**), `models/registry.json` and the result JSONs under `runs/` (`eval/`, `benchmark`, `drift`, `bootstrap`, `deepweeds/*.json`) that the API and docs read.
- **Not versioned:** `data/` (raw datasets; see `docs/pt-BR/02_dados.md`), `.pt` weights and other training artifacts in `runs/` (≈ 250 MB), `models/pretrained/` (Ultralytics re-downloads them).
- Full training (`make train`) needs the data and is slow; Docker/compose were **never executed** (see *Status de verificação*).
- **Code licence:** GPL-3.0 (`LICENSE`). Third-party data/model licences and attributions: see the sections and docs above.
