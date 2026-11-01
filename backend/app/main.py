"""API do WeedVision AI. Só DETECTA e mede; não recomenda pulverização nem controla atuadores."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from weedvision import georef
from weedvision.onnx_infer import OnnxDetector

ROOT = Path(os.environ.get("WEEDVISION_ROOT", Path(__file__).resolve().parents[2]))
MAX_PIXELS = 40_000_000
MAX_VIDEO_FRAMES = 300
ADVISORY = ("Detecção de planta daninha NÃO é recomendação de pulverização: aplicação exige critério agronômico "
            "(espécie, estádio, limiar de dano econômico, registro do produto) e validação em campo.")

app = FastAPI(title="WeedVision AI", version="0.1.0",
              description="Detecção de plantas daninhas em imagens (pesquisa/demo). Sem controle de atuadores.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])
_detectors: dict[str, OnnxDetector] = {}


def _finite(x):
    """NaN/Inf não são JSON válido (o FastAPI responderia 500): vira null. Ex.: precisão sem nenhuma predição = NaN."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


def registry() -> list[dict]:
    p = ROOT / "models/registry.json"
    return json.loads(p.read_text()) if p.exists() else []


def _entry(model_id: str) -> dict:
    for m in registry():
        if m["id"] == model_id:
            return m
    raise HTTPException(404, f"modelo '{model_id}' não existe no registry")


def _detector(model_id: str) -> tuple[OnnxDetector, dict]:
    e = _entry(model_id)
    if not e.get("onnx"):
        raise HTTPException(422, f"modelo '{model_id}' não tem versão ONNX para inferência na API (apenas avaliação offline)")
    if model_id not in _detectors:
        path = ROOT / e["onnx"]
        if not path.exists():
            raise HTTPException(503, f"arquivo do modelo ausente: {e['onnx']}")
        _detectors[model_id] = OnnxDetector(str(path))
    return _detectors[model_id], e


def _decode_image(raw: bytes) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(raw))
        if im.width * im.height > MAX_PIXELS:
            raise HTTPException(413, "imagem grande demais")
        return np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "arquivo não é uma imagem válida (JPEG/PNG)")


def _run(img, model_id, conf, iou):
    det, entry = _detector(model_id)
    t: dict = {}
    d = det(img, conf=conf, iou=iou, timings=t)
    names = entry["classes"]
    dets = [{"xyxy": [round(v, 1) for v in x.xyxy], "score": round(x.score, 4), "label": names[x.cls]} for x in d]
    return dets, t, entry


def annotate(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    th = max(2, round(max(out.shape[:2]) / 400))
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 230), th)
        cv2.putText(out, f"{d['label']} {d['score']:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, th / 3 + 0.3, (0, 0, 230), max(1, th - 1))
    cv2.putText(out, f"deteccoes: {len(dets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)
    cv2.putText(out, f"deteccoes: {len(dets)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


@app.get("/health", tags=["sistema"])
def health():
    return {"status": "ok", "modelos": len(registry()), "versao": app.version}


@app.get("/models", tags=["modelos"])
def models():
    return [{k: v for k, v in m.items() if k != "metrics"} | {"tem_metricas": bool(m.get("metrics"))} for m in registry()]


@app.get("/models/{model_id}", tags=["modelos"])
def model_detail(model_id: str):
    return _entry(model_id)


@app.get("/evaluations", tags=["avaliação"])
def evaluations():
    """Resumo de todas as avaliações salvas em runs/eval (geradas pelos scripts; nada é digitado à mão)."""
    out = []
    for p in sorted((ROOT / "runs/eval").glob("*.json")):
        r = _finite(json.loads(p.read_text()))
        prf = r["prf"].get("weed", {})
        out.append({"arquivo": p.name, "modelo": r["model"], "split": r["split"], "dataset": r["dataset"], "n_images": r["n_images"],
                    "map50": r["map50"], "map50_95": r["map50_95"], "precision": prf.get("precision"), "recall": prf.get("recall"),
                    "f1": prf.get("f1"), "conf": r["conf_threshold"], "fp_on_crop": r.get("fp_on_crop")})
    return out


@app.get("/evaluations/{name}", tags=["avaliação"])
def evaluation_detail(name: str):
    p = ROOT / "runs/eval" / f"{Path(name).name.removesuffix('.json')}.json"  # aceita com ou sem .json; Path(...).name bloqueia path traversal
    if not p.exists():
        raise HTTPException(404, "avaliação não encontrada")
    return _finite(json.loads(p.read_text()))


@app.get("/benchmark", tags=["avaliação"])
def benchmark():
    p = ROOT / "runs/benchmark.json"
    if not p.exists():
        raise HTTPException(404, "benchmark ainda não executado (scripts/benchmark.py)")
    return json.loads(p.read_text())


@app.get("/classification", tags=["avaliação"])
def classification():
    """Baseline de CLASSIFICAÇÃO (DeepWeeds): acurácia por split — não é detecção."""
    res = {}
    for p in sorted((ROOT / "runs/deepweeds").glob("cls_*.json")):
        r = json.loads(p.read_text())
        key = r["split"] if r.get("seed", 0) == 0 else f'{r["split"]}_s{r["seed"]}'
        res[key] = {"split": r["split"], "seed": r.get("seed", 0), "vazamento": r["leakage"], "test": {k: r["test"][k] for k in ("n", "accuracy", "macro_f1", "recall_por_classe")}}
    if not res:
        raise HTTPException(404, "baseline de classificação ainda não executado")
    return {"tarefa": "classificação de imagem inteira (sem caixas)", "resultados": res}


@app.post("/infer/image", tags=["inferência"])
async def infer_image(file: UploadFile = File(...), model_id: str = Form(...), conf: float = Form(0.25, ge=0.01, le=0.99),
                      iou: float = Form(0.6, ge=0.1, le=0.95), with_image: bool = Form(False)):
    raw = await file.read()
    img = _decode_image(raw)
    dets, t, entry = _run(img, model_id, conf, iou)
    res = {"model_id": model_id, "image": {"width": img.shape[1], "height": img.shape[0]}, "conf": conf, "iou": iou,
           "count": len(dets), "detections": dets, "timings_ms": {k: round(v, 2) for k, v in t.items()},
           "task": "detecção (caixas)", "domain_note": entry.get("domain_note"), "advisory": ADVISORY}
    if with_image:
        buf = io.BytesIO(); Image.fromarray(annotate(img, dets)).save(buf, "JPEG", quality=88)
        res["annotated_jpeg_base64"] = base64.b64encode(buf.getvalue()).decode()
    return res


@app.post("/infer/image/annotated", tags=["inferência"], response_class=Response)
async def infer_image_annotated(file: UploadFile = File(...), model_id: str = Form(...), conf: float = Form(0.25), iou: float = Form(0.6)):
    img = _decode_image(await file.read())
    dets, _, _ = _run(img, model_id, conf, iou)
    buf = io.BytesIO(); Image.fromarray(annotate(img, dets)).save(buf, "JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"X-Detection-Count": str(len(dets))})


@app.post("/infer/video", tags=["inferência"])
async def infer_video(file: UploadFile = File(...), model_id: str = Form(...), conf: float = Form(0.25), stride: int = Form(5, ge=1, le=60)):
    """Processa 1 a cada `stride` quadros (máx. 300 quadros analisados). Devolve contagem por quadro; sem rastreamento."""
    suffix = Path(file.filename or "v.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await file.read()); tmp = f.name
    try:
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(415, "vídeo não pôde ser aberto")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames, i = [], 0
        while len(frames) < MAX_VIDEO_FRAMES:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                dets, t, _ = _run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), model_id, conf, 0.6)
                frames.append({"frame": i, "t_s": round(i / fps, 3) if fps else None, "count": len(dets), "infer_ms": round(t["infer_ms"], 2)})
            i += 1
        cap.release()
    finally:
        os.unlink(tmp)
    return {"model_id": model_id, "fps_video": fps, "stride": stride, "frames_analisados": len(frames), "frames": frames,
            "aviso": "contagem por quadro; objetos repetidos entre quadros NÃO são deduplicados (sem rastreamento)", "advisory": ADVISORY}


class Camera(BaseModel):
    fx: float; fy: float; cx: float; cy: float; width: int; height: int
    dist: list[float] = Field(default_factory=list)


class PoseIn(BaseModel):
    lat: float; lon: float; height_m: float; yaw_deg: float; pitch_deg: float; roll_deg: float = 0.0


class GeoRequest(BaseModel):
    detections: list[dict]
    camera: Camera | None = None
    pose: PoseIn | None = None
    source: str = "upload"
    max_range_m: float = 100.0


@app.post("/export/geojson", tags=["exportação"])
def export_geojson(req: GeoRequest):
    """GeoJSON só com calibração E pose válidas; caso contrário 422 (nunca inventa coordenadas a partir de pixels)."""
    if req.camera is None or req.pose is None:
        raise HTTPException(422, "georreferenciamento exige calibração da câmera (intrínsecos) E pose (lat, lon, altura, rumo, inclinação). "
                                 "Sem isso, exporte CSV em pixels.")
    try:
        return georef.detections_to_geojson(req.detections, georef.Intrinsics(**{**req.camera.model_dump(), "dist": tuple(req.camera.dist)}),
                                            georef.Pose(**req.pose.model_dump()), req.source, req.max_range_m)
    except georef.GeorefError as e:
        raise HTTPException(422, str(e))


@app.post("/export/csv", tags=["exportação"], response_class=Response)
def export_csv(req: GeoRequest):
    """CSV em coordenadas de PIXEL (sem georreferenciamento)."""
    rows = ["label,score,x1,y1,x2,y2"] + [f"{d['label']},{d['score']},{','.join(str(v) for v in d['xyxy'])}" for d in req.detections]
    return Response("\n".join(rows) + "\n", media_type="text/csv")


@app.get("/limits", tags=["sistema"])
def limits():
    return {"advisory": ADVISORY,
            "limites": [
                "Treinado em Sudars et al. 2020 (câmeras próximas ao solo; vasos em estufa e campo aberto, sem indicação por imagem); NÃO validado para café brasileiro.",
                "Teste zero-shot em milho (Nigéria) falhou — ver /evaluations; anotações dos dois conjuntos têm protocolos diferentes.",
                "Classe 'crop' não é aprendível com este split (3 caixas no treino).",
                "Segmentação NÃO foi treinada; só detecção por caixas.",
                "Coordenadas só com calibração + pose; solo plano; sem validação de campo.",
                "Benchmark feito em Apple M4 Max, não em hardware de borda real.",
                "Sem controle de atuadores."]}
