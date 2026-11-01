from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = sorted((ROOT / "sample_data").glob("sudars_test_1*.jpg"))
c = TestClient(app)
needs_model = pytest.mark.skipif(not (ROOT / "models/onnx/yolo11n_weed.onnx").exists() or not (ROOT / "models/registry.json").exists(), reason="modelo ONNX/registry ausente")


def test_health_e_limites():
    assert c.get("/health").json()["status"] == "ok"
    lim = c.get("/limits").json()
    assert "NÃO é recomendação de pulverização" in lim["advisory"]
    assert any("café brasileiro" in x for x in lim["limites"])


@needs_model
def test_modelos_e_inferencia():
    ms = c.get("/models").json()
    assert any(m["id"] == "yolo11n" for m in ms)
    with open(SAMPLE[0], "rb") as f:
        r = c.post("/infer/image", files={"file": ("a.jpg", f, "image/jpeg")}, data={"model_id": "yolo11n", "conf": "0.25", "with_image": "true"})
    assert r.status_code == 200
    j = r.json()
    assert j["count"] == len(j["detections"]) and j["image"] == {"width": 1280, "height": 720} and j["annotated_jpeg_base64"]
    assert all(set(d) == {"xyxy", "score", "label"} and d["score"] >= 0.25 for d in j["detections"])
    assert "advisory" in j and j["timings_ms"]["infer_ms"] > 0


@needs_model
def test_entradas_invalidas():
    assert c.post("/infer/image", files={"file": ("a.txt", b"nao sou imagem", "text/plain")}, data={"model_id": "yolo11n"}).status_code == 415
    with open(SAMPLE[0], "rb") as f:
        assert c.post("/infer/image", files={"file": ("a.jpg", f, "image/jpeg")}, data={"model_id": "nao_existe"}).status_code == 404
    with open(SAMPLE[0], "rb") as f:
        assert c.post("/infer/image", files={"file": ("a.jpg", f, "image/jpeg")}, data={"model_id": "yolo11n", "conf": "5"}).status_code == 422


@needs_model
def test_video_sintetico():
    v = ROOT / "sample_data/demo_pan_synthetic.mp4"
    with open(v, "rb") as f:
        r = c.post("/infer/video", files={"file": ("v.mp4", f, "video/mp4")}, data={"model_id": "yolo11n", "stride": "10"})
    assert r.status_code == 200 and r.json()["frames_analisados"] == 6 and "NÃO são deduplicados" in r.json()["aviso"]


DETS = [{"xyxy": [600, 500, 700, 600], "score": 0.9, "label": "weed"}]
CAM = {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360, "width": 1280, "height": 720}
POSE = {"lat": -23.0, "lon": -46.0, "height_m": 1.5, "yaw_deg": 0, "pitch_deg": 50}


def test_geojson_exige_calibracao_e_pose():
    assert c.post("/export/geojson", json={"detections": DETS}).status_code == 422
    assert c.post("/export/geojson", json={"detections": DETS, "camera": CAM}).status_code == 422
    r = c.post("/export/geojson", json={"detections": DETS, "camera": CAM, "pose": POSE})
    assert r.status_code == 200 and r.json()["features"][0]["geometry"]["type"] == "Point"
    above = [{"xyxy": [600, 200, 700, 300], "score": 0.9, "label": "weed"}]  # base da caixa acima do horizonte com pitch 2°
    bad = c.post("/export/geojson", json={"detections": above, "camera": CAM, "pose": {**POSE, "pitch_deg": 2}})
    assert bad.status_code == 422 and "horizonte" in bad.json()["detail"]


def test_csv_em_pixels():
    r = c.post("/export/csv", json={"detections": DETS})
    assert r.text.splitlines()[0] == "label,score,x1,y1,x2,y2" and "weed,0.9,600,500,700,600" in r.text


@pytest.mark.skipif(not (ROOT / "runs/eval").exists(), reason="sem avaliações em runs/eval")
def test_avaliacoes_com_nan_viram_null_e_nao_500():
    r = c.get("/evaluations")
    assert r.status_code == 200
    maize = [x for x in r.json() if x["dataset"].startswith("maize")]
    assert maize and all(x["precision"] is None for x in maize)  # NaN (nenhuma predição) → null
    d = c.get(f"/evaluations/{maize[0]['arquivo']}")
    assert d.status_code == 200 and d.json()["prf"]["weed"]["precision"] is None
