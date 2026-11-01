"""Benchmark de latência/FPS/memória, uma configuração por subprocesso (RSS pico justo). Batch 1, 100 imagens 1280×720, 10 de aquecimento.
Latência = ponta a ponta a partir do array RGB em memória (pré-processamento + inferência + pós-processamento). Sem E/S de disco.
Uso: benchmark.py  (orquestra)  |  benchmark.py --worker <config>
"""
import glob, json, os, platform, resource, subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
N, WARM = 100, 10
CONFIGS = {
    "yolo11n · onnxruntime CPU": ("ort", "models/onnx/yolo11n_weed.onnx", ["CPUExecutionProvider"]),
    "yolo11s · onnxruntime CPU": ("ort", "models/onnx/yolo11s_weed.onnx", ["CPUExecutionProvider"]),
    "yolo11n · onnxruntime CoreML": ("ort", "models/onnx/yolo11n_weed.onnx", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    "yolo11s · onnxruntime CoreML": ("ort", "models/onnx/yolo11s_weed.onnx", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    "yolo11n · PyTorch CPU": ("yolo", "runs/yolo11n_weed/weights/best.pt", "cpu"),
    "yolo11s · PyTorch CPU": ("yolo", "runs/yolo11s_weed/weights/best.pt", "cpu"),
    "yolo11n · PyTorch MPS": ("yolo", "runs/yolo11n_weed/weights/best.pt", "mps"),
    "yolo11s · PyTorch MPS": ("yolo", "runs/yolo11s_weed/weights/best.pt", "mps"),
    "frcnn_mnv3 · PyTorch CPU": ("frcnn", "runs/frcnn_mnv3_weed/model.pt", "cpu"),
    "frcnn_mnv3 · PyTorch MPS": ("frcnn", "runs/frcnn_mnv3_weed/model.pt", "mps"),
}


def images():
    import numpy as np
    from PIL import Image
    fs = sorted(glob.glob(str(root / "data/processed/sudars_weed/images/test/*.jpg")))
    fs = [f for f in fs if Image.open(f).size == (1280, 720)][:N + WARM]
    return [np.asarray(Image.open(f).convert("RGB")) for f in fs]


def worker(name):
    kind, path, extra = CONFIGS[name]
    imgs = images()
    if kind == "ort":
        from weedvision.onnx_infer import OnnxDetector
        d = OnnxDetector(str(root / path), providers=extra)
        used = d.providers
        fn = lambda im: d(im, conf=0.25)  # noqa: E731
    elif kind == "yolo":
        from ultralytics import YOLO
        m = YOLO(str(root / path)); used = [f"torch:{extra}"]
        fn = lambda im: m.predict(im[..., ::-1], conf=0.25, iou=0.6, imgsz=640, device=extra, verbose=False)  # noqa: E731  (ultralytics espera BGR)
    else:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        import torch
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import torchvision.transforms.functional as TF
        m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, min_size=640, max_size=1138, box_score_thresh=0.25)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.load_state_dict(torch.load(root / path, map_location="cpu")); m.eval().to(extra); used = [f"torch:{extra}"]
        def fn(im):
            with torch.no_grad():
                out = m([TF.to_tensor(im).to(extra)])[0]
                return out["boxes"].cpu()
    for im in imgs[:WARM]:
        fn(im)
    ts = []
    for im in imgs[WARM:WARM + N]:
        t = time.perf_counter(); fn(im); ts.append((time.perf_counter() - t) * 1e3)
    import numpy as np
    ts = np.array(ts)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    print("RESULT" + json.dumps({"nome": name, "mean_ms": ts.mean(), "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
                                 "fps": 1000 / ts.mean(), "rss_mb": rss, "n": len(ts), "providers": used}))


def main():
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    out = {"hardware": f"{cpu}, {os.cpu_count()} núcleos, {platform.platform()}",
           "nota": "Batch 1, 100 imagens 1280×720, 10 de aquecimento; latência ponta a ponta (pré+infer+pós) sem E/S de disco. "
                   "RSS = pico do processo: inclui bibliotecas, o modelo E ≈ 300 MB de imagens de teste pré-carregadas (110 × 1280×720×3) — serve para comparar configurações entre si, NÃO é o consumo de borda. Uma execução por configuração; sem repetição entre dias/temperatura.",
           "resultados": [], "falhas": []}
    for name in CONFIGS:
        if not (root / CONFIGS[name][1]).exists():
            out["falhas"].append({"nome": name, "motivo": "arquivo do modelo ausente"}); continue
        p = subprocess.run([sys.executable, __file__, "--worker", name], capture_output=True, text=True, cwd=root)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if line:
            r = json.loads(line[-1][6:]); out["resultados"].append(r); print(f"{name}: {r['mean_ms']:.1f} ms, {r['fps']:.1f} FPS, {r['rss_mb']:.0f} MB {r['providers']}", flush=True)
        else:
            out["falhas"].append({"nome": name, "motivo": (p.stderr or p.stdout)[-300:]}); print("FALHOU", name, (p.stderr or "")[-200:], flush=True)
    (root / "runs/benchmark.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    worker(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--worker" else main()
