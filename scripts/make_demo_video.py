"""Gera um vídeo de demonstração SINTÉTICO: janela deslizante sobre uma foto real (não é filmagem de campo)."""
import glob, sys
from pathlib import Path
import cv2, numpy as np

root = Path(__file__).resolve().parents[1]
src = sorted(glob.glob(str(root / "sample_data/sudars_test_*.jpg")))[0]
img = cv2.imread(src); h, w = img.shape[:2]
big = cv2.resize(img, (int(w * 1.0), int(h * 1.0)))
vw, vh, n = 640, 480, 60
out = cv2.VideoWriter(str(root / "sample_data/demo_pan_synthetic.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (vw, vh))
for k in range(n):
    x = int((w - vw) * k / (n - 1)); out.write(big[0:vh, x:x + vw])
out.release(); print("ok", src)
