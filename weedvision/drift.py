"""Estatísticas de imagem e métricas de data drift (PSI e KS) entre conjuntos.

Drift de COVARIÁVEIS (aparência), não de rótulos. PSI com decis do conjunto de referência; regra prática comum:
< 0,1 pequeno, 0,1–0,25 moderado, > 0,25 grande (convenção de mercado, não teorema). KS = distância máxima entre CDFs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURES = ["brilho", "contraste", "nitidez_laplaciano", "saturacao", "verde_relativo", "largura_px"]


def image_stats(path: Path, max_side: int = 320) -> dict:
    import cv2
    bgr = cv2.imread(str(path))
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    b, g, r = (small[..., i].astype(np.float32).mean() for i in range(3))
    return {"brilho": float(gray.mean()), "contraste": float(gray.std()), "nitidez_laplaciano": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "saturacao": float(hsv[..., 1].mean()), "verde_relativo": float(g / max(r + g + b, 1e-6)), "largura_px": float(w)}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(np.mean(ref), np.mean(cur)) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + eps
    q = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def compare(ref: dict[str, np.ndarray], cur: dict[str, np.ndarray]) -> dict:
    from scipy.stats import ks_2samp
    out = {}
    for f in ref:
        k = ks_2samp(ref[f], cur[f])
        out[f] = {"psi": psi(ref[f], cur[f]), "ks": float(k.statistic), "ks_p": float(k.pvalue),
                  "media_ref": float(np.mean(ref[f])), "media_cur": float(np.mean(cur[f]))}
    return out


def collect(paths: list[Path]) -> dict[str, np.ndarray]:
    rows = [image_stats(p) for p in paths]
    return {f: np.array([r[f] for r in rows]) for f in FEATURES}
