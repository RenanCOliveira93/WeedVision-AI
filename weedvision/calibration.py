"""Calibração de câmera com tabuleiro de xadrez (OpenCV). Resultado serializável e reutilizável em georef.Intrinsics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .georef import Intrinsics


def calibrate_from_points(obj_pts, img_pts, size_wh):
    """obj_pts/img_pts: listas por imagem de (n,3)/(n,2). Retorna (Intrinsics, rms_reprojecao_px)."""
    import cv2
    if len(obj_pts) < 8:
        raise ValueError("use ≥ 8 imagens do tabuleiro em poses variadas (inclinado, perto/longe, cantos da imagem)")
    rms, K, dist, _, _ = cv2.calibrateCamera([np.asarray(o, np.float32) for o in obj_pts], [np.asarray(i, np.float32) for i in img_pts], size_wh, None, None)
    return Intrinsics(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), size_wh[0], size_wh[1], tuple(float(d) for d in dist.ravel()[:5])), float(rms)


def calibrate_from_images(image_paths, pattern=(9, 6), square_m=0.025):
    """Detecta cantos internos (findChessboardCornersSB) e calibra. Falha com mensagem clara se poucas imagens servirem."""
    import cv2
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    objs, imgs, size = [], [], None
    for p in image_paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCornersSB(g, pattern)
        if ok:
            objs.append(objp); imgs.append(c.reshape(-1, 2))
    if len(objs) < 8:
        raise ValueError(f"apenas {len(objs)} imagens com tabuleiro detectado (mínimo 8)")
    return calibrate_from_points(objs, imgs, size)
