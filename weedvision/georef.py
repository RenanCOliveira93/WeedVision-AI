"""Georreferenciamento de pixels por interseção raio–solo (modelo pinhole + pose + solo PLANO).

NUNCA estima coordenadas sem calibração e pose válidas: `GeorefError` é lançado. Limitações que o código NÃO resolve:
solo plano (relevo/inclinação do terreno causa erro), pose com erro (GNSS/IMU), distorção residual, e horizonte
(pixels perto do horizonte têm erro enorme → recusados acima de `max_range_m`).
Convenções: ENU (x=Leste, y=Norte, z=Cima). Câmera: x direita, y baixo, z à frente.
`yaw_deg` = rumo (horário a partir do Norte); `pitch_deg` = inclinação PARA BAIXO (0 = horizonte, 90 = nadir); `roll_deg` = giro em torno do eixo óptico.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


class GeorefError(ValueError):
    """Calibração/pose ausente ou inválida, ou pixel que não intercepta o solo de forma confiável."""


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple = field(default_factory=tuple)  # (k1, k2, p1, p2[, k3]) estilo OpenCV; vazio = sem distorção

    def validate(self):
        if not (self.fx > 0 and self.fy > 0 and self.width > 0 and self.height > 0):
            raise GeorefError("intrínsecos inválidos (fx, fy, largura, altura devem ser > 0)")
        if not (0 < self.cx < self.width and 0 < self.cy < self.height):
            raise GeorefError("ponto principal fora da imagem")


@dataclass(frozen=True)
class Pose:
    lat: float
    lon: float
    height_m: float     # altura da câmera sobre o solo (não altitude elipsoidal)
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0

    def validate(self):
        if not (-90 <= self.lat <= 90 and -180 <= self.lon <= 180):
            raise GeorefError("lat/lon inválidos")
        if not self.height_m > 0:
            raise GeorefError("altura da câmera deve ser > 0 m")
        if not 0 <= self.pitch_deg <= 90:
            raise GeorefError("pitch deve estar em [0, 90] graus (inclinação para baixo)")


def _rotation(p: Pose) -> np.ndarray:
    ps, th, ph = map(math.radians, (p.yaw_deg, p.pitch_deg, p.roll_deg))
    f = np.array([math.sin(ps) * math.cos(th), math.cos(ps) * math.cos(th), -math.sin(th)])
    r = np.array([math.cos(ps), -math.sin(ps), 0.0])
    if abs(f[2]) > 1 - 1e-12:  # nadir: rumo define a direção "direita"
        r = np.array([math.cos(ps), -math.sin(ps), 0.0])
    d = np.cross(f, r)
    r2, d2 = r * math.cos(ph) + d * math.sin(ph), -r * math.sin(ph) + d * math.cos(ph)
    return np.stack([r2, d2, f], axis=1)  # colunas: direita, baixo, frente


def _undistort(K: Intrinsics, uv: np.ndarray) -> np.ndarray:
    if not K.dist:
        return np.c_[(uv[:, 0] - K.cx) / K.fx, (uv[:, 1] - K.cy) / K.fy]
    import cv2
    M = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1.0]])
    return cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), M, np.array(K.dist, float)).reshape(-1, 2)


def pixels_to_enu(uv, K: Intrinsics, pose: Pose, max_range_m: float = 100.0) -> np.ndarray:
    """(n,2) pixels → (n,2) ENU em metros relativo à câmera. Lança GeorefError se algum raio não atinge o solo em ≤ max_range_m."""
    K.validate(); pose.validate()
    uv = np.asarray(uv, float).reshape(-1, 2)
    n = _undistort(K, uv)
    rays = (_rotation(pose) @ np.c_[n, np.ones(len(n))].T).T
    if np.any(rays[:, 2] >= -1e-9):
        raise GeorefError("raio acima do horizonte: não intercepta o solo")
    t = -pose.height_m / rays[:, 2]
    en = rays[:, :2] * t[:, None]
    if np.any(np.hypot(en[:, 0], en[:, 1]) > max_range_m):
        raise GeorefError(f"ponto a mais de {max_range_m:g} m da câmera: erro de pose/solo plano domina; recusado")
    return en


def enu_to_latlon(en: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Deslocamento ENU (m) → (lat, lon). Elipsoide WGS84 via pyproj.Geod se disponível; senão aproximação local (< ~1 km)."""
    try:
        from pyproj import Geod
        g = Geod(ellps="WGS84")
        az = np.degrees(np.arctan2(en[:, 0], en[:, 1]))
        dist = np.hypot(en[:, 0], en[:, 1])
        lon, lat, _ = g.fwd(np.full(len(en), lon0), np.full(len(en), lat0), az, dist)
        return np.c_[lat, lon]
    except ImportError:
        k_lat = 111132.92
        k_lon = 111412.84 * math.cos(math.radians(lat0))
        return np.c_[lat0 + en[:, 1] / k_lat, lon0 + en[:, 0] / k_lon]


def pixels_to_latlon(uv, K: Intrinsics, pose: Pose, max_range_m: float = 100.0) -> np.ndarray:
    return enu_to_latlon(pixels_to_enu(uv, K, pose, max_range_m), pose.lat, pose.lon)


def enu_to_pixels(en, K: Intrinsics, pose: Pose) -> np.ndarray:
    """Projeção direta (usada para validar a inversa em testes sintéticos). Pontos ENU relativos à câmera, no solo."""
    en = np.asarray(en, float).reshape(-1, 2)
    P = np.c_[en, np.zeros(len(en))] - np.array([0, 0, pose.height_m])
    c = (_rotation(pose).T @ P.T).T
    return np.c_[K.fx * c[:, 0] / c[:, 2] + K.cx, K.fy * c[:, 1] / c[:, 2] + K.cy]


def detections_to_geojson(dets: list[dict], K: Intrinsics, pose: Pose, source: str, max_range_m: float = 100.0) -> dict:
    """dets: [{'xyxy': [x1,y1,x2,y2], 'score': s, 'label': str}]. Ponto = centro da base da caixa (contato com o solo
    assumido). Qualquer falha de georreferenciamento aborta a exportação inteira (sem coordenadas parciais)."""
    base = np.array([[(d["xyxy"][0] + d["xyxy"][2]) / 2, d["xyxy"][3]] for d in dets]).reshape(-1, 2)
    ll = pixels_to_latlon(base, K, pose, max_range_m) if len(dets) else np.zeros((0, 2))
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
              "properties": {"label": d["label"], "score": round(float(d["score"]), 4), "xyxy_px": [round(float(v), 1) for v in d["xyxy"]]}}
             for d, (lat, lon) in zip(dets, ll)]
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"source": source, "crs": "EPSG:4326", "method": "pinhole + pose + solo plano",
                           "camera_pose": pose.__dict__, "aviso": "posição estimada; sem validação de campo; erro depende da pose e do relevo"}}


def ground_error_sensitivity(uv, K: Intrinsics, pose: Pose, d_pitch_deg=1.0, d_height_m=0.1) -> dict:
    """Erro (m) no solo causado por erro de pose, por perturbação numérica — para ilustrar a sensibilidade."""
    base = pixels_to_enu(uv, K, pose)
    out = {}
    for name, p2 in {"pitch": Pose(**{**pose.__dict__, "pitch_deg": min(90, pose.pitch_deg + d_pitch_deg)}),
                     "altura": Pose(**{**pose.__dict__, "height_m": pose.height_m + d_height_m}),
                     "rumo": Pose(**{**pose.__dict__, "yaw_deg": pose.yaw_deg + d_pitch_deg})}.items():
        out[f"{name}_erro_m"] = float(np.max(np.hypot(*(pixels_to_enu(uv, K, p2) - base).T)))
    return out
