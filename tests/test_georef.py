import math
import numpy as np
import pytest

from weedvision import calibration as C, georef as G

K = G.Intrinsics(1000, 1000, 640, 360, 1280, 720)
POSE = G.Pose(-30.0, -51.0, 1.5, yaw_deg=40, pitch_deg=50, roll_deg=3)


def test_ida_e_volta_sintetica():
    en = np.array([[0.5, 2.0], [-1.0, 3.5], [1.2, 1.7]])
    uv = G.enu_to_pixels(en, K, POSE)
    np.testing.assert_allclose(G.pixels_to_enu(uv, K, POSE), en, atol=1e-6)


def test_nadir_centro_fica_embaixo_da_camera():
    p = G.Pose(-30, -51, 2.0, 0, 90)
    np.testing.assert_allclose(G.pixels_to_enu([[640, 360]], K, p), [[0, 0]], atol=1e-9)
    # 100 px à direita do centro, a 2 m de altura, f=1000 → 0,2 m a Leste (rumo 0 → direita = Leste)
    np.testing.assert_allclose(G.pixels_to_enu([[740, 360]], K, p), [[0.2, 0]], atol=1e-9)


def test_rumo_leste_olhando_a_frente():
    p = G.Pose(-30, -51, 1.0, 90, 45)  # olhando para o Leste, 45° para baixo → centro a 1 m ao Leste
    np.testing.assert_allclose(G.pixels_to_enu([[640, 360]], K, p), [[1.0, 0.0]], atol=1e-9)


def test_recusa_acima_do_horizonte_e_longe():
    with pytest.raises(G.GeorefError):
        G.pixels_to_enu([[640, 0]], K, G.Pose(0, 0, 1.5, 0, 10))   # topo da imagem aponta acima do horizonte
    with pytest.raises(G.GeorefError):
        G.pixels_to_enu([[640, 360]], K, G.Pose(0, 0, 1.5, 0, 2), max_range_m=20)


def test_recusa_pose_e_calibracao_invalidas():
    with pytest.raises(G.GeorefError):
        G.pixels_to_enu([[1, 1]], G.Intrinsics(0, 1000, 640, 360, 1280, 720), POSE)
    with pytest.raises(G.GeorefError):
        G.pixels_to_enu([[1, 1]], K, G.Pose(0, 0, -1, 0, 45))


def test_latlon_norte_100m():
    ll = G.enu_to_latlon(np.array([[0.0, 100.0]]), -30.0, -51.0)
    assert ll[0, 1] == pytest.approx(-51.0, abs=1e-9) and (ll[0, 0] - (-30.0)) == pytest.approx(100 / 110_900, rel=0.01)


def test_geojson_aborta_sem_coordenadas_parciais():
    ok = [{"xyxy": [600, 500, 700, 600], "score": 0.9, "label": "weed"}]
    gj = G.detections_to_geojson(ok, K, POSE, "x.jpg")
    assert gj["features"][0]["geometry"]["type"] == "Point" and "aviso" in gj["properties"]
    with pytest.raises(G.GeorefError):
        G.detections_to_geojson(ok + [{"xyxy": [0, 0, 10, 5], "score": 0.5, "label": "weed"}], K, G.Pose(0, 0, 1.5, 0, 10), "x.jpg")


def test_sensibilidade_pitch_cresce_perto_do_horizonte():
    near = G.ground_error_sensitivity([[640, 500]], K, G.Pose(0, 0, 1.5, 0, 60))["pitch_erro_m"]
    far = G.ground_error_sensitivity([[640, 500]], K, G.Pose(0, 0, 1.5, 0, 25))["pitch_erro_m"]
    assert far > near > 0


def test_calibracao_recupera_intrinsecos_sinteticos():
    import cv2
    rng = np.random.default_rng(0)
    Kt = np.array([[900, 0, 650], [0, 905, 350], [0, 0, 1.0]]); dist = np.array([-0.1, 0.05, 0.0, 0.0, 0.0])
    objp = np.zeros((54, 3), np.float32); objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.03
    objs, imgs = [], []
    for _ in range(15):
        rvec = rng.normal(0, 0.25, 3); tvec = np.array([-0.12, -0.08, 0.7]) + rng.normal(0, [0.05, 0.05, 0.1])
        pts, _ = cv2.projectPoints(objp, rvec, tvec, Kt, dist)
        objs.append(objp); imgs.append((pts.reshape(-1, 2) + rng.normal(0, 0.1, (54, 2))).astype(np.float32))
    cam, rms = C.calibrate_from_points(objs, imgs, (1280, 720))
    assert cam.fx == pytest.approx(900, rel=0.02) and cam.fy == pytest.approx(905, rel=0.02) and rms < 0.5
    with pytest.raises(ValueError):
        C.calibrate_from_points(objs[:3], imgs[:3], (1280, 720))
