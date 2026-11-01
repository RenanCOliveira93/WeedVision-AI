import numpy as np
from weedvision import drift as D


def test_psi_identico_proximo_de_zero_e_deslocado_grande():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert D.psi(a, b) < 0.05
    assert D.psi(a, rng.normal(1.5, 1, 5000)) > 0.5


def test_compare_ks():
    rng = np.random.default_rng(1)
    ref = {"x": rng.normal(0, 1, 500)}
    same, shifted = D.compare(ref, {"x": rng.normal(0, 1, 500)})["x"], D.compare(ref, {"x": rng.normal(2, 1, 500)})["x"]
    assert same["ks"] < 0.12 and shifted["ks"] > 0.6 and shifted["ks_p"] < 1e-6


def test_image_stats(tmp_path):
    import cv2
    img = np.zeros((100, 200, 3), np.uint8); img[..., 1] = 200
    cv2.imwrite(str(tmp_path / "g.png"), img)
    s = D.image_stats(tmp_path / "g.png")
    assert s["largura_px"] == 200 and s["verde_relativo"] > 0.99 and s["nitidez_laplaciano"] < 1e-6
