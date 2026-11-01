import numpy as np
import pytest

from weedvision import onnx_infer as O


def test_letterbox_preserva_proporcao():
    img = np.full((720, 1280, 3), 200, np.uint8)
    out, r, (px, py) = O.letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 0.5 and (px, py) == (0, 140)
    assert (out[0, 0] == 114).all() and (out[320, 320] == 200).all()


def test_nms_remove_sobrepostas():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60.]]); s = np.array([0.9, 0.8, 0.7])
    assert O.nms(b, s, 0.6) == [0, 2]


def test_decode_mapeia_para_coordenadas_originais():
    out = np.zeros((1, 5, 3)); out[0, :, 0] = [320, 320, 100, 100, 0.9]   # caixa no centro do tensor
    d = O.decode(out, scale=0.5, pad=(0, 140), orig_hw=(720, 1280), conf=0.25)
    assert len(d) == 1 and d[0].xyxy == pytest.approx((540, 260, 740, 460)) and d[0].cls == 0
    assert O.decode(out, 0.5, (0, 140), (720, 1280), conf=0.95) == []
