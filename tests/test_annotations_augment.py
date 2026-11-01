import json
import numpy as np
from PIL import Image

from weedvision import annotations as AN, augment as AU


def _mk(tmp_path, lines):
    (tmp_path / "images").mkdir(); (tmp_path / "labels").mkdir()
    Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(tmp_path / "images/a.jpg")
    (tmp_path / "labels/a.txt").write_text("\n".join(lines) + "\n")
    return tmp_path / "images", tmp_path / "labels"


def test_yolo_valido(tmp_path):
    i, l = _mk(tmp_path, ["0 0.5 0.5 0.2 0.2"])
    assert AN.validate_yolo(i, l, 1)["ok"]


def test_yolo_detecta_problemas(tmp_path):
    i, l = _mk(tmp_path, ["3 0.5 0.5 0.2 0.2", "0 0.99 0.5 0.2 0.2", "0 0.5 0.5", "0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2", "0 0.1 0.1 0.001 0.001"])
    r = AN.validate_yolo(i, l, 1)
    assert not r["ok"]
    for t in ("classe_invalida", "coordenada_fora_da_imagem", "linha_malformada", "duplicata", "caixa_minuscula"):
        assert t in r["problemas_por_tipo"], t


def test_coco(tmp_path):
    d = {"images": [{"id": 1, "width": 100, "height": 100}], "categories": [{"id": 0}],
         "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
                         {"id": 2, "image_id": 1, "category_id": 5, "bbox": [90, 90, 20, 20]},
                         {"id": 2, "image_id": 9, "category_id": 0, "bbox": [0, 0, 0, 5]}]}
    p = tmp_path / "c.json"; p.write_text(json.dumps(d))
    r = AN.validate_coco(p)
    assert {"categoria_inexistente", "coordenada_fora_da_imagem", "id_duplicado", "image_id_inexistente"} <= set(r["problemas_por_tipo"])


def test_augment_caixas_validas():
    import random
    random.seed(0); np.random.seed(0)
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    boxes = [(100, 100, 220, 200), (400, 300, 520, 420)]
    aug = AU.build()
    for _ in range(50):
        im, b, lab = AU.apply(aug, img, boxes, [0, 0])
        assert im.shape == img.shape and len(b) == len(lab) <= 2
        for x1, y1, x2, y2 in b:
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
