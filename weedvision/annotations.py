"""Validação de anotações YOLO (.txt) e COCO (.json). Retorna problemas estruturados; não corrige nada em silêncio."""
from __future__ import annotations

import json
from pathlib import Path


def validate_yolo(images_dir: Path, labels_dir: Path, n_classes: int, min_px: float = 2.0) -> dict:
    """Checa: imagem sem label/label órfão, linha malformada, classe fora de [0,n), coordenadas fora de [0,1],
    caixa minúscula, duplicata exata. `min_px` usa a dimensão real da imagem (PIL)."""
    from PIL import Image
    imgs = {p.stem: p for p in Path(images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    labs = {p.stem: p for p in Path(labels_dir).glob("*.txt")}
    issues, n_boxes = [], 0
    for s in sorted(set(labs) - set(imgs)):
        issues.append({"arquivo": s, "tipo": "label_orfao"})
    for s in sorted(set(imgs) - set(labs)):
        issues.append({"arquivo": s, "tipo": "imagem_sem_label_file"})  # imagem negativa precisa de .txt vazio
    for s in sorted(set(imgs) & set(labs)):
        w, h = Image.open(imgs[s]).size
        seen = set()
        for ln, line in enumerate(labs[s].read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError
                c, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])
            except ValueError:
                issues.append({"arquivo": s, "linha": ln, "tipo": "linha_malformada"}); continue
            n_boxes += 1
            if not 0 <= c < n_classes:
                issues.append({"arquivo": s, "linha": ln, "tipo": "classe_invalida", "valor": c})
            if not all(0 <= v <= 1 for v in (x, y, bw, bh)) or x - bw / 2 < -1e-6 or x + bw / 2 > 1 + 1e-6 or y - bh / 2 < -1e-6 or y + bh / 2 > 1 + 1e-6:
                issues.append({"arquivo": s, "linha": ln, "tipo": "coordenada_fora_da_imagem"})
            if bw * w < min_px or bh * h < min_px:
                issues.append({"arquivo": s, "linha": ln, "tipo": "caixa_minuscula"})
            if (c, x, y, bw, bh) in seen:
                issues.append({"arquivo": s, "linha": ln, "tipo": "duplicata"})
            seen.add((c, x, y, bw, bh))
    por_tipo: dict[str, int] = {}
    for i in issues:
        por_tipo[i["tipo"]] = por_tipo.get(i["tipo"], 0) + 1
    return {"formato": "yolo", "imagens": len(imgs), "caixas": n_boxes, "ok": not issues, "problemas_por_tipo": por_tipo, "problemas": issues[:200]}


def validate_coco(coco_path: Path) -> dict:
    d = json.loads(Path(coco_path).read_text())
    imgs = {i["id"]: i for i in d.get("images", [])}
    cats = {c["id"] for c in d.get("categories", [])}
    issues, ids = [], set()
    for a in d.get("annotations", []):
        if a["id"] in ids:
            issues.append({"id": a["id"], "tipo": "id_duplicado"})
        ids.add(a["id"])
        if a["image_id"] not in imgs:
            issues.append({"id": a["id"], "tipo": "image_id_inexistente"}); continue
        if a["category_id"] not in cats:
            issues.append({"id": a["id"], "tipo": "categoria_inexistente"})
        x, y, w, h = a["bbox"]
        im = imgs[a["image_id"]]
        if w <= 0 or h <= 0:
            issues.append({"id": a["id"], "tipo": "caixa_degenerada"})
        elif x < -1e-6 or y < -1e-6 or x + w > im["width"] + 1e-6 or y + h > im["height"] + 1e-6:
            issues.append({"id": a["id"], "tipo": "coordenada_fora_da_imagem"})
    por_tipo: dict[str, int] = {}
    for i in issues:
        por_tipo[i["tipo"]] = por_tipo.get(i["tipo"], 0) + 1
    return {"formato": "coco", "imagens": len(imgs), "anotacoes": len(d.get("annotations", [])), "ok": not issues, "problemas_por_tipo": por_tipo, "problemas": issues[:200]}
