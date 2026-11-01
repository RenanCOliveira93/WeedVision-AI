"""Preparação do dataset Sudars (Ronin_OPEN_DB, CC BY 4.0): VOC XML → YOLO + COCO, com split por SESSÃO.

Sessão é uma SUPOSIÇÃO: o dataset não traz metadados de sessão/fazenda. Usamos corridas contíguas de IDs de
arquivo (salto de ID > `gap` abre nova sessão) e o grupo `IMG_####` (câmera/origem distinta, tamanhos mistos)
como origem retida. Isso reduz, mas não elimina, vazamento entre treino e teste.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["weed", "crop"]  # weed=0, crop=1


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_voc(xml: Path) -> dict:
    r = ET.parse(xml).getroot()
    w, h = int(r.findtext("size/width")), int(r.findtext("size/height"))
    boxes = []
    for o in r.findall("object"):
        name = o.findtext("name").strip().lower()
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, x1, y1, x2, y2))
    return {"filename": r.findtext("filename"), "w": w, "h": h, "boxes": boxes}


def clean_box(x1, y1, x2, y2, w, h):
    """Recorta à imagem; devolve None se a caixa ficar degenerada (<2 px de lado)."""
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def assign_sessions(stems: list[str], gap: int = 100) -> dict[str, str]:
    """stem → id de sessão. Numéricos: corridas com salto ≤ gap. `IMG_*`: um grupo próprio (origem retida)."""
    num = sorted((int(s), s) for s in stems if re.fullmatch(r"\d+", s))
    out, sid, prev = {}, 0, None
    for n, s in num:
        if prev is not None and n - prev > gap:
            sid += 1
        out[s] = f"S{sid:03d}"
        prev = n
    for s in stems:
        if s not in out:
            out[s] = "ORIGIN_IMG"
    return out


def split_sessions(sess: dict[str, str], fractions=(0.70, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """Atribui SESSÕES INTEIRAS a train/val/test (guloso por contagem de imagens). ORIGIN_IMG → 'test_origin'."""
    counts: dict[str, int] = {}
    for s, g in sess.items():
        counts[g] = counts.get(g, 0) + 1
    groups = [g for g in counts if g != "ORIGIN_IMG"]
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -counts[g])  # maiores primeiro, depois preenche o déficit
    total = sum(counts[g] for g in groups)
    target = dict(zip(("train", "val", "test"), (f * total for f in fractions)))
    cur = dict.fromkeys(target, 0)
    gsplit = {}
    for g in groups:
        name = max(target, key=lambda k: target[k] - cur[k])
        gsplit[g] = name
        cur[name] += counts[g]
    gsplit["ORIGIN_IMG"] = "test_origin"
    return {s: gsplit[g] for s, g in sess.items()}


def build(raw: Path, out: Path, gap: int = 100, seed: int = 0, link: bool = True) -> dict:
    imgdir = raw / "raw images"
    xmls = sorted((raw / "annotations").glob("*.xml"))
    items, dropped = [], {"caixas_degeneradas": 0, "sem_imagem": 0, "classe_desconhecida": 0}
    for x in xmls:
        a = parse_voc(x)
        stem = x.stem
        img = next((p for p in (imgdir / f"{stem}.jpg", imgdir / f"{stem}.JPG") if p.exists()), None)
        if img is None:
            dropped["sem_imagem"] += 1
            continue
        boxes = []
        for name, *b in a["boxes"]:
            if name not in CLASSES:
                dropped["classe_desconhecida"] += 1
                continue
            cb = clean_box(*b, a["w"], a["h"])
            if cb is None:
                dropped["caixas_degeneradas"] += 1
                continue
            boxes.append((CLASSES.index(name), *cb))
        items.append({"stem": stem, "img": img, "w": a["w"], "h": a["h"], "boxes": boxes})
    sess = assign_sessions([i["stem"] for i in items], gap)
    split = split_sessions(sess, seed=seed)
    if out.exists():
        shutil.rmtree(out)
    manifest, coco = [], {}
    for sp in ("train", "val", "test", "test_origin"):
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)
        coco[sp] = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    aid = 1
    for k, it in enumerate(items):
        sp = split[it["stem"]]
        dst = out / "images" / sp / it["img"].name.lower()
        (shutil.copy if not link else lambda s, d: d.symlink_to(s.resolve()))(it["img"], dst)
        lines = [f"{c} {(x1 + x2) / 2 / it['w']:.6f} {(y1 + y2) / 2 / it['h']:.6f} {(x2 - x1) / it['w']:.6f} {(y2 - y1) / it['h']:.6f}"
                 for c, x1, y1, x2, y2 in it["boxes"]]
        (out / "labels" / sp / f"{it['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        coco[sp]["images"].append({"id": k, "file_name": dst.name, "width": it["w"], "height": it["h"]})
        for c, x1, y1, x2, y2 in it["boxes"]:
            coco[sp]["annotations"].append({"id": aid, "image_id": k, "category_id": c, "bbox": [x1, y1, x2 - x1, y2 - y1],
                                            "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}); aid += 1
        manifest.append({"stem": it["stem"], "split": sp, "session": sess[it["stem"]], "w": it["w"], "h": it["h"],
                         "n_weed": sum(b[0] == 0 for b in it["boxes"]), "n_crop": sum(b[0] == 1 for b in it["boxes"]),
                         "sha256": sha256(it["img"])})
    for sp, c in coco.items():
        (out / f"coco_{sp}.json").write_text(json.dumps(c))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: weed\n  1: crop\n")
    stats = {}
    for sp in ("train", "val", "test", "test_origin"):
        rows = [m for m in manifest if m["split"] == sp]
        stats[sp] = {"imagens": len(rows), "sessoes": len({m["session"] for m in rows}),
                     "weed": sum(m["n_weed"] for m in rows), "crop": sum(m["n_crop"] for m in rows)}
    info = {"classes": CLASSES, "gap": gap, "seed": seed, "descartes": dropped, "stats": stats,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps({"info": info, "items": manifest}, indent=1))
    return info


if __name__ == "__main__":
    import sys
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    print(json.dumps(build(base / "raw/sudars/ex/Ronin_OPEN_DB", base / "processed/sudars_yolo"), indent=1, ensure_ascii=False))


def weed_only(src: Path, dst: Path) -> dict:
    """Deriva o dataset de 1 classe (weed) do dataset de 2 classes. As caixas `crop` ficam em `crop_boxes.json`
    (pixels xyxy por imagem) só para a análise de falsos positivos sobre a cultura; não são alvo de treino.
    Motivo: 403 das 411 caixas `crop` estão no grupo de origem retida; no treino há 3 → classe não aprendível."""
    if dst.exists():
        shutil.rmtree(dst)
    crop, stats = {}, {}
    for sp in ("train", "val", "test", "test_origin"):
        (dst / "images" / sp).mkdir(parents=True)
        (dst / "labels" / sp).mkdir(parents=True)
        for im in (src / "images" / sp).iterdir():
            (dst / "images" / sp / im.name).symlink_to(im.resolve())
        coco = json.loads((src / f"coco_{sp}.json").read_text())
        ids = {i["id"]: i for i in coco["images"]}
        keep = [a for a in coco["annotations"] if a["category_id"] == 0]
        for a in coco["annotations"]:
            if a["category_id"] == 1:
                x, y, w, h = a["bbox"]
                crop.setdefault(sp, {}).setdefault(ids[a["image_id"]]["file_name"], []).append([x, y, x + w, y + h])
        (dst / f"coco_{sp}.json").write_text(json.dumps({**coco, "annotations": keep, "categories": [{"id": 0, "name": "weed"}]}))
        n = 0
        for lb in (src / "labels" / sp).glob("*.txt"):
            ls = [l for l in lb.read_text().splitlines() if l.startswith("0 ")]
            (dst / "labels" / sp / lb.name).write_text("\n".join(ls) + ("\n" if ls else ""))
            n += len(ls)
        stats[sp] = {"imagens": len(coco["images"]), "weed": n}
    (dst / "crop_boxes.json").write_text(json.dumps(crop))
    (dst / "data.yaml").write_text(f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: weed\n")
    return stats


def build_maize(raw: Path, dst: Path) -> dict:
    """Maize Nigéria (HF parquet; bbox [x,y,w,h]; 0=maize, 1=weed) → YOLO 1 classe (weed) como split `test`.
    Atenção: aqui `weed` anota REGIÕES de mato, protocolo diferente do Sudars (plantas individuais)."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images/test").mkdir(parents=True)
    (dst / "labels/test").mkdir(parents=True)
    crop, n_img, n_w, n_m = {}, 0, 0, 0
    for f in sorted(raw.glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=25):
            for r in batch.to_pylist():
                name = f"{f.stem[-14:-9]}_{n_img:04d}.jpg"
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                W, H = im.size
                im.save(dst / "images/test" / name, quality=95)
                lines = []
                for (x, y, w, h), k in zip(r["objects"]["bbox"], r["objects"]["categories"]):
                    if k == 1:
                        lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                        n_w += 1
                    else:
                        crop.setdefault("test", {}).setdefault(name, []).append([x, y, x + w, y + h])
                        n_m += 1
                (dst / "labels/test" / name.replace(".jpg", ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
    (dst / "crop_boxes.json").write_text(json.dumps(crop))
    return {"imagens": n_img, "weed": n_w, "maize": n_m}
