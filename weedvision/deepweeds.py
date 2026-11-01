"""DeepWeeds (Olsen et al. 2019): baseline de CLASSIFICAÇÃO de imagem inteira (9 classes). NÃO há caixas nem máscaras.

Os rótulos são por imagem; este módulo nunca os converte em bounding boxes.
Splits: `random` (estratificado, vaza quadros quase idênticos do mesmo dia) vs `date` (dias inteiros → treino/val/teste).
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative"]


def read_labels(csv_path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(csv_path)):
        rows.append({"file": r["Filename"], "label": CLASSES.index(r["Species"]), "date": r["Filename"][:8]})
    return rows


def split_random(rows, fractions=(0.70, 0.15, 0.15), seed=0) -> dict[str, str]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r["file"])
    out = {}
    for files in by.values():
        rng.shuffle(files)
        a, b = int(len(files) * fractions[0]), int(len(files) * (fractions[0] + fractions[1]))
        for k, f in enumerate(files):
            out[f] = "train" if k < a else "val" if k < b else "test"
    return out


def split_date(rows, fractions=(0.70, 0.0, 0.30), seed=0, tries=5000) -> dict[str, str]:
    """Dias inteiros por split. Sorteia atribuições e fica com a que (i) tem todas as classes em todo split com fração > 0 e
    (ii) mais se aproxima das frações-alvo de imagens. Determinístico dado `seed`.
    Padrão: SEM validação (0%) — Parkinsonia foi fotografada em só 2 dias (Parthenium e Siam weed em 3), então não há
    como ter as 9 classes em treino, validação e teste ao mesmo tempo com separação por dia."""
    days = Counter(r["date"] for r in rows)
    per_day = defaultdict(Counter)
    for r in rows:
        per_day[r["date"]][r["label"]] += 1
    names, total = ("train", "val", "test"), len(rows)
    rng, best, best_err = random.Random(seed), None, 9.0
    for _ in range(tries):
        order = list(days)
        rng.shuffle(order)
        assign = {d: names[rng.choices(range(3), weights=[f if f > 0 else 0 for f in fractions])[0]] for d in order}
        cnt = {n: Counter() for n in names}
        for d, n in assign.items():
            cnt[n].update(per_day[d])
        if any(f > 0 and len(cnt[n]) < len(CLASSES) for n, f in zip(names, fractions)):
            continue
        err = sum(abs(sum(cnt[n].values()) / total - f) for n, f in zip(names, fractions))
        if err < best_err:
            best, best_err = assign, err
    if best is None:
        raise RuntimeError("nenhuma atribuição por data cobre todas as classes nos splits pedidos")
    return {r["file"]: best[r["date"]] for r in rows}


def leakage_report(rows, split: dict[str, str]) -> dict:
    """Dias que aparecem em mais de um split (0 = sem vazamento por sessão)."""
    d2s = defaultdict(set)
    for r in rows:
        d2s[r["date"]].add(split[r["file"]])
    return {"dias_total": len(d2s), "dias_em_mais_de_um_split": sum(len(s) > 1 for s in d2s.values()),
            "imagens": dict(Counter(split.values()))}
