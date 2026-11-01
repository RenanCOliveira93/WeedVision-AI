from weedvision import deepweeds as D


def _rows():
    rows = []
    for d in range(30):
        for k in range(40):
            rows.append({"file": f"2017{d // 28 + 1:02d}{d % 28 + 1:02d}-{k}.jpg", "label": (d + k) % len(D.CLASSES), "date": f"2017{d // 28 + 1:02d}{d % 28 + 1:02d}"})
    return rows


def test_split_date_sem_vazamento_e_todas_as_classes():
    rows = _rows()
    sp = D.split_date(rows, fractions=(0.70, 0.15, 0.15), seed=1)
    assert D.leakage_report(rows, sp)["dias_em_mais_de_um_split"] == 0
    for s in ("train", "val", "test"):
        assert {r["label"] for r in rows if sp[r["file"]] == s} == set(range(len(D.CLASSES)))


def test_split_random_vaza_dias_e_e_deterministico():
    rows = _rows()
    a, b = D.split_random(rows, seed=3), D.split_random(rows, seed=3)
    assert a == b
    assert D.leakage_report(rows, a)["dias_em_mais_de_um_split"] > 0


def test_split_date_padrao_sem_validacao_e_com_classe_rara():
    rows = _rows()
    for r in rows:  # uma classe aparece em só 2 dias → precisa cair 1 dia no treino e 1 no teste
        if r["label"] == 2 and r["date"] not in ("20170101", "20170102"):
            r["label"] = 3
    sp = D.split_date(rows, seed=0)
    assert set(sp.values()) <= {"train", "test"} and D.leakage_report(rows, sp)["dias_em_mais_de_um_split"] == 0
    for s in ("train", "test"):
        assert 2 in {r["label"] for r in rows if sp[r["file"]] == s}
