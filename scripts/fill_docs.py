"""Preenche tabelas dos docs a partir de runs/*.json, entre marcadores <!--T:NOME--> ... <!--/T:NOME-->. Reexecutável; nenhum número é digitado à mão."""
import json, re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
J = lambda p: json.loads((root / p).read_text())  # noqa: E731
f3 = lambda v: "—" if v is None or v != v else f"{v:.3f}"
pc = lambda v: "—" if v is None or v != v else f"{100 * v:.1f}%"


def t_eval():
    rows = ["| Modelo | Conjunto | Domínio | Imgs | mAP50 | mAP50-95 | P@0,25 | R@0,25 | F1@0,25 |", "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted((root / "runs/eval").glob("*.json")):
        r = json.loads(p.read_text()); w = r["prf"]["weed"]
        dom = "outro (zero-shot)" if r["dataset"].startswith("maize") else ("origem retida" if r["split"] == "test_origin" else "mesmo")
        rows.append(f"| {r['model']} | {r['split']} | {dom} | {r['n_images']} | {f3(r['map50'])} | {f3(r['map50_95'])} | {pc(w['precision'])} | {pc(w['recall'])} | {pc(w['f1'])} |")
    return "\n".join(rows)


def t_boot():
    b = J("runs/bootstrap.json"); rows = ["| Conjunto | Modelo | mAP50 (IC 95%) | mAP50-95 (IC 95%) | Δ mAP50 vs yolo11n (IC 95%) |", "|---|---|---|---|---|"]
    for s, d in b.items():
        for m, r in d.items():
            if not (isinstance(r, dict) and "map50" in r): continue
            diff = next((v for k, v in r.items() if k.startswith("diferenca")), None)
            dtxt = "—" if diff is None else f"{diff['map50']['media']:+.3f} [{diff['map50']['ic95'][0]:+.3f}, {diff['map50']['ic95'][1]:+.3f}]"
            rows.append(f"| {s} (n={d['n_images']}) | {m} | {r['map50']['media']:.3f} [{r['map50']['ic95'][0]:.3f}, {r['map50']['ic95'][1]:.3f}] | [{r['map50_95']['ic95'][0]:.3f}, {r['map50_95']['ic95'][1]:.3f}] | {dtxt} |")
    return "\n".join(rows)


def t_cls():
    rows = ["| Split | Sorteio | Dias em >1 split | Imagens de teste | Acurácia | F1 macro | Menor recall por classe |", "|---|---|---|---|---|---|---|"]
    for p in sorted((root / "runs/deepweeds").glob("cls_*.json")):
        r = json.loads(p.read_text()); t = r["test"]; lo = min(t["recall_por_classe"].items(), key=lambda kv: kv[1])
        rows.append(f"| {'aleatório' if r['split'] == 'random' else 'por data'} | {r.get('seed', 0)} | {r['leakage']['dias_em_mais_de_um_split']}/{r['leakage']['dias_total']} | {t['n']} | {pc(t['accuracy'])} | {f3(t['macro_f1'])} | {lo[0]}: {pc(lo[1])} |")
    return "\n".join(rows)


def t_bench():
    b = J("runs/benchmark.json"); rows = ["| Configuração | média (ms) | p50 | p95 | FPS | RSS pico (MB) |", "|---|---|---|---|---|---|"]
    for r in b["resultados"]:
        rows.append(f"| {r['nome']} | {r['mean_ms']:.1f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps']:.1f} | {r['rss_mb']:.0f} |")
    return "\n".join(rows) + f"\n\n_Hardware: {b['hardware']}._ {b['nota']}" + (f"\n\n**Falhas:** {b['falhas']}" if b["falhas"] else "")


def t_drift():
    d = J("runs/drift.json"); feats = list(d["piso_de_ruido_metades_aleatorias_do_treino"])
    rows = ["| Comparação (ref. = treino) | " + " | ".join(feats) + " |", "|---|" + "---|" * len(feats)]
    fmt = lambda v: "∞" if v == float("inf") else f"{v:.2f}"  # noqa: E731
    rows.append("| piso de ruído (metades do treino) | " + " | ".join(fmt(d["piso_de_ruido_metades_aleatorias_do_treino"][f]["psi"]) for f in feats) + " |")
    for k, c in d["comparacoes"].items():
        rows.append(f"| {k} (n={d['n_por_conjunto'][k]}) | " + " | ".join(fmt(c[f]["psi"]) for f in feats) + " |")
    return "\n".join(rows)


def t_fn():
    rows = ["| Modelo / conjunto | FN / GT | muito pequeno | pequeno | médio | grande |", "|---|---|---|---|---|---|"]
    for m in ("yolo11n", "yolo11s", "frcnn_mnv3_v2"):
        for s in ("test", "test_origin"):
            p = root / f"runs/eval/{m}__{s}.json"
            if not p.exists(): continue
            fn = json.loads(p.read_text())["false_negatives"]
            cells = [f"{v['fn']}/{v['n']} ({pc(v['taxa_fn'])})" for v in fn["por_tamanho"].values()]
            rows.append(f"| {m} / {s} | {fn['fn_total']}/{fn['n_gt']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


GEN = {"EVAL": t_eval, "BOOT": t_boot, "CLS": t_cls, "BENCH": t_bench, "DRIFT": t_drift, "FN": t_fn}
for doc in list((root / "docs/pt-BR").glob("*.md")) + [root / "docs/MODEL_CARD.md", root / "README.md", root / "README.en.md"]:
    if not doc.exists(): continue
    txt = doc.read_text(); new = txt
    for k, fn in GEN.items():
        if f"<!--T:{k}-->" in new:
            try: body = fn()
            except FileNotFoundError as e: body = f"_(indisponível: {e.filename})_"
            new = re.sub(rf"<!--T:{k}-->.*?<!--/T:{k}-->", lambda _: f"<!--T:{k}-->\n{body}\n<!--/T:{k}-->", new, flags=re.S)
    if new != txt: doc.write_text(new); print("atualizado", doc.name)
