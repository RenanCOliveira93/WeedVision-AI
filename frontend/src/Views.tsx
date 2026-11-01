import { useEffect, useState } from "react";
import { api, type EvalRow } from "./api";
import { domainTag, num, pct, prPath } from "./lib";

function useLoad<T>(f: () => Promise<T>) {
  const [d, setD] = useState<T | null>(null); const [e, setE] = useState("");
  useEffect(() => { f().then(setD).catch((x) => setE(String(x instanceof Error ? x.message : x))); }, []); // eslint-disable-line
  return { d, e };
}

function PR({ curve }: { curve: { recall: number[]; precision: number[] } }) {
  const W = 320, H = 220;
  return (
    <svg viewBox={`-40 -10 ${W + 60} ${H + 50}`} role="img" aria-label="Curva precisão-revocação (IoU 0,5)" className="chart">
      <rect x={0} y={0} width={W} height={H} fill="none" stroke="currentColor" opacity={0.3} />
      <path d={prPath(curve.recall, curve.precision, W, H)} fill="none" stroke="#1f77b4" strokeWidth={2} />
      <text x={W / 2} y={H + 32} textAnchor="middle" fontSize={12}>revocação</text>
      <text x={-30} y={H / 2} fontSize={12} transform={`rotate(-90 -30 ${H / 2})`} textAnchor="middle">precisão</text>
      {[0, 0.5, 1].map((v) => <g key={v}><text x={v * W} y={H + 14} textAnchor="middle" fontSize={11}>{v}</text><text x={-6} y={(1 - v) * H + 4} textAnchor="end" fontSize={11}>{v}</text></g>)}
    </svg>
  );
}

function Detail({ name }: { name: string }) {
  const { d, e } = useLoad(() => api.evaluation(name));
  if (e) return <p className="err" role="alert">{e}</p>;
  if (!d) return <p>Carregando…</p>;
  const cm: number[][] = d.confusion_matrix; const labels = [...d.class_names, "fundo"];
  const fn = d.false_negatives;
  return (
    <div>
      <h3>{d.model} · {d.split} <span className="tag">{domainTag(d.dataset, d.split)}</span></h3>
      <div className="cards">
        {[["mAP50", num(d.map50)], ["mAP50-95", num(d.map50_95)], [`Precisão @${d.conf_threshold}`, pct(d.prf.weed.precision)], [`Revocação @${d.conf_threshold}`, pct(d.prf.weed.recall)], [`F1 @${d.conf_threshold}`, pct(d.prf.weed.f1)], ["Melhor F1 (grade)", `${pct(d.best_f1.f1)} @${d.best_f1.conf}`]].map(([k, v]) => <div key={k} className="card"><small>{k}</small><b>{v}</b></div>)}
      </div>
      <div className="two">
        <div><h4>Curva PR (IoU 0,5)</h4>{d.pr_curves.weed ? <PR curve={d.pr_curves.weed} /> : <p>sem dados</p>}</div>
        <div><h4>Matriz de confusão (linhas = predito, colunas = real)</h4>
          <table className="cm"><thead><tr><th></th>{labels.map((l: string) => <th key={l}>real: {l}</th>)}</tr></thead>
            <tbody>{cm.map((row, i) => <tr key={i}><th>pred: {labels[i]}</th>{row.map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody></table>
          <small>“fundo” = sem objeto: coluna = falso positivo; linha = falso negativo.</small></div>
      </div>
      {fn && <><h4>Análise de falsos negativos</h4>
        <p>{fn.fn_total} de {fn.n_gt} objetos não detectados (conf ≥ {d.conf_threshold}, IoU ≥ 0,5). Motivos: {Object.entries(fn.motivos).map(([k, v]) => `${k}: ${v}`).join(" · ")}</p>
        <table><thead><tr><th>Grupo</th><th>n</th><th>FN</th><th>taxa FN</th></tr></thead>
          <tbody>{[...Object.entries(fn.por_tamanho), ...Object.entries(fn.por_aglomeracao)].map(([k, v]: any) => <tr key={k}><td>{k}</td><td>{v.n}</td><td>{v.fn}</td><td>{pct(v.taxa_fn)}</td></tr>)}</tbody></table></>}
      {d.fp_on_crop && <p>Falsos positivos: {d.fp_on_crop.fp_total}; com centro dentro de uma caixa <em>crop</em> anotada: {d.fp_on_crop.fp_com_centro_em_crop}.</p>}
    </div>
  );
}

export function Evaluations() {
  const { d, e } = useLoad(() => api.evaluations());
  const [sel, setSel] = useState("");
  useEffect(() => { if (d?.length && !sel) setSel(d[0].arquivo); }, [d, sel]);
  if (e) return <p className="err" role="alert">{e}</p>;
  if (!d) return <p>Carregando…</p>;
  const rows: EvalRow[] = d;
  return (
    <section>
      <h2>Avaliação e comparação de modelos</h2>
      <p className="notice">Todas as linhas vêm de <code>runs/eval/*.json</code>, geradas pelo mesmo módulo de métricas para todos os modelos. Linhas “outro domínio” medem generalização zero-shot, não desempenho no domínio de treino.</p>
      <table><thead><tr><th>Modelo</th><th>Conjunto</th><th>Domínio</th><th>Imagens</th><th>mAP50</th><th>mAP50-95</th><th>P</th><th>R</th><th></th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.arquivo} className={r.arquivo === sel ? "sel" : ""}>
            <td>{r.modelo}</td><td>{r.split} ({r.dataset})</td><td>{domainTag(r.dataset, r.split)}</td><td>{r.n_images}</td><td>{num(r.map50)}</td><td>{num(r.map50_95)}</td><td>{pct(r.precision)}</td><td>{pct(r.recall)}</td>
            <td><button onClick={() => setSel(r.arquivo)}>detalhes</button></td></tr>))}</tbody></table>
      {sel && <Detail name={sel} />}
    </section>
  );
}

export function Classification() {
  const { d, e } = useLoad(() => api.classification());
  if (e) return <p className="err" role="alert">{e}</p>;
  if (!d) return <p>Carregando…</p>;
  return (
    <section>
      <h2>Baseline de classificação (DeepWeeds)</h2>
      <p className="notice">Tarefa: <b>classificar a imagem inteira</b> em 9 classes. Os rótulos são por imagem; <b>não há caixas</b> e nenhuma foi inventada. Resultado de classificação não é resultado de detecção.</p>
      <table><thead><tr><th>Split</th><th>Dias em &gt;1 split</th><th>Imagens de teste</th><th>Acurácia</th><th>F1 macro</th></tr></thead>
        <tbody>{Object.entries(d.resultados).map(([k, v]: any) => <tr key={k}><td>{v.split === "random" ? "aleatório" : "por data (sessão)"} · sorteio {v.seed}</td><td>{v.vazamento.dias_em_mais_de_um_split}</td><td>{v.test.n}</td><td>{pct(v.test.accuracy)}</td><td>{num(v.test.macro_f1)}</td></tr>)}</tbody></table>
      <p>Split aleatório coloca quadros do mesmo dia em treino e teste (vazamento de sessão); o split por data separa dias inteiros. Os conjuntos de teste <b>não são os mesmos</b> (outra mistura de espécies e dias), e cada sorteio de dias é uma amostra: veja a variação entre sorteios.</p>
    </section>
  );
}

export function Benchmark() {
  const { d, e } = useLoad(() => api.benchmark());
  if (e) return <p className="notice">{e}</p>;
  if (!d) return <p>Carregando…</p>;
  return (
    <section>
      <h2>Benchmark de inferência</h2>
      <p className="notice">{d.hardware} — <b>não é hardware de borda real</b>. {d.nota}</p>
      <table><thead><tr><th>Modelo / runtime</th><th>média (ms)</th><th>p50</th><th>p95</th><th>FPS</th><th>RSS pico (MB)</th><th>n</th></tr></thead>
        <tbody>{d.resultados.map((r: any) => <tr key={r.nome}><td>{r.nome}</td><td>{num(r.mean_ms, 1)}</td><td>{num(r.p50_ms, 1)}</td><td>{num(r.p95_ms, 1)}</td><td>{num(r.fps, 1)}</td><td>{num(r.rss_mb, 0)}</td><td>{r.n}</td></tr>)}</tbody></table>
    </section>
  );
}
