import { useEffect, useState } from "react";
import { api } from "./api";
import { Infer } from "./Infer";
import { Evaluations, Classification, Benchmark } from "./Views";

const TABS = ["Inferência", "Avaliação e comparação", "Classificação (DeepWeeds)", "Benchmark", "Limites"] as const;
type Tab = (typeof TABS)[number];

function Limits() {
  const [d, setD] = useState<{ advisory: string; limites: string[] } | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.limits().then(setD).catch((e) => setErr(String(e))); }, []);
  if (err) return <p className="err" role="alert">{err}</p>;
  if (!d) return <p>Carregando…</p>;
  return (
    <section>
      <h2>Limites e uso responsável</h2>
      <p className="notice">{d.advisory}</p>
      <ul>{d.limites.map((l) => <li key={l}>{l}</li>)}</ul>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("Inferência");
  return (
    <>
      <header>
        <h1>WeedVision AI</h1>
        <p className="sub">Detecção de plantas daninhas — pesquisa/demo. Não recomenda pulverização nem controla atuadores.</p>
        <nav role="tablist">
          {TABS.map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "Inferência" && <Infer />}
        {tab === "Avaliação e comparação" && <Evaluations />}
        {tab === "Classificação (DeepWeeds)" && <Classification />}
        {tab === "Benchmark" && <Benchmark />}
        {tab === "Limites" && <Limits />}
      </main>
    </>
  );
}
