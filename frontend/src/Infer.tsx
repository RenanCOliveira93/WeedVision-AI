import { useCallback, useEffect, useRef, useState } from "react";
import { api, type InferResult, type ModelInfo } from "./api";
import { csvEscape, scaleBox } from "./lib";

export function Infer() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.6);
  const [res, setRes] = useState<InferResult | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [geo, setGeo] = useState({ fx: "", fy: "", cx: "", cy: "", lat: "", lon: "", height_m: "", yaw_deg: "", pitch_deg: "" });
  const [geoOut, setGeoOut] = useState("");

  useEffect(() => { api.models().then((m) => { const ok = m.filter((x) => x.onnx); setModels(ok); if (ok[0]) setModel(ok[0].id); }).catch((e) => setErr(String(e))); }, []);

  const draw = useCallback((bmp: ImageBitmap | HTMLVideoElement, r: InferResult) => {
    const c = canvas.current; if (!c) return;
    const w = r.image.width, h = r.image.height;
    c.width = Math.min(w, 960); c.height = Math.round((c.width * h) / w);
    const g = c.getContext("2d")!; g.drawImage(bmp, 0, 0, c.width, c.height);
    g.lineWidth = 2; g.font = "14px sans-serif";
    for (const d of r.detections) {
      const [x1, y1, x2, y2] = scaleBox(d.xyxy, w, h, c.width, c.height);
      g.strokeStyle = "#d62728"; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      g.fillStyle = "#d62728"; g.fillText(`${d.label} ${d.score.toFixed(2)}`, x1, Math.max(12, y1 - 3));
    }
  }, []);

  const run = useCallback(async (blob: Blob, src: ImageBitmap | HTMLVideoElement) => {
    setBusy(true); setErr("");
    try { const r = await api.infer(blob, model, conf, iou); setRes(r); draw(src, r); setGeoOut(""); }
    catch (e) { setErr(String(e instanceof Error ? e.message : e)); setLive(false); }
    finally { setBusy(false); }
  }, [model, conf, iou, draw]);

  const onFile = async (f: File | undefined) => { if (f) await run(f, await createImageBitmap(f)); };

  const stopCam = () => { stream.current?.getTracks().forEach((t) => t.stop()); stream.current = null; setLive(false); };
  const startCam = async () => {
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      if (video.current) { video.current.srcObject = stream.current; await video.current.play(); }
      setLive(true);
    } catch (e) { setErr(`Webcam indisponível: ${e instanceof Error ? e.message : e}`); }
  };
  useEffect(() => () => stopCam(), []);

  useEffect(() => {
    if (!live) return;
    let stop = false;
    (async () => {
      while (!stop && video.current) {
        const v = video.current; const t = document.createElement("canvas"); t.width = v.videoWidth; t.height = v.videoHeight;
        if (t.width) {
          t.getContext("2d")!.drawImage(v, 0, 0);
          const blob: Blob | null = await new Promise((ok) => t.toBlob(ok, "image/jpeg", 0.85));
          if (blob && !stop) await run(blob, v);
        }
        await new Promise((r) => setTimeout(r, 400));   // um quadro por vez, sem fila
      }
    })();
    return () => { stop = true; };
  }, [live, run]);

  const dl = (name: string, text: string, type: string) => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); };
  const exportCsv = async () => { if (res) dl("deteccoes_pixels.csv", await api.csv(res.detections), "text/csv"); };
  const exportGeo = async () => {
    if (!res) return;
    try {
      const n = (k: keyof typeof geo) => Number(geo[k]);
      const camera = geo.fx ? { fx: n("fx"), fy: n("fy"), cx: n("cx"), cy: n("cy"), width: res.image.width, height: res.image.height } : null;
      const pose = geo.lat ? { lat: n("lat"), lon: n("lon"), height_m: n("height_m"), yaw_deg: n("yaw_deg"), pitch_deg: n("pitch_deg") } : null;
      const gj = await api.geojson({ detections: res.detections, camera, pose, source: "upload" });
      setGeoOut(`OK: ${gj.features.length} pontos`); dl("deteccoes.geojson", JSON.stringify(gj, null, 2), "application/geo+json");
    } catch (e) { setGeoOut(String(e instanceof Error ? e.message : e)); }
  };
  const sel = models.find((m) => m.id === model);

  return (
    <section>
      <h2>Inferência</h2>
      <div className="row">
        <label>Modelo <select value={model} onChange={(e) => setModel(e.target.value)} aria-label="Modelo">{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
        <label>Confiança mín. {conf.toFixed(2)} <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(+e.target.value)} aria-label="Confiança" /></label>
        <label>IoU NMS {iou.toFixed(2)} <input type="range" min={0.2} max={0.9} step={0.05} value={iou} onChange={(e) => setIou(+e.target.value)} aria-label="IoU NMS" /></label>
      </div>
      {sel && <p className="notice">{sel.domain_note} Licença: {sel.license}.</p>}
      <div className="row">
        <label className="btn">Enviar imagem <input type="file" accept="image/*" hidden onChange={(e) => onFile(e.target.files?.[0])} aria-label="Enviar imagem" /></label>
        {!live ? <button onClick={startCam}>Ligar webcam</button> : <button onClick={stopCam}>Desligar webcam</button>}
        {busy && <span>processando…</span>}
      </div>
      <video ref={video} muted playsInline style={{ display: "none" }} />
      {err && <p className="err" role="alert">{err}</p>}
      <canvas ref={canvas} className="view" aria-label="Resultado da detecção" />
      {res && (
        <div>
          <p><strong data-testid="count">{res.count}</strong> detecções · preprocessamento {res.timings_ms.pre_ms} ms · inferência {res.timings_ms.infer_ms} ms · pós {res.timings_ms.post_ms} ms</p>
          <p className="notice">{res.advisory}</p>
          <table><thead><tr><th>#</th><th>classe</th><th>confiança</th><th>caixa (px)</th></tr></thead>
            <tbody>{res.detections.slice(0, 50).map((d, i) => <tr key={i}><td>{i + 1}</td><td>{d.label}</td><td>{d.score.toFixed(2)}</td><td>{d.xyxy.map((v) => Math.round(v)).join(", ")}</td></tr>)}</tbody></table>
          <div className="row"><button onClick={exportCsv}>Exportar CSV (pixels)</button></div>
          <details><summary>Exportar GeoJSON (exige calibração e pose)</summary>
            <p>Sem intrínsecos da câmera <em>e</em> pose, a API recusa (422): não há coordenada a partir de pixels sem isso. Solo plano assumido.</p>
            <div className="grid">{(Object.keys(geo) as (keyof typeof geo)[]).map((k) => <label key={k}>{k}<input value={geo[k]} onChange={(e) => setGeo({ ...geo, [k]: e.target.value })} aria-label={k} /></label>)}</div>
            <button onClick={exportGeo}>Gerar GeoJSON</button> {geoOut && <p role="status" className={geoOut.startsWith("OK") ? "" : "err"}>{geoOut}</p>}
          </details>
          <small>CSV: {csvEscape("pixels, sem georreferenciamento")}</small>
        </div>
      )}
    </section>
  );
}
