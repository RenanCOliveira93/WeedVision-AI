import type { Det } from "./lib";

const B = "/api";
async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText;
    try { const b = await r.json(); msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* corpo não-JSON */ }
    throw new Error(`${r.status}: ${msg}`);
  }
  return r.json() as Promise<T>;
}
export interface ModelInfo { id: string; name: string; format: string; license: string; onnx: string | null; domain_note: string; trained_on: string; tem_metricas: boolean }
export interface InferResult { count: number; detections: Det[]; image: { width: number; height: number }; timings_ms: Record<string, number>; advisory: string; domain_note?: string; annotated_jpeg_base64?: string }
export interface EvalRow { arquivo: string; modelo: string; split: string; dataset: string; n_images: number; map50: number; map50_95: number; precision: number | null; recall: number | null; f1: number | null; conf: number; fp_on_crop?: { fp_total: number; fp_com_centro_em_crop: number } }

export const api = {
  models: () => fetch(`${B}/models`).then(j<ModelInfo[]>),
  evaluations: () => fetch(`${B}/evaluations`).then(j<EvalRow[]>),
  evaluation: (n: string) => fetch(`${B}/evaluations/${encodeURIComponent(n.replace(/\.json$/, ""))}`).then(j<any>),
  benchmark: () => fetch(`${B}/benchmark`).then(j<any>),
  classification: () => fetch(`${B}/classification`).then(j<any>),
  limits: () => fetch(`${B}/limits`).then(j<{ advisory: string; limites: string[] }>),
  infer: (file: Blob, model: string, conf: number, iou: number) => {
    const f = new FormData(); f.append("file", file, "frame.jpg"); f.append("model_id", model); f.append("conf", String(conf)); f.append("iou", String(iou));
    return fetch(`${B}/infer/image`, { method: "POST", body: f }).then(j<InferResult>);
  },
  csv: (detections: Det[]) => fetch(`${B}/export/csv`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ detections }) }).then((r) => r.text()),
  geojson: (body: unknown) => fetch(`${B}/export/geojson`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then(j<any>),
};
