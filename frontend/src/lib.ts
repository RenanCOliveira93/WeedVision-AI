export interface Det { xyxy: [number, number, number, number]; score: number; label: string }

export const pct = (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));

/** Converte caixa em pixels da imagem original para pixels do canvas exibido. */
export function scaleBox(b: Det["xyxy"], imgW: number, imgH: number, canvasW: number, canvasH: number): [number, number, number, number] {
  const sx = canvasW / imgW, sy = canvasH / imgH;
  return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy];
}

/** Rótulo de aviso quando o conjunto de avaliação não é o domínio de treino. */
export function domainTag(dataset: string, split = ""): "mesmo domínio" | "outro domínio (zero-shot)" | "origem retida" {
  if (dataset.startsWith("maize")) return "outro domínio (zero-shot)";
  if (split === "test_origin") return "origem retida";
  return "mesmo domínio";
}

export function csvEscape(s: string) { return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }

/** Caminho SVG da curva PR (x = revocação, y = precisão), área w×h. */
export function prPath(recall: number[], precision: number[], w: number, h: number): string {
  return recall.map((r, i) => `${i ? "L" : "M"}${(r * w).toFixed(1)},${((1 - precision[i]) * h).toFixed(1)}`).join(" ");
}
