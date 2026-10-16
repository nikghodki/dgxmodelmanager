import type { Estimate } from "@/types";

export interface EstimatePart { key: string; label: string; bytes: number }

const ORDER: [keyof Estimate, string][] = [
  ["weights", "Weights"], ["draft_weights", "Draft weights"], ["kv_cache", "KV cache"], ["state", "Recurrent state"],
  ["activations", "Activations"], ["runtime", "Runtime (CUDA, graphs)"], ["margin", "Margin / calibration"],
];

export function estimateParts(e: Partial<Estimate>): EstimatePart[] {
  return ORDER.map(([key, label]) => ({ key, label, bytes: Number(e[key] ?? 0) })).filter((p) => p.bytes !== 0);
}
