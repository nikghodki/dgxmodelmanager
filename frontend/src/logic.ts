// Small pure helpers used by pages; unit-tested in logic.test.ts.

/** I5: only stopped/failed deployments can be removed (DELETE /deployments/{id}). */
export const canRemove = (state: string) => state === "stopped" || state === "failed";

const isGgufRef = (ref: string) => /\.gguf$/i.test(ref) || ref.startsWith("/");

/** M8: advice can be applied as a reconfigure only if it stays on the same engine.
 *  A GGUF model_ref (llama.cpp) on a vLLM deployment is informational. */
export function adviceApplicable(a: { patch?: Record<string, any> | null; model_ref?: string | null }, d: { engine: string }): boolean {
  if (a.model_ref) return d.engine === "llamacpp" ? isGgufRef(a.model_ref) : !isGgufRef(a.model_ref);
  return !!a.patch && Object.keys(a.patch).length > 0;
}

/** M9: refs may be absolute GGUF paths; encode so "/" is not collapsed by the proxy. */
export const favouritePath = (ref: string) => `/planner/favourites/${encodeURIComponent(ref)}`;

/** M12: NaN/Infinity break chart scales; treat them as gaps. */
export function finiteSeries<T extends Record<string, any>>(data: T[], keys: string[]): T[] {
  return data.map((row) => {
    const out: Record<string, any> = { ...row };
    for (const k of keys) {
      const v = out[k];
      if (typeof v === "number" && !Number.isFinite(v)) out[k] = null;
    }
    return out as T;
  });
}
