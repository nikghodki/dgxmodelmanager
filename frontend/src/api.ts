const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
let csrf = "";
export const setCsrf = (t: string) => {
  csrf = t;
};

export class ApiError extends Error {
  constructor(public status: number, public detail: any) {
    super(typeof detail === "string" ? detail : detail?.message ?? `HTTP ${status}`);
  }
  get code(): string | undefined {
    return typeof this.detail === "object" ? this.detail?.code : undefined;
  }
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const method = opts.method ?? "GET";
  const headers: Record<string, string> = {};
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET") headers["X-CSRF-Token"] = csrf;
  const res = await fetch(`${BASE}/api${path}`, {
    method,
    headers,
    credentials: "same-origin",
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = Array.isArray(data?.detail)
      ? { code: "validation", message: data.detail.map((d: any) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ") }
      : data?.detail ?? data;
    throw new ApiError(res.status, detail);
  }
  return data as T;
}

export const streamUrl = `${BASE}/api/stream`;
