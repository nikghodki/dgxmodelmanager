import { describe, expect, it } from "vitest";
import { ApiError } from "@/api";
import {
  ALIAS_RE, apiErrorText, keysScopedTo, nameFixHref, rewrittenCount, suggestAlias, takenHolder, aliasProblem, buildKeyCreateBody, buildKeyPatchBody, connectionPatch, curlSnippet, emptyKeyForm, endpointPreview,
  isAllModels, keyFormFromKey, keyFormProblems, litellmProblem, modelNameOptions, pythonSnippet, urlProblem,
} from "./litellm";

describe("endpointPreview", () => {
  it("prefers the public URL and appends /v1 without doubling slashes", () => {
    expect(endpointPreview("http://litellm:4000", "https://llm.example.com/")).toBe("https://llm.example.com/v1");
    expect(endpointPreview("http://litellm:4000/", "")).toBe("http://litellm:4000/v1");
    expect(endpointPreview("", "  ")).toBe("");
  });
});

describe("urlProblem", () => {
  it("accepts http(s) URLs with a host and rejects everything else", () => {
    expect(urlProblem("http://litellm-proxy:4000")).toBeNull();
    expect(urlProblem("https://llm.example.com")).toBeNull();
    expect(urlProblem("ftp://x")).toMatch(/http/);
    expect(urlProblem("litellm:4000")).toMatch(/http/);
    expect(urlProblem("http://u:p@host")).toMatch(/credentials/);
    expect(urlProblem("http://host/?a=1")).toMatch(/query/);
    expect(urlProblem("http://ho st")).not.toBeNull();
  });
  it("treats empty as fine (means default / cleared)", () => expect(urlProblem("")).toBeNull());
});

describe("connectionPatch", () => {
  const saved = { enabled: true, url: "http://a:4000", public_url: "", master_key_set: true };
  it("sends only changed fields", () => {
    expect(connectionPatch(saved, { enabled: true, url: "http://a:4000", public_url: "https://x", keyMode: "keep", masterKey: "" }))
      .toEqual({ body: { public_url: "https://x" }, keyDetaches: false });
  });
  it("flags that the current key stops applying when the internal URL changes, without blocking the save", () => {
    const r = connectionPatch(saved, { enabled: true, url: "http://evil:4000", public_url: "", keyMode: "keep", masterKey: "" });
    expect(r).toEqual({ body: { url: "http://evil:4000" }, keyDetaches: true });
    const ok = connectionPatch(saved, { enabled: true, url: "http://b:4000", public_url: "", keyMode: "replace", masterKey: "sk-new" });
    expect(ok).toEqual({ body: { url: "http://b:4000", master_key: "sk-new" }, keyDetaches: false });
  });
  it("no key, no detach; clear sends an empty key", () => {
    expect(connectionPatch({ ...saved, master_key_set: false }, { enabled: true, url: "http://b:4000", public_url: "", keyMode: "keep", masterKey: "" }).keyDetaches).toBe(false);
    expect(connectionPatch(saved, { enabled: false, url: "http://a:4000", public_url: "", keyMode: "clear", masterKey: "" }).body)
      .toEqual({ enabled: false, master_key: "" });
  });
  it("ignores trailing slashes when comparing URLs", () => {
    expect(connectionPatch(saved, { enabled: true, url: "http://a:4000/", public_url: "", keyMode: "keep", masterKey: "" }))
      .toEqual({ body: {}, keyDetaches: false });
  });
});

describe("aliasProblem", () => {
  it("follows the backend alias pattern", () => {
    expect(ALIAS_RE.test("default-chat")).toBe(true);
    expect(aliasProblem("org/model:v1.2_x")).toBeNull();
    expect(aliasProblem("")).toBeNull();
    expect(aliasProblem("has space")).toMatch(/letters/);
    expect(aliasProblem("x".repeat(65))).toMatch(/64/);
  });
});

describe("snippets", () => {
  it("curl and python target the single /v1 endpoint with the model name", () => {
    const c = curlSnippet("https://llm.example.com/v1", "chat");
    expect(c).toContain("https://llm.example.com/v1/chat/completions");
    expect(c).toContain('"model": "chat"');
    expect(c).toContain("$LITELLM_API_KEY");
    const p = pythonSnippet("https://llm.example.com/v1", "chat");
    expect(p).toContain('base_url="https://llm.example.com/v1"');
    expect(p).toContain('model="chat"');
    expect(curlSnippet("http://x/v1", "chat", "sk-abc")).toContain("Bearer sk-abc");
  });
});

describe("key form", () => {
  it("all models is [] or all-proxy-models", () => {
    expect(isAllModels([])).toBe(true);
    expect(isAllModels(["all-proxy-models"])).toBe(true);
    expect(isAllModels(["chat"])).toBe(false);
  });
  it("validates alias, models, numbers and durations", () => {
    const f = { ...emptyKeyForm(), alias: "team-a", allModels: false, models: [] };
    expect(keyFormProblems(f, "create").models).toMatch(/model/);
    const g = { ...emptyKeyForm(), alias: "", rpm: "1.5", maxBudget: "-1", budgetDuration: "30 days", duration: "x" };
    const p = keyFormProblems(g, "create");
    expect(Object.keys(p).sort()).toEqual(["alias", "budgetDuration", "duration", "maxBudget", "rpm"]);
    expect(keyFormProblems({ ...emptyKeyForm(), alias: "ok", maxBudget: "10", budgetDuration: "30d", rpm: "5", tpm: "100", duration: "90d" }, "create")).toEqual({});
  });
  it("builds a create body with only the given fields", () => {
    expect(buildKeyCreateBody({ ...emptyKeyForm(), alias: " team-a ", allModels: false, models: ["chat"], maxBudget: "5", budgetDuration: "30d", rpm: "10", duration: "30d" }))
      .toEqual({ key_alias: "team-a", models: ["chat"], max_budget: 5, budget_duration: "30d", rpm_limit: 10, duration: "30d" });
    expect(buildKeyCreateBody({ ...emptyKeyForm(), alias: "all", models: ["chat"], allModels: true })).toEqual({ key_alias: "all", models: [] });
  });
  it("edit sends only changes and never tries to clear a limit", () => {
    const key = { token: "t", key_name: "sk-...ab", key_alias: "a", models: ["chat"], spend: 0, max_budget: 5, budget_duration: "30d",
      rpm_limit: 10, tpm_limit: null, expires: null, created_at: "", blocked: false, metadata: {}, created_by_dgxmm: true } as any;
    const f = keyFormFromKey(key);
    expect(buildKeyPatchBody(key, f)).toEqual({});
    expect(buildKeyPatchBody(key, { ...f, rpm: "", tpm: "50", models: ["chat", "code"] })).toEqual({ tpm_limit: 50, models: ["chat", "code"] });
    expect(buildKeyPatchBody(key, { ...f, allModels: true })).toEqual({ models: [] });
    expect(buildKeyPatchBody(key, { ...f, blocked: true })).toEqual({ blocked: true });
  });
});

describe("modelNameOptions", () => {
  it("collects unique model names from deployments, plus extras, sorted", () => {
    const deps = [{ model_name: "chat", alias: "chat", served_name: "q" }, { model_name: null, alias: null, served_name: "coder" }, { model_name: "chat" }] as any;
    expect(modelNameOptions(deps, ["zeta", "chat"])).toEqual(["chat", "coder", "zeta"]);
  });
});

describe("litellmProblem", () => {
  it("explains the common integration failures", () => {
    expect(litellmProblem({ code: "litellm_no_db", message: "m" }).description).toMatch(/store_model_in_db|DATABASE_URL/i);
    expect(litellmProblem({ code: "litellm_not_configured", message: "m" }).title).toMatch(/master key/i);
    expect(litellmProblem({ code: "litellm_unreachable", message: "could not reach" }).title).toMatch(/reach/i);
    expect(litellmProblem({ code: "weird", message: "boom" }).description).toBe("boom");
  });
});

describe("model name conflicts", () => {
  const taken = new ApiError(409, { code: "model_name_taken", message: "the model name chat is already used", holder: { id: 7, name: "other", state: "ready" } });
  it("finds the holder and always names it in error text", () => {
    expect(takenHolder(taken)).toEqual({ id: 7, name: "other", state: "ready" });
    expect(takenHolder(new Error("x"))).toBeNull();
    expect(apiErrorText(taken)).toBe("the model name chat is already used (held by deployment other)");
    const named = new ApiError(409, { code: "model_name_taken", message: "used by deployment other (ready)", holder: { id: 7, name: "other", state: "ready" } });
    expect(apiErrorText(named)).toBe("used by deployment other (ready)");
    expect(apiErrorText(new Error("boom"))).toBe("boom");
  });
  it("explains a sync conflict and shows the key hint for a missing key", () => {
    expect(litellmProblem({ code: "model_name_conflict", message: "model name chat is already registered for deployment a" }).description).toMatch(/registered for deployment a/);
    expect(litellmProblem({ code: "litellm_not_configured", message: "re-enter it for this URL" }).description).toBe("re-enter it for this URL");
  });
  it("keysScopedTo lists only keys naming the model explicitly", () => {
    const k = (models: string[]) => ({ token: models.join(), models } as any);
    expect(keysScopedTo([k(["chat"]), k([]), k(["all-proxy-models"]), k(["chat", "x"]), k(["x"])], "chat").map((x) => x.token)).toEqual(["chat", "chat,x"]);
  });
});

describe("model name fixes", () => {
  it("suggests a stable, valid alias from served name + deployment name", () => {
    expect(suggestAlias("qwen3-8b", "coder")).toBe("qwen3-8b-coder");
    expect(suggestAlias("Qwen/Qwen3 8B", "a b")).toBe("Qwen/Qwen3-8B-a-b");
    expect(suggestAlias(null, "coder")).toBe("coder");
    expect(suggestAlias("x".repeat(60), "deployment")).toHaveLength(64);
    expect(ALIAS_RE.test(suggestAlias("x".repeat(60), "deployment"))).toBe(true);
  });
  it("links to the deployment's LiteLLM tab with the suggestion prefilled", () => {
    expect(nameFixHref({ id: 3, name: "coder", served_name: "qwen" })).toBe("/deployments/3?tab=litellm&alias=qwen-coder");
  });
  it("counts rewritten keys whether the backend sends a bool, a number or a list", () => {
    expect(rewrittenCount(true, 2)).toBe(2);
    expect(rewrittenCount(false, 2)).toBe(0);
    expect(rewrittenCount(1, 2)).toBe(1);
    expect(rewrittenCount([{ token: "a" }], 2)).toBe(1);
    expect(rewrittenCount(undefined, 2)).toBe(0);
  });
});
