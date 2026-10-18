import { describe, expect, it } from "vitest";
import { aliasSnippet, sourceText } from "@/lib/routing";

describe("routing advice helpers", () => {
  it("describes the source", () => {
    expect(sourceText({ source: "model", confidence: 0.912 })).toBe("Decision model · 91 %");
    expect(sourceText({ source: "rules", confidence: null })).toBe("Rules");
    expect(sourceText({ source: "none", confidence: null })).toBe("—");
  });
  it("builds a per-class alias snippet onto the existing model name", () => {
    expect(aliasSnippet({ alias: "dgxmm-code", model_name: "qwen-coder", routable: true })).toBe('# merge into your existing router_settings\nrouter_settings:\n  model_group_alias:\n    dgxmm-code: "qwen-coder"');
    expect(aliasSnippet({ alias: "dgxmm-vision", model_name: null, routable: false })).toBeNull();
    expect(aliasSnippet({ alias: "dgxmm-chat", model_name: "sleepy", routable: false })).toBeNull();
  });
});
