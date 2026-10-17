import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { Catalog } from "./Catalog";
import * as auth from "@/auth";
import * as apiModule from "@/api";

vi.mock("@/auth");
vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return { ...actual, api: vi.fn() };
});

const long = "unsloth/Qwen3.8-27B-Instruct-2607-GGUF-with-a-very-long-repository-name";

describe("Catalog on a phone", () => {
  it("keeps a long model name inside the card (no horizontal overflow at 390 px)", async () => {
    const mm = window.matchMedia;
    window.matchMedia = ((q: string) => ({ matches: q.includes("max-width: 767px"), media: q, onchange: null,
      addEventListener: vi.fn(), removeEventListener: vi.fn(), addListener: vi.fn(), removeListener: vi.fn(),
      dispatchEvent: vi.fn() })) as any;
    try {
      vi.mocked(auth.useIsAdmin).mockReturnValue(false);
      vi.mocked(apiModule.api).mockImplementation((url: string) => url.startsWith("/catalog/search")
        ? Promise.resolve({ results: [{ repo_id: long, downloads: 1, est_bytes: 1e9, quant: null, gated: false,
          downloaded: false, fit: { badge: "fits_now" }, support: { vllm: true, llamacpp: null }, llmfit: null }],
          free_for_new: 5e9, banner: null })
        : new Promise(() => undefined));
      const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      render(<QueryClientProvider client={qc}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Catalog /></MemoryRouter></QueryClientProvider>);
      const btn = await screen.findByRole("button", { name: long });
      expect(btn.closest("li")).not.toBeNull(); // the mobile card layout
      // jsdom has no layout: the name may be at most as wide as its card (truncated), the 44ch cap only from md up
      expect(btn).toHaveClass("max-w-full", "truncate");
      expect(btn.className).not.toMatch(/(^|\s)max-w-\[44ch\]/);
      expect(btn.className).toMatch(/(^|\s)md:max-w-\[44ch\]/);
    } finally {
      window.matchMedia = mm;
    }
  });
});
