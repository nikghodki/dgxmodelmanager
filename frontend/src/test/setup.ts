import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// jsdom lacks these browser APIs used by Radix, cmdk and our media-query hook.
if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false, media: query, onchange: null,
      addEventListener: vi.fn(), removeEventListener: vi.fn(), addListener: vi.fn(), removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}
class RO { observe() {} unobserve() {} disconnect() {} }
(globalThis as any).ResizeObserver ??= RO;
Element.prototype.scrollIntoView ??= vi.fn();
Element.prototype.hasPointerCapture ??= vi.fn(() => false) as any;
Element.prototype.releasePointerCapture ??= vi.fn();
