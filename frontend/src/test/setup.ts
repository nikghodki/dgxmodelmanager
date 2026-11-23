import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup, configure } from "@testing-library/react";

afterEach(() => cleanup());
// findBy*/waitFor give up after 1 s by default: on a loaded CI runner a debounced preview (350 ms) plus its render can
// take longer, which made the deploy wizard's tests flaky there. Passing tests are not slowed down.
configure({ asyncUtilTimeout: 5000 });

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
