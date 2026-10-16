import { describe, expect, it } from "vitest";

// Responsive grids must declare a base column template (`grid-cols-1` or similar); otherwise the implicit
// auto track is sized to min-content and long content overflows narrow phones.
const sources = import.meta.glob(["../pages/*.tsx", "../components/app/*.tsx"], { query: "?raw", import: "default", eager: true }) as Record<string, string>;

describe("responsive grids", () => {
  it("every grid with a breakpoint column template also has a base column template", () => {
    const bad: string[] = [];
    for (const [file, src] of Object.entries(sources)) {
      for (const m of src.matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\})/g)) {
        const cls = (m[1] ?? m[2]).split(/\s+/);
        if (!cls.includes("grid")) continue;
        const responsive = cls.some((c) => /^(sm|md|lg|xl):grid-cols-/.test(c));
        const base = cls.some((c) => /^grid-cols-/.test(c));
        if (responsive && !base) bad.push(`${file}: ${m[1] ?? m[2]}`);
      }
    }
    expect(bad).toEqual([]);
  });
});
