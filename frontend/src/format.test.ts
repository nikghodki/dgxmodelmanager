import { describe, expect, it } from "vitest";
import { GIB, fmtBytes, fmtPct, stateTone } from "./format";

describe("format", () => {
  it("formats bytes in binary units", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(1536)).toBe("1.5 KiB");
    expect(fmtBytes(8 * GIB)).toBe("8.0 GiB");
    expect(fmtBytes(-2 * GIB)).toBe("-2.0 GiB");
    expect(fmtBytes(null)).toBe("—");
  });
  it("formats ratios and states", () => {
    expect(fmtPct(0.256)).toBe("26%");
    expect(fmtPct(undefined)).toBe("—");
    expect(stateTone("ready")).toBe("green");
    expect(stateTone("failed")).toBe("red");
    expect(stateTone("loading")).toBe("amber");
  });
});
