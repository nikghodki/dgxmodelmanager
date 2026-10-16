import { describe, expect, it } from "vitest";
import { GIB, fmtBytes, fmtPct, fmtRelative, stateTone } from "./format";

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

describe("fmtRelative", () => {
  const now = Date.parse("2026-10-16T12:00:00Z");
  it("formats recent and older times", () => {
    expect(fmtRelative("2026-10-16T11:59:30Z", now)).toBe("just now");
    expect(fmtRelative("2026-10-16T11:55:00Z", now)).toBe("5 min ago");
    expect(fmtRelative("2026-10-16T09:00:00Z", now)).toBe("3 h ago");
    expect(fmtRelative("2026-10-13T12:00:00Z", now)).toBe("3 d ago");
  });
});
