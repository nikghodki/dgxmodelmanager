import { describe, expect, it } from "vitest";
import { ALL_DAYS, WEEKDAYS, actionLabel, daysLabel, isHHMM, nextRunPreview, pairOf, parseProposal, proposalRows, scheduleKey, toggleDay, validTimeZone } from "./schedules";

describe("schedules", () => {
  it("labels days like the server", () => {
    expect(daysLabel(WEEKDAYS)).toBe("Mon–Fri");
    expect(daysLabel(ALL_DAYS)).toBe("every day");
    expect(daysLabel(0b1100000)).toBe("Sat, Sun");
    expect(daysLabel(0b1010101)).toBe("Mon, Wed, Fri, Sun");
    expect(daysLabel(0)).toBe("no day");
  });
  it("toggles days and validates times", () => {
    expect(toggleDay(WEEKDAYS, 5)).toBe(0b0111111);
    expect(toggleDay(WEEKDAYS, 0)).toBe(0b0011110);
    expect(isHHMM("07:45") && isHHMM("23:59") && isHHMM("00:00")).toBe(true);
    expect(isHHMM("24:00") || isHHMM("7:45") || isHHMM("07:60") || isHHMM("")).toBe(false);
  });
  it("pairs and labels actions", () => {
    expect(pairOf("sleep")).toBe("wake");
    expect(pairOf("stop")).toBe("start");
    expect(actionLabel("sleep")).toBe("Sleep");
  });
  it("parses a proposal and refuses malformed ones", () => {
    const p = { sleep_action: "sleep", sleep_at: "19:00", sleep_days: 31, wake_action: "wake", wake_at: "07:45", wake_days: 31, timezone: "Europe/Berlin", weeks: 3 };
    expect(parseProposal(p)?.sleep_at).toBe("19:00");
    expect(proposalRows(parseProposal(p)!)).toEqual([
      { action: "sleep", days: 31, at_local: "19:00", timezone: "Europe/Berlin" },
      { action: "wake", days: 31, at_local: "07:45", timezone: "Europe/Berlin" },
    ]);
    expect(parseProposal({ ...p, sleep_action: "remove" })).toBeNull();
    expect(parseProposal({ ...p, sleep_days: 0 })).toBeNull();
    expect(parseProposal({ ...p, wake_at: "25:00" })).toBeNull();
    expect(parseProposal({ ...p, wake_action: "start" })).toBeNull(); // sleep pairs with wake
    expect(parseProposal(null)).toBeNull();
  });
  it("query key under the deployment (deployment SSE events refresh it)", () => {
    expect(scheduleKey(3)).toEqual(["deployment", 3, "schedules"]);
  });
});

describe("next run preview", () => {
  const now = Date.UTC(2026, 9, 19, 10, 0); // Monday 10:00 UTC
  it("finds the next matching day in the zone", () => {
    expect(nextRunPreview(ALL_DAYS, "10:30", "UTC", now)?.toISOString()).toBe("2026-10-19T10:30:00.000Z");
    expect(nextRunPreview(ALL_DAYS, "09:00", "UTC", now)?.toISOString()).toBe("2026-10-20T09:00:00.000Z");
    expect(nextRunPreview(0b1000000, "09:00", "UTC", now)?.toISOString()).toBe("2026-10-25T09:00:00.000Z"); // Sunday
    expect(nextRunPreview(ALL_DAYS, "19:00", "Europe/Berlin", now)?.toISOString()).toBe("2026-10-19T17:00:00.000Z"); // CEST
  });
  it("handles DST: a gap runs after it, an overlap at the first occurrence", () => {
    const sat = Date.UTC(2026, 2, 28, 12, 0);
    expect(nextRunPreview(ALL_DAYS, "02:30", "Europe/Berlin", sat)?.toISOString()).toBe("2026-03-29T01:30:00.000Z");
    const satOct = Date.UTC(2026, 9, 24, 12, 0);
    expect(nextRunPreview(ALL_DAYS, "02:30", "Europe/Berlin", satOct)?.toISOString()).toBe("2026-10-25T00:30:00.000Z");
  });
  it("refuses bad input", () => {
    expect(nextRunPreview(0, "10:00", "UTC", now)).toBeNull();
    expect(nextRunPreview(ALL_DAYS, "10:00", "Mars/Olympus", now)).toBeNull();
    expect(validTimeZone("Europe/Berlin") && !validTimeZone("Mars/Olympus") && !validTimeZone("../x")).toBe(true);
  });
});
