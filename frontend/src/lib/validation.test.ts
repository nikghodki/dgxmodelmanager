import { describe, expect, it } from "vitest";
import { passwordProblems } from "@/lib/validation";

describe("passwordProblems", () => {
  it("requires 12+ characters, a match and a change", () => {
    expect(passwordProblems("short", "short")).toEqual(["Use at least 12 characters"]);
    expect(passwordProblems("long-enough-pw", "long-enough-px")).toEqual(["Passwords do not match"]);
    expect(passwordProblems("same-old-password", "same-old-password", "same-old-password")).toEqual(["Choose a password different from the current one"]);
    expect(passwordProblems("a-good-new-password", "a-good-new-password", "old-password-123")).toEqual([]);
  });
  it("does not complain about a match before the second field is typed", () => {
    expect(passwordProblems("a-good-new-password", "")).toEqual([]);
  });
});
