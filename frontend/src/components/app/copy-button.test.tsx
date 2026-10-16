import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CopyButton } from "./copy-button";

const original = Object.getOwnPropertyDescriptor(navigator, "clipboard");
afterEach(() => {
  if (original) Object.defineProperty(navigator, "clipboard", original);
  else delete (navigator as any).clipboard;
});

describe("CopyButton", () => {
  it("falls back to execCommand when the Clipboard API is unavailable (plain HTTP)", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    let copied = "";
    const exec = vi.fn(() => { copied = document.querySelector("textarea")!.value; return true; });
    (document as any).execCommand = exec;
    render(<CopyButton value="http://10.0.0.5:8000/v1" label="Copy endpoint" />);
    await user.click(screen.getByRole("button", { name: "Copy endpoint" }));
    await waitFor(() => expect(exec).toHaveBeenCalledWith("copy"));
    expect(copied).toBe("http://10.0.0.5:8000/v1");
    expect(document.querySelector("textarea")).toBeNull();
  });
});
