import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfigForm } from "@/components/app/config-form";

describe("ConfigForm", () => {
  it("edits ints, removes cleared values, toggles switches, hides advanced", async () => {
    const onChange = vi.fn();
    render(<ConfigForm engine="vllm" value={{ max_model_len: 4096, enable_prefix_caching: true }} onChange={onChange} showAdvanced={false} />);
    expect(screen.queryByLabelText("Served model name")).toBeNull();
    const ctx = screen.getByLabelText("Context length (tokens)");
    await userEvent.clear(ctx);
    expect(onChange).toHaveBeenLastCalledWith({ enable_prefix_caching: true });
    await userEvent.click(screen.getByRole("switch", { name: "Prefix caching" }));
    expect(onChange).toHaveBeenLastCalledWith({ max_model_len: 4096, enable_prefix_caching: false });
  });
});
