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

  it("has no engine image select while the allowlist is empty", () => {
    render(<ConfigForm engine="vllm" value={{}} onChange={vi.fn()} showAdvanced={true} images={{ default: "vllm/vllm-openai@sha256:aa", allowed: [] }} />);
    expect(screen.queryByRole("combobox", { name: "Engine image" })).toBeNull();
  });

  it("offers the default and the allowed images, and the default removes the setting", async () => {
    const onChange = vi.fn();
    const allowed = ["vllm/vllm-openai:new-model@sha256:11"];
    const { rerender } = render(<ConfigForm engine="vllm" value={{ max_model_len: 4096 }} onChange={onChange} showAdvanced={false}
      images={{ default: "vllm/vllm-openai@sha256:aa", allowed }} />);
    await userEvent.click(screen.getByRole("combobox", { name: "Engine image" }));
    expect(await screen.findByRole("option", { name: "Default (vllm/vllm-openai@sha256:aa)" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("option", { name: allowed[0] }));
    expect(onChange).toHaveBeenLastCalledWith({ max_model_len: 4096, image: allowed[0] });
    rerender(<ConfigForm engine="vllm" value={{ max_model_len: 4096, image: allowed[0] }} onChange={onChange} showAdvanced={false}
      images={{ default: "vllm/vllm-openai@sha256:aa", allowed }} />);
    await userEvent.click(screen.getByRole("combobox", { name: "Engine image" }));
    await userEvent.click(await screen.findByRole("option", { name: /^Default/ }));
    expect(onChange).toHaveBeenLastCalledWith({ max_model_len: 4096 });
  });

  it("keeps showing a stored image the allowlist no longer holds", async () => {
    render(<ConfigForm engine="vllm" value={{ image: "old/image@sha256:22" }} onChange={vi.fn()} showAdvanced={false}
      images={{ default: null, allowed: [] }} />);
    await userEvent.click(screen.getByRole("combobox", { name: "Engine image" }));
    expect(await screen.findByRole("option", { name: "old/image@sha256:22 (no longer allowed)" })).toBeInTheDocument();
  });
});
