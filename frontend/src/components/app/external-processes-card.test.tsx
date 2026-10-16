import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ExternalProcessesCard } from "@/components/app/external-processes-card";

const GIB = 1024 ** 3;

describe("ExternalProcessesCard", () => {
  it("shows container (or host), the redacted command with a full-command title and the RAM/GPU split", () => {
    const long = "python /home/n/build_submission_v2.py --model Qwen/Qwen3-8B --hf-token=*** " + "x".repeat(200);
    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ExternalProcessesCard items={[
          { pid: 1, name: "python", rss: 48 * GIB, gpu: 27 * GIB, command: long, user: "nik", container: null },
          { pid: 2, name: "python3", rss: GIB, gpu: 0, command: "litellm --port 4000", user: "uid 1000", user_scope: "container",
            container: { id: "bbbbbbbbbbbb", name: "litellm-proxy", image: "ghcr.io/berriai/litellm:main-latest" } },
          { pid: 3, name: "oldserver", rss: GIB, gpu: 0 },
          { pid: 4, name: "python3", rss: GIB, gpu: 0, command: "vllm serve old", user: "root", user_scope: "container",
            container: { id: "dddddddddddd", name: "dgxmm-old", image: "vllm" }, dgxmm_label: { deployment_id: 9, name: "old-qwen" } },
        ]} />
      </MemoryRouter>,
    );
    expect(screen.getByText("host")).toBeInTheDocument();
    expect(screen.getByText("litellm-proxy")).toBeInTheDocument();
    expect(screen.getByTitle(long)).toBeInTheDocument();
    expect(screen.getByText(/RAM 48\.0 GiB/)).toBeInTheDocument();
    expect(screen.getByText(/GPU 27\.0 GiB/)).toBeInTheDocument();
    expect(screen.getByText(/uid 1000 \(in container\)/)).toHaveAttribute("title", "uid as seen inside the container; not a host account");
    expect(screen.getByText("DGXMM-labelled")).toHaveAttribute("title", expect.stringMatching(/old-qwen/));
    expect(screen.getByText("oldserver")).toBeInTheDocument(); // older servers: falls back to the process name
    expect(screen.getByRole("link", { name: /view all processes/i })).toHaveAttribute("href", "/processes");
  });
});
