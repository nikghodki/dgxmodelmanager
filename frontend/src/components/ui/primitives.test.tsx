import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

describe("shadcn primitives", () => {
  it("renders button variants with token classes", () => {
    render(<Button variant="destructive">Stop</Button>);
    expect(screen.getByRole("button", { name: "Stop" }).className).toContain("bg-destructive");
  });
  it("renders a badge", () => {
    render(<Badge variant="outline">nvfp4</Badge>);
    expect(screen.getByText("nvfp4")).toBeInTheDocument();
  });
  it("opens a dialog from its trigger", async () => {
    render(
      <Dialog>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent><DialogTitle>Hello</DialogTitle><DialogDescription>World</DialogDescription></DialogContent>
      </Dialog>,
    );
    await userEvent.click(screen.getByText("Open"));
    expect(await screen.findByRole("dialog")).toHaveTextContent("Hello");
  });
});
