import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { InsightBody, parseInsightBody } from "./insight-body";

const TUNING = [
  "Raising the Health triage threshold from 0.80 to 0.90 is supported by 129 judged model answers.",
  "",
  "threshold       | answers | precision",
  "--------------- | ------- | ---------",
  "0.80 (current)  | 120     | 83.3 %",
  "0.90 (proposed) | 9       | 100.0 %",
  "",
  "Apply sets the threshold; nothing changes until then.",
].join("\n");

const ROUTING = [
  "Suggested LiteLLM aliases per traffic class:",
  "",
  "- dgxmm-chat (chat): tiny-chat on deployment d1 (rules)",
  "- dgxmm-batch (batch): big on deployment d2 (rules)",
  "",
  "Copy the snippet into LiteLLM's config.",
].join("\n");

describe("InsightBody", () => {
  it("renders a templated evidence table as a table", () => {
    render(<InsightBody body={TUNING} />);
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["threshold", "answers", "precision"]);
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(3);
    expect(within(rows[2]).getAllByRole("cell").map((c) => c.textContent)).toEqual(["0.90 (proposed)", "9", "100.0 %"]);
    expect(screen.getByText(/Raising the Health triage threshold/)).toBeInTheDocument();
    expect(screen.getByText(/Apply sets the threshold/)).toBeInTheDocument();
  });

  it("renders bullet lines as a list, separate from the paragraphs", () => {
    render(<InsightBody body={ROUTING} />);
    const items = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(items.map((i) => i.textContent)).toEqual(["dgxmm-chat (chat): tiny-chat on deployment d1 (rules)", "dgxmm-batch (batch): big on deployment d2 (rules)"]);
    expect(screen.getByText("Copy the snippet into LiteLLM's config.").tagName).toBe("P");
  });

  it("keeps single line breaks inside a paragraph", () => {
    render(<InsightBody body={"line one\nline two"} />);
    const p = screen.getByText(/line one/);
    expect(p.textContent).toBe("line one\nline two");
    expect(p).toHaveClass("whitespace-pre-line");
  });

  it("never interprets HTML in the body", () => {
    const { container } = render(<InsightBody body={'<img src=x onerror="alert(1)"> <b>bold</b>\n\n- <script>x</script>'} />);
    expect(container.querySelector("img, b, script")).toBeNull();
    expect(screen.getByText(/<img src=x/)).toBeInTheDocument();
  });

  it("a pipe in ordinary text is not a table", () => {
    expect(parseInsightBody("a | b\nnot a rule")).toEqual([{ kind: "p", text: "a | b\nnot a rule" }]);
  });
});
