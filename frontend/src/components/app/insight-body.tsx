import { cn } from "@/lib/utils";

/** A templated insight body as structured text: blank-line separated blocks; a block of "- " lines is a list, a block
 *  of "a | b" rows whose second row is a "--- | ---" rule is a table, anything else a paragraph that keeps its line
 *  breaks. Everything is rendered as text nodes (no HTML is ever interpreted). */
type Block = { kind: "p"; text: string } | { kind: "ul"; items: string[] } | { kind: "table"; head: string[]; rows: string[][] };

const RULE = /^[\s|:-]+$/;
const cells = (line: string) => line.split("|").map((c) => c.trim());

export function parseInsightBody(body: string): Block[] {
  return body.replace(/\r\n?/g, "\n").split(/\n\s*\n/).map((b) => b.replace(/^\n+|\n+$/g, "")).filter((b) => b.trim())
    .map((b): Block => {
      const lines = b.split("\n");
      if (lines.every((l) => /^\s*- /.test(l))) return { kind: "ul", items: lines.map((l) => l.replace(/^\s*- /, "")) };
      if (lines.length >= 2 && lines.every((l) => l.includes("|")) && RULE.test(lines[1]) && lines[1].includes("-")) {
        const head = cells(lines[0]);
        return { kind: "table", head, rows: lines.slice(2).map((l) => cells(l)) };
      }
      return { kind: "p", text: b };
    });
}

export function InsightBody({ body, className }: { body: string; className?: string }) {
  return (
    <div data-insight-body className={cn("space-y-2 break-words text-sm text-muted-foreground", className)}>
      {parseInsightBody(body).map((b, i) => (
        b.kind === "ul" ? <ul key={i} className="list-disc space-y-0.5 pl-5">{b.items.map((t, j) => <li key={j}>{t}</li>)}</ul>
        : b.kind === "table" ? (
          <div key={i} className="overflow-x-auto">
            <table className="text-xs tabular-nums">
              <thead><tr>{b.head.map((h, j) => <th key={j} scope="col" className="px-2 py-1 text-left font-medium">{h}</th>)}</tr></thead>
              <tbody>{b.rows.map((r, j) => <tr key={j} className="border-t">{r.map((c, k) => <td key={k} className="whitespace-nowrap px-2 py-1">{c}</td>)}</tr>)}</tbody>
            </table>
          </div>
        )
        : <p key={i} className="whitespace-pre-line">{b.text}</p>
      ))}
    </div>
  );
}
