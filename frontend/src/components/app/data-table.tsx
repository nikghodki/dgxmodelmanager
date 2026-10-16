import { Fragment, useState, type ReactNode } from "react";
import {
  flexRender, getCoreRowModel, getFilteredRowModel, getSortedRowModel, useReactTable,
  type ColumnDef, type SortingState,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronDown, ChevronRight, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useIsMobile } from "@/hooks/use-media-query";
import { cn } from "@/lib/utils";

export interface ColumnMeta { label?: string; className?: string; hideOnMobile?: boolean; mono?: boolean }

export interface DataTableProps<T> {
  columns: ColumnDef<T, any>[];
  data: T[];
  getRowId?: (r: T) => string;
  searchPlaceholder?: string;
  searchText?: (r: T) => string;
  toolbar?: ReactNode;
  rowActions?: (r: T) => ReactNode;
  renderDetail?: (r: T) => ReactNode | null;
  mobileTitle?: (r: T) => ReactNode;
  onRowClick?: (r: T) => void;
  loading?: boolean;
  empty?: ReactNode;
  initialSort?: { id: string; desc?: boolean };
}

const metaOf = (c: { columnDef: ColumnDef<any, any> }) => (c.columnDef.meta ?? {}) as ColumnMeta;
const headerText = (c: { id: string; columnDef: ColumnDef<any, any> }) =>
  metaOf(c).label ?? (typeof c.columnDef.header === "string" ? c.columnDef.header : c.id);

export function DataTable<T>({
  columns, data, getRowId, searchPlaceholder, searchText, toolbar, rowActions, renderDetail, mobileTitle, onRowClick,
  loading, empty, initialSort,
}: DataTableProps<T>) {
  const mobile = useIsMobile();
  const [sorting, setSorting] = useState<SortingState>(initialSort ? [initialSort as { id: string; desc: boolean }] : []);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const table = useReactTable({
    data, columns, getRowId: getRowId as any,
    state: { sorting, globalFilter: query },
    onSortingChange: setSorting,
    onGlobalFilterChange: setQuery,
    globalFilterFn: (row, _id, value: string) => {
      const text = searchText ? searchText(row.original) : JSON.stringify(row.original);
      return text.toLowerCase().includes(String(value).toLowerCase());
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });
  const rows = table.getRowModel().rows;

  const bar = (searchPlaceholder || toolbar) && (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      {searchPlaceholder && (
        <div className="relative w-full sm:w-72">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={searchPlaceholder} className="pl-8" aria-label={searchPlaceholder} />
        </div>
      )}
      {toolbar}
    </div>
  );

  if (loading) {
    return <div className="space-y-2">{bar}{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full" />)}</div>;
  }
  if (!rows.length) return <div>{bar}{empty ?? <p className="py-8 text-center text-sm text-muted-foreground">No results.</p>}</div>;

  if (mobile) {
    return (
      <div>
        {bar}
        <ul className="space-y-3">
          {rows.map((row) => {
            const cells = row.getVisibleCells();
            const detail = renderDetail?.(row.original);
            return (
              <li key={row.id}>
                <Card className="shadow-card p-4" onClick={onRowClick ? () => onRowClick(row.original) : undefined}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1 break-words font-medium">
                      {mobileTitle ? mobileTitle(row.original) : flexRender(cells[0].column.columnDef.cell, cells[0].getContext())}
                    </div>
                    {rowActions && <div onClick={(e) => e.stopPropagation()}>{rowActions(row.original)}</div>}
                  </div>
                  <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-sm">
                    {cells.slice(mobileTitle ? 0 : 1).filter((c) => !metaOf(c.column).hideOnMobile).map((c) => (
                      <Fragment key={c.id}>
                        <dt className="text-muted-foreground">{headerText(c.column)}</dt>
                        <dd className={cn("min-w-0 break-all", metaOf(c.column).mono && "font-mono text-xs")}>
                          {flexRender(c.column.columnDef.cell, c.getContext())}
                        </dd>
                      </Fragment>
                    ))}
                  </dl>
                  {detail && <div className="mt-3 border-t pt-3 text-sm">{detail}</div>}
                </Card>
              </li>
            );
          })}
        </ul>
      </div>
    );
  }

  return (
    <div>
      {bar}
      <div className="overflow-x-auto rounded-lg border bg-card shadow-card">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id} className="hover:bg-transparent">
                {renderDetail && <TableHead className="w-8" />}
                {hg.headers.map((h) => {
                  const canSort = h.column.getCanSort();
                  const dir = h.column.getIsSorted();
                  const label = flexRender(h.column.columnDef.header, h.getContext());
                  return (
                    <TableHead key={h.id} className={cn("whitespace-nowrap text-xs uppercase tracking-wider", metaOf(h.column).className)}>
                      {canSort ? (
                        <button type="button" className="inline-flex items-center gap-1 hover:text-foreground" onClick={h.column.getToggleSortingHandler()}>
                          {label}
                          {dir === "asc" ? <ArrowUp className="h-3 w-3" /> : dir === "desc" ? <ArrowDown className="h-3 w-3" /> : <ArrowUpDown className="h-3 w-3 opacity-40" />}
                        </button>
                      ) : label}
                    </TableHead>
                  );
                })}
                {rowActions && <TableHead className="w-12"><span className="sr-only">Actions</span></TableHead>}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {rows.map((row) => {
              const detail = renderDetail?.(row.original);
              const isOpen = !!open[row.id];
              return (
                <Fragment key={row.id}>
                  <TableRow className={cn(onRowClick && "cursor-pointer")} onClick={onRowClick ? () => onRowClick(row.original) : undefined}>
                    {renderDetail && (
                      <TableCell className="w-8 p-0 pl-2">
                        {detail && (
                          <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={isOpen ? "Collapse" : "Expand"} aria-expanded={isOpen}
                            onClick={(e) => { e.stopPropagation(); setOpen({ ...open, [row.id]: !isOpen }); }}>
                            {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                          </Button>
                        )}
                      </TableCell>
                    )}
                    {row.getVisibleCells().map((c) => (
                      <TableCell key={c.id} className={cn("tabular-nums", metaOf(c.column).mono && "font-mono text-xs", metaOf(c.column).className)}>
                        {flexRender(c.column.columnDef.cell, c.getContext())}
                      </TableCell>
                    ))}
                    {rowActions && <TableCell className="w-12 text-right" onClick={(e) => e.stopPropagation()}>{rowActions(row.original)}</TableCell>}
                  </TableRow>
                  {detail && isOpen && (
                    <TableRow className="bg-muted/40 hover:bg-muted/40">
                      <TableCell colSpan={row.getVisibleCells().length + (rowActions ? 2 : 1)} className="text-sm">{detail}</TableCell>
                    </TableRow>
                  )}
                </Fragment>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
