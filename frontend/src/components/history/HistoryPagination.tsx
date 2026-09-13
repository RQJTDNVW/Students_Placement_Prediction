import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '../common/Button.tsx'

export function HistoryPagination({ page, totalPages, totalCount, pageSize, onPageChange }: { page: number; totalPages: number; totalCount: number; pageSize: number; onPageChange: (page: number) => void }) {
  const start = totalCount ? (page - 1) * pageSize + 1 : 0
  const end = Math.min(page * pageSize, totalCount)
  return <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4"><span className="mono text-[10px] text-meta">{start}–{end} of {totalCount} predictions</span><div className="flex items-center gap-2"><Button variant="secondary" size="sm" aria-label="Previous page" disabled={page <= 1} onClick={() => onPageChange(page - 1)}><ChevronLeft aria-hidden="true" className="h-3.5 w-3.5" />Previous</Button><span className="mono px-2 text-xs text-ink">{page} / {totalPages}</span><Button variant="secondary" size="sm" aria-label="Next page" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>Next<ChevronRight aria-hidden="true" className="h-3.5 w-3.5" /></Button></div></div>
}
