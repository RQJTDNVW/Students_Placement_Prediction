import { MoreHorizontal, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { PredictionRecord } from '../../types/prediction.ts'
import { formatDateTime, formatProbability } from '../../utils/format.ts'
import { RiskBadge, StatusMarker } from '../common/Badge.tsx'
import { Button } from '../common/Button.tsx'
import { ProgressBar } from '../common/ProgressBar.tsx'

interface TableProps {
  records: PredictionRecord[]
  onView: (record: PredictionRecord) => void
  onDelete: (record: PredictionRecord) => void
}

export function PredictionTable({ records, onView, onDelete }: TableProps) {
  return <div className="overflow-x-auto"><table className="w-full min-w-[930px] border-collapse text-left"><thead className="sticky top-0 z-10 bg-raised"><tr className="border-b border-line text-[10px] uppercase tracking-[.12em] text-meta"><th className="px-4 py-3 font-semibold">Prediction ID</th><th className="px-4 py-3 font-semibold">CGPA</th><th className="px-4 py-3 font-semibold">Internships</th><th className="px-4 py-3 font-semibold">Projects</th><th className="px-4 py-3 font-semibold">Placement probability</th><th className="px-4 py-3 font-semibold">Status</th><th className="px-4 py-3 font-semibold">Risk</th><th className="px-4 py-3 font-semibold">Date</th><th className="px-4 py-3" /></tr></thead><tbody>{records.map((record) => <PredictionRow key={record.prediction_id} record={record} onView={onView} onDelete={onDelete} />)}</tbody></table></div>
}

function PredictionRow({ record, onView, onDelete }: { record: PredictionRecord; onView: (record: PredictionRecord) => void; onDelete: (record: PredictionRecord) => void }) {
  const [open, setOpen] = useState(false)
  return <tr className="border-b border-line/70 last:border-0 hover:bg-raised/40"><td className="mono px-4 py-4 text-xs text-cyan">{record.prediction_id}</td><td className="mono px-4 py-4 text-xs">{record.cgpa.toFixed(1)}</td><td className="mono px-4 py-4 text-xs">{record.internships}</td><td className="mono px-4 py-4 text-xs">{record.projects_count}</td><td className="w-48 px-4 py-4"><div className="flex items-center gap-3"><span className="mono w-9 text-xs">{formatProbability(record.probability)}</span><div className="flex-1"><ProgressBar value={record.probability * 100} showValue={false} /></div></div></td><td className="px-4 py-4"><StatusMarker status={record.status} /></td><td className="px-4 py-4"><RiskBadge risk={record.risk_level} /></td><td className="mono whitespace-nowrap px-4 py-4 text-[10px] text-meta">{formatDateTime(record.timestamp)}</td><td className="relative px-4 py-4"><button type="button" aria-label={`Actions for ${record.prediction_id}`} onClick={() => setOpen((value) => !value)} className="rounded-sm p-1 text-meta hover:bg-recessed hover:text-cyan"><MoreHorizontal aria-hidden="true" className="h-4 w-4" /></button>{open && <div className="absolute right-3 top-11 z-20 w-36 border border-line bg-raised p-1 shadow-lg"><button type="button" onClick={() => { setOpen(false); onView(record) }} className="block w-full px-3 py-2 text-left text-xs text-muted hover:bg-recessed hover:text-ink">View details</button><button type="button" onClick={() => { setOpen(false); onDelete(record) }} className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-negative hover:bg-negative/10"><Trash2 aria-hidden="true" className="h-3.5 w-3.5" />Delete record</button></div>}</td></tr>
}

export function HistoryMobileList({ records, onView, onDelete }: TableProps) {
  return <div className="divide-y divide-line border-y border-line lg:hidden">{records.map((record) => <article key={record.prediction_id} className="py-4"><div className="flex items-start justify-between gap-3"><div><p className="mono text-xs text-cyan">{record.prediction_id}</p><p className="mono mt-1 text-[10px] text-meta">{formatDateTime(record.timestamp)}</p></div><p className="mono text-2xl text-ink">{formatProbability(record.probability)}</p></div><div className="mt-4 grid grid-cols-3 gap-3 border-y border-line py-3"><div><p className="eyebrow text-[9px]">CGPA</p><p className="mono mt-1 text-xs">{record.cgpa.toFixed(1)}</p></div><div><p className="eyebrow text-[9px]">Internships</p><p className="mono mt-1 text-xs">{record.internships}</p></div><div><p className="eyebrow text-[9px]">Projects</p><p className="mono mt-1 text-xs">{record.projects_count}</p></div></div><div className="mt-3 flex flex-wrap items-center justify-between gap-3"><div className="flex gap-4"><StatusMarker status={record.status} /><RiskBadge risk={record.risk_level} /></div><div className="flex gap-2"><Button variant="text" size="sm" onClick={() => onView(record)}>View details</Button><Button variant="danger" size="sm" onClick={() => onDelete(record)}><Trash2 aria-hidden="true" className="h-3.5 w-3.5" />Delete</Button></div></div></article>)}</div>
}
