import { BarChart3, CirclePercent, Database, ShieldAlert } from 'lucide-react'
import { Card } from '../common/Card.tsx'

export function AnalyticsCards({ total = 126, average = 71, placed = 68, highRisk = 29 }: { total?: number; average?: number; placed?: number; highRisk?: number }) {
  const cards = [{ label: 'Saved estimates', value: total.toLocaleString(), detail: 'Records in the current history', icon: Database, tone: 'text-cyan' }, { label: 'Average probability', value: `${average}%`, detail: 'Across saved estimates', icon: CirclePercent, tone: 'text-blue' }, { label: 'Placed estimate rate', value: `${placed}%`, detail: 'Predicted placed / total', icon: BarChart3, tone: 'text-positive' }, { label: 'High-risk records', value: highRisk.toLocaleString(), detail: 'Requires focused review', icon: ShieldAlert, tone: 'text-negative' }] as const
  return <div className="mb-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{cards.map(({ label, value, detail, icon: Icon, tone }) => <Card key={label} className="p-4"><div className="flex items-start justify-between"><p className="eyebrow text-[9px]">{label}</p><Icon aria-hidden="true" className={`h-4 w-4 ${tone}`} /></div><p className="mono mt-4 text-2xl font-semibold text-ink">{value}</p><p className="mt-2 text-xs text-meta">{detail}</p></Card>)}</div>
}
