import type { ReactNode } from 'react'
import { Card } from '../common/Card.tsx'

export function ModelMetricCard({ label, value, detail, icon }: { label: string; value: string; detail: string; icon?: ReactNode }) { return <Card variant="solid" className="p-4"><div className="flex items-start justify-between gap-3"><p className="eyebrow text-[9px]">{label}</p><span className="text-cyan">{icon}</span></div><p className="mono mt-4 text-2xl font-semibold text-ink">{value}</p><p className="mt-2 text-xs text-meta">{detail}</p></Card> }
