import type { ReactNode } from 'react'
import type { RiskLevel } from '../../types/prediction.ts'
import { getRiskMeta, toneDotClass, toneTextClass } from '../../utils/risk.ts'

export function StatusMarker({ status, risk }: { status?: 'Placed' | 'Not Placed'; risk?: RiskLevel }) {
  const meta = risk ? getRiskMeta(risk === 'Low Risk' ? .9 : risk === 'Moderate Risk' ? .6 : .2) : null
  const tone = status === 'Placed' ? 'positive' : status === 'Not Placed' ? 'negative' : meta?.tone || 'neutral'
  const label = status || risk || 'Not available'
  const color = tone === 'positive' ? 'text-positive' : tone === 'moderate' ? 'text-moderate' : tone === 'negative' ? 'text-negative' : 'text-meta'
  const dot = tone === 'positive' ? 'bg-positive' : tone === 'moderate' ? 'bg-moderate' : tone === 'negative' ? 'bg-negative' : 'bg-meta'
  return <span className={`inline-flex items-center gap-2 text-xs font-semibold ${color}`}><span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${dot}`} />{label}</span>
}

export function RiskBadge({ risk }: { risk: RiskLevel }) {
  const meta = getRiskMeta(risk === 'Low Risk' ? .9 : risk === 'Moderate Risk' ? .6 : risk === 'High Risk' ? .2 : .5)
  return <span className={`inline-flex items-center gap-2 text-xs font-semibold ${toneTextClass(meta.tone)}`}><span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${toneDotClass(meta.tone)}`} />{risk}
  </span>
}

export function ContextBadge({ children }: { children: ReactNode }) { return <span className="eyebrow rounded-sm border border-line bg-recessed px-2 py-1 text-[9px]">{children}</span> }
