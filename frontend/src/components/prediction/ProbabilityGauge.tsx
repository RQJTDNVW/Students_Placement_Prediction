import { useId } from 'react'
import { getRiskMeta, toneTextClass } from '../../utils/risk.ts'
import { formatProbability } from '../../utils/format.ts'

export function ProbabilityGauge({ probability }: { probability: number }) {
  const id = useId()
  const meta = getRiskMeta(probability)
  const radius = 82
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - probability)
  const stroke = meta.tone === 'positive' ? 'var(--positive)' : meta.tone === 'moderate' ? 'var(--moderate)' : 'var(--negative)'
  return <div className="flex flex-col items-center"><div className="relative h-56 w-56"><svg viewBox="0 0 200 200" className="h-full w-full -rotate-90" role="img" aria-labelledby={`${id}-title ${id}-desc`}><title id={`${id}-title`}>{formatProbability(probability)} estimated placement probability</title><desc id={`${id}-desc`}>A circular indicator showing a model-based estimate, not a guarantee.</desc><circle cx="100" cy="100" r={radius} fill="none" stroke="var(--line)" strokeWidth="12" strokeDasharray="2 8" /><circle cx="100" cy="100" r={radius} fill="none" stroke={stroke} strokeWidth="8" strokeLinecap="square" strokeDasharray={circumference} strokeDashoffset={offset} className="transition-[stroke-dashoffset] duration-500" /></svg><div className="absolute inset-0 flex flex-col items-center justify-center"><span className="mono text-4xl font-semibold tracking-[-0.06em] text-ink">{formatProbability(probability)}</span><span className="mt-2 text-[10px] uppercase tracking-[.13em] text-meta">estimated</span></div></div><p className={`mono mt-1 text-xs font-semibold ${toneTextClass(meta.tone)}`}>{meta.label}</p></div>
}
