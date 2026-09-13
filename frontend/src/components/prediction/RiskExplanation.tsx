import { ShieldCheck } from 'lucide-react'
import type { PredictionResponse } from '../../types/prediction.ts'
import { getRiskMeta, toneTextClass } from '../../utils/risk.ts'
import { Card } from '../common/Card.tsx'

export function RiskExplanation({ result }: { result: PredictionResponse }) { const meta = getRiskMeta(result.probability); return <Card variant="solid" className="p-5"><div className="flex gap-4"><span className={`mt-0.5 ${toneTextClass(meta.tone)}`}><ShieldCheck aria-hidden="true" className="h-5 w-5" /></span><div><p className="eyebrow mb-2">Risk explanation</p><h2 className="text-base font-semibold text-ink">{meta.label}</h2><p className="mt-2 text-sm leading-6 text-muted">{meta.explanation}</p><p className="mt-3 text-xs font-semibold text-meta">This is a model-based interpretation, not a guaranteed outcome.</p></div></div></Card> }
