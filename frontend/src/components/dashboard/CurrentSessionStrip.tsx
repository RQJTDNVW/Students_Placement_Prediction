import { ArrowRight, CircleDashed } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { StudentInput, PredictionResponse } from '../../types/prediction.ts'
import { formatProbability } from '../../utils/format.ts'
import { RiskBadge, StatusMarker } from '../common/Badge.tsx'
import { ContextBadge } from '../common/Badge.tsx'

export function CurrentSessionStrip({ session }: { session: { input: StudentInput; result: PredictionResponse; saved: boolean } | null }) {
  return <div className="mb-6 border border-line bg-recessed/70 px-4 py-3 sm:px-5"><div className="flex flex-wrap items-center gap-x-5 gap-y-3"><ContextBadge>Current session</ContextBadge>{session ? <><StatusMarker status={session.result.status} /><span className="mono text-sm text-ink">{formatProbability(session.result.probability)} estimated probability</span><RiskBadge risk={session.result.risk_level} /><Link to="/predict/result" className="ml-auto inline-flex items-center gap-1 text-xs font-semibold text-cyan hover:underline">Open result <ArrowRight aria-hidden="true" className="h-3.5 w-3.5" /></Link></> : <><CircleDashed aria-hidden="true" className="h-4 w-4 text-meta" /><span className="text-sm text-muted">No prediction has been requested in this session.</span><Link to="/predict" className="ml-auto text-xs font-semibold text-cyan hover:underline">Predict student</Link></>}</div></div>
}
