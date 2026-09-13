import { Clock3, Cpu, FileChartColumn, Sparkles } from 'lucide-react'
import type { PredictionResponse } from '../../types/prediction.ts'
import { formatDateTime } from '../../utils/format.ts'
import { RiskBadge, StatusMarker } from '../common/Badge.tsx'
import { Card, ContextLabel } from '../common/Card.tsx'
import { ProbabilityGauge } from './ProbabilityGauge.tsx'

export function PredictionResultCard({ result }: { result: PredictionResponse }) {
  return (
    <Card className="overflow-hidden">
      <div className="grid items-center gap-7 p-5 sm:p-7 lg:grid-cols-[minmax(250px,1fr)_minmax(0,1.25fr)]">
        <div className="flex justify-center border-b border-line pb-7 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-7">
          <ProbabilityGauge probability={result.probability} />
        </div>
        <div>
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
            <ContextLabel>Current session</ContextLabel>
            <span className="eyebrow flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-positive" />
              Model estimate
            </span>
          </div>
          <p className="eyebrow mb-3">Main prediction</p>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <h2 className="text-3xl font-extrabold tracking-[-0.05em] text-ink">{result.status}</h2>
            <StatusMarker status={result.status} />
            <RiskBadge risk={result.risk_level} />
          </div>

          {result.expected_salary_lpa ? (
            <div className="mt-4 flex flex-wrap items-center gap-2.5 rounded-lg border border-cyan/30 bg-cyan/5 px-3.5 py-2">
              <Sparkles aria-hidden="true" className="h-4 w-4 text-cyan" />
              <span className="text-xs font-semibold text-ink">Estimated Salary:</span>
              <span className="mono text-sm font-bold text-cyan">{result.expected_salary_lpa} LPA</span>
              {result.salary_range && (
                <span className="mono rounded bg-recessed px-2 py-0.5 text-[11px] text-muted">
                  Bracket: {result.salary_range}
                </span>
              )}
            </div>
          ) : null}

          <p className="mt-4 max-w-lg text-sm leading-6 text-muted">
            Estimated placement probability based on the submitted student profile. This is a model-based interpretation, not a guaranteed outcome.
          </p>
          <dl className="mt-7 grid gap-x-6 gap-y-4 border-t border-line pt-5 sm:grid-cols-3">
            <Meta label="Model" value={result.model_name || 'XGBoost'} icon={<Cpu aria-hidden="true" className="h-3.5 w-3.5" />} />
            <Meta label="Version" value={result.model_version || 'V4'} icon={<FileChartColumn aria-hidden="true" className="h-3.5 w-3.5" />} />
            <Meta label="Timestamp" value={result.timestamp ? formatDateTime(result.timestamp) : 'Just now'} icon={<Clock3 aria-hidden="true" className="h-3.5 w-3.5" />} />
          </dl>
        </div>
      </div>
    </Card>
  )
}

function Meta({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div>
      <dt className="eyebrow flex items-center gap-2 text-[9px]">
        {icon}
        {label}
      </dt>
      <dd className="mono mt-2 text-xs text-ink">{value}</dd>
    </div>
  )
}

