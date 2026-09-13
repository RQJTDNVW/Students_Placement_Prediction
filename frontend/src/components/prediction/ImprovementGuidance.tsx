import { ArrowUpRight, CheckCircle2, Cpu, ListChecks, ShieldAlert } from 'lucide-react'
import type { PredictionResponse, StudentInput } from '../../types/prediction.ts'
import { getRecommendations } from '../../utils/recommendations.ts'
import { Card, SectionHeader } from '../common/Card.tsx'

export function ImprovementGuidance({ input, result }: { input: StudentInput; result?: PredictionResponse }) {
  const recommendations = getRecommendations(input)
  const posFactors = result?.top_positive_factors ?? []
  const negFactors = result?.top_negative_factors ?? []
  const hasShap = posFactors.length > 0 || negFactors.length > 0

  return (
    <div className="space-y-3">
      {hasShap && (
        <Card className="p-5 sm:p-6">
          <SectionHeader
            eyebrow="Explainable AI (TreeSHAP)"
            title="Model-Driven Feature Influences"
            description="Specific factors that contributed most positively or negatively to the XGBoost placement decision."
          />
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Positive Drivers */}
            <div className="rounded-lg border border-positive/20 bg-positive/5 p-4">
              <div className="mb-3 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-positive" />
                <h3 className="text-sm font-semibold text-ink">Top Positive Drivers</h3>
              </div>
              {posFactors.length > 0 ? (
                <ul className="space-y-2 text-xs text-muted">
                  {posFactors.map((factor, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-0.5 text-positive font-bold">+</span>
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-meta">No dominant positive factors detected.</p>
              )}
            </div>

            {/* Negative Detractors */}
            <div className="rounded-lg border border-negative/20 bg-negative/5 p-4">
              <div className="mb-3 flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-negative" />
                <h3 className="text-sm font-semibold text-ink">Primary Risk Factors</h3>
              </div>
              {negFactors.length > 0 ? (
                <ul className="space-y-2 text-xs text-muted">
                  {negFactors.map((factor, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-0.5 text-negative font-bold">−</span>
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-meta">No critical risk flags triggered by the model.</p>
              )}
            </div>
          </div>
          <p className="mt-4 flex items-center gap-2 text-[11px] text-meta">
            <Cpu aria-hidden="true" className="h-3.5 w-3.5 text-cyan" />
            Computed in real-time via XGBoost TreeSHAP local attribution algorithms.
          </p>
        </Card>
      )}

      <Card className="p-5 sm:p-6">
        <SectionHeader
          eyebrow="Rule-based guidance"
          title="Profile improvement areas"
          description="Actionable developmental suggestions based on domain benchmarks."
        />
        {recommendations.length ? (
          <div className="divide-y divide-line border-y border-line">
            {recommendations.map((item) => (
              <div key={item.title} className="flex gap-4 py-4">
                <span className="mt-0.5 text-moderate">
                  <ArrowUpRight aria-hidden="true" className="h-4 w-4" />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap justify-between gap-2">
                    <h3 className="text-sm font-semibold text-ink">{item.title}</h3>
                    <span className="mono text-[10px] text-meta">{item.value}</span>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-muted">{item.detail}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="border-y border-line py-8 text-center">
            <CheckCircle2 aria-hidden="true" className="mx-auto h-5 w-5 text-positive" />
            <p className="mt-3 text-sm font-semibold text-ink">No profile improvement rule was triggered.</p>
            <p className="mt-1 text-xs text-muted">The submitted values did not cross the current guidance thresholds.</p>
          </div>
        )}
        <p className="mt-4 flex items-center gap-2 text-[11px] text-meta">
          <ListChecks aria-hidden="true" className="h-3.5 w-3.5" />
          Transparent, rule-based suggestions for holistic career development.
        </p>
      </Card>
    </div>
  )
}

