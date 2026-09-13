import { useEffect, useState } from 'react'
import { Database, Table2 } from 'lucide-react'
import { PageContainer } from '../components/layout/PageContainer.tsx'
import { Card, SectionHeader } from '../components/common/Card.tsx'
import { ContextBadge } from '../components/common/Badge.tsx'
import { ErrorMessage } from '../components/common/ErrorMessage.tsx'
import { EmptyState } from '../components/common/EmptyState.tsx'
import { Button } from '../components/common/Button.tsx'
import { AnalyticsCards } from '../components/analytics/AnalyticsCards.tsx'
import { AnalyticsCharts } from '../components/analytics/AnalyticsCharts.tsx'
import { ProfileComparisonChart } from '../components/analytics/ProfileComparisonChart.tsx'
import { mockAnalytics } from '../data/mockData.ts'
import { getAnalyticsData } from '../services/api.ts'
import { useApiActivity, usePreferences } from '../context/AppContext.tsx'
import type { AnalyticsData } from '../types/dashboard.ts'

export default function StudentAnalytics() {
  const { dataMode } = usePreferences()
  const { startRequest, endRequest } = useApiActivity()
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    setLoading(true)
    startRequest()
    void (async () => {
      try {
        const next = dataMode === 'demo' ? mockAnalytics : await getAnalyticsData()
        if (active) setData(next)
      } catch {
        if (active) setError('Analytics could not be loaded.')
      } finally {
        if (active) {
          setLoading(false)
          endRequest()
        }
      }
    })()
    return () => {
      active = false
      endRequest()
    }
  }, [dataMode, endRequest, startRequest])

  return <PageContainer title="Student analytics" description="Aggregated patterns across saved placement estimates." context={dataMode === 'demo' ? 'Demo data' : 'Saved predictions'}>
    {error && <div className="mb-5"><ErrorMessage message={error} /></div>}
    {loading && <div className="grid gap-3 md:grid-cols-2"><div className="h-72 animate-pulse bg-raised/50" /><div className="h-72 animate-pulse bg-raised/50" /></div>}
    {!loading && data && <>
      <AnalyticsCards total={data.placement_distribution.reduce((sum, item) => sum + item.value, 0)} placed={Math.round((data.placement_distribution.find((item) => item.name === 'Placed')?.value || 0) / Math.max(1, data.placement_distribution.reduce((sum, item) => sum + item.value, 0)) * 100)} average={Math.round(data.probability_distribution.reduce((sum, item) => sum + (Number(item.range.split('–')[0]) + Number(item.range.split('–')[1].replace('%', ''))) / 2 * item.count, 0) / Math.max(1, data.probability_distribution.reduce((sum, item) => sum + item.count, 0)))} highRisk={data.risk_distribution.find((item) => item.name === 'High Risk')?.value || 0} />
      <Card className="mb-3 p-4 sm:p-5"><SectionHeader eyebrow="Distribution" title="Placement probability distribution" description="The number of saved predictions within each estimated probability range." action={<ContextBadge>Saved predictions</ContextBadge>} /><div className="grid items-center gap-4 md:grid-cols-[minmax(0,1fr)_220px]"><div className="flex h-40 items-end gap-2 border-b border-l border-line px-3 pb-0 pt-4">{data.probability_distribution.map((item) => <div key={item.range} className="flex h-full flex-1 flex-col justify-end gap-2"><div className="bg-cyan/75 transition-all" style={{ height: `${Math.max(12, item.count / 52 * 100)}%` }} title={`${item.range}: ${item.count} predictions`} /><span className="mono whitespace-nowrap text-center text-[9px] text-meta">{item.range}</span></div>)}</div><div className="border-l border-line pl-4"><p className="eyebrow text-[9px]">Reading the chart</p><p className="mt-2 text-xs leading-5 text-muted">Higher bars indicate more saved estimates in that probability band. This is an aggregate view, not a guarantee for an individual student.</p></div></div><div className="mt-4 flex items-center gap-2 text-[11px] text-meta"><Table2 aria-hidden="true" className="h-3.5 w-3.5" />Exact bins are available through the visualization tooltips.</div></Card>
      <AnalyticsCharts data={data} />
      <div className="mt-3"><ProfileComparisonChart data={data.profile_comparison} /></div>
      <div className="mt-3 border border-line bg-recessed/60 p-4 text-sm leading-6 text-muted"><Database aria-hidden="true" className="mb-2 h-4 w-4 text-cyan" /><strong className="font-semibold text-ink">Interpretation note. </strong>These views summarize saved model estimates. They show association within this prediction history and do not establish that a student characteristic causes placement.</div>
    </>}
    {!loading && !data && !error && <EmptyState title="Analytics will appear after predictions are saved." description="Build a saved prediction history to explore aggregate patterns." action={{ label: 'Go to prediction form', onClick: () => window.location.assign('/predict') }} />}
    <div className="sr-only"><Button onClick={() => undefined}>View data table</Button></div>
  </PageContainer>
}
