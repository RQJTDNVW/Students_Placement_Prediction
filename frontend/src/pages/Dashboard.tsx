import { Activity, AlertTriangle, BarChart3, Gauge, Users } from 'lucide-react'
import { Link } from 'react-router-dom'
import { PageContainer } from '../components/layout/PageContainer.tsx'
import { Card, SectionHeader } from '../components/common/Card.tsx'
import { ContextBadge } from '../components/common/Badge.tsx'
import { EmptyState } from '../components/common/EmptyState.tsx'
import { ErrorMessage } from '../components/common/ErrorMessage.tsx'
import { StatCard } from '../components/dashboard/StatCard.tsx'
import { PlacementDistributionChart } from '../components/dashboard/PlacementDistributionChart.tsx'
import { ProbabilityDistributionChart } from '../components/dashboard/ProbabilityDistributionChart.tsx'
import { RiskDistributionChart } from '../components/dashboard/RiskDistributionChart.tsx'
import { RecentPredictions } from '../components/dashboard/RecentPredictions.tsx'
import { QuickActions } from '../components/dashboard/QuickActions.tsx'
import { CurrentSessionStrip } from '../components/dashboard/CurrentSessionStrip.tsx'
import { useDashboardStats } from '../hooks/useDashboardStats.ts'
import { useSessionPrediction, usePreferences } from '../context/AppContext.tsx'
import { formatCount, formatProbability } from '../utils/format.ts'

export default function Dashboard() {
  const { stats, isLoading, error, refresh } = useDashboardStats()
  const { currentPrediction } = useSessionPrediction()
  const { dataMode } = usePreferences()
  return <PageContainer title="Dashboard" description="Historical prediction activity and model reference metrics." context={dataMode === 'demo' ? 'Demo data' : 'Saved predictions'} actions={<Link to="/predict" className="primary-action inline-flex min-h-10 items-center gap-2 bg-cyan px-4 text-sm font-bold text-page hover:bg-cyan/85"><Users aria-hidden="true" className="h-4 w-4" />Predict student</Link>}>
    <CurrentSessionStrip session={currentPrediction} />
    {error && <div className="mb-6"><ErrorMessage message={error} onRetry={() => void refresh()} /></div>}
    {isLoading && <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><div className="h-32 animate-pulse bg-raised/50" /><div className="h-32 animate-pulse bg-raised/50" /><div className="h-32 animate-pulse bg-raised/50" /><div className="h-32 animate-pulse bg-raised/50" /></div>}
    {stats && <>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><StatCard label="Total students evaluated" value={formatCount(stats.total_students)} detail="Saved prediction records" icon={<Users className="h-4 w-4" />} /><StatCard label="Predicted placement rate" value={formatProbability(stats.placement_rate)} detail="Placed / total evaluated" icon={<BarChart3 className="h-4 w-4" />} tone="positive" /><StatCard label="Average probability" value={formatProbability(stats.average_probability)} detail="Across saved estimates" icon={<Activity className="h-4 w-4" />} tone="cyan" /><StatCard label="High-risk students" value={formatCount(stats.high_risk_students)} detail="Requires focused review" icon={<AlertTriangle className="h-4 w-4" />} tone="negative" /></div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)]"><Card className="p-4 sm:p-5"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="eyebrow mb-2">Historical activity</p><h2 className="text-lg font-semibold text-ink">Placement probability distribution</h2><p className="mt-1 text-sm text-muted">How saved predictions are distributed across estimated probability ranges.</p></div><ContextBadge>Saved predictions</ContextBadge></div><div className="mt-5"><ProbabilityDistributionChart data={stats.probability_distribution} /></div></Card><Card variant="solid" className="p-4 sm:p-5"><div className="flex items-start justify-between gap-3"><div><p className="eyebrow mb-2">Model reference</p><h2 className="text-lg font-semibold text-ink">XGBoost Technical</h2></div><Gauge aria-hidden="true" className="h-5 w-5 text-cyan" /></div><div className="mt-7 space-y-5"><div className="flex items-end justify-between border-b border-line pb-4"><span className="text-sm text-muted">Version</span><span className="mono text-sm text-ink">V4</span></div><div className="flex items-end justify-between border-b border-line pb-4"><span className="text-sm text-muted">Features</span><span className="mono text-sm text-ink">27</span></div><div className="flex items-end justify-between border-b border-line pb-4"><span className="text-sm text-muted">Internal ROC-AUC</span><span className="mono text-lg text-positive">68.22%</span></div><p className="text-xs leading-5 text-meta">Synthetic-data evaluation — not live prediction accuracy.</p></div></Card></div>
      <div className="mt-3 grid gap-3 lg:grid-cols-2"><Card className="p-4 sm:p-5"><SectionHeader eyebrow="Composition" title="Placement distribution" description="Saved predictions grouped by predicted placement status." action={<ContextBadge>Counts</ContextBadge>} /><PlacementDistributionChart data={stats.placement_distribution} /></Card><Card className="p-4 sm:p-5"><SectionHeader eyebrow="Composition" title="Risk distribution" description="Risk interpretation bands across saved estimates." action={<ContextBadge>Counts</ContextBadge>} /><RiskDistributionChart data={stats.risk_distribution} /></Card></div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,3fr)_minmax(240px,1fr)]"><Card className="overflow-hidden"><div className="p-4 pb-2 sm:p-5 sm:pb-2"><SectionHeader eyebrow="Saved predictions" title="Recent predictions" description="Latest records from the prediction history." action={<ContextBadge>{dataMode === 'demo' ? 'Demo data' : 'Saved predictions'}</ContextBadge>} /></div><RecentPredictions records={stats.recent_predictions} /></Card><Card className="p-4 sm:p-5"><SectionHeader eyebrow="Workflow" title="Quick actions" description="Move from overview to the next useful task." /><QuickActions /></Card></div>
    </>}
    {!isLoading && !error && !stats && <EmptyState title="Historical statistics will appear after predictions are saved." description="Start with a student profile to create the first estimate." action={{ label: 'Predict student', onClick: () => window.location.assign('/predict') }} />}
  </PageContainer>
}
