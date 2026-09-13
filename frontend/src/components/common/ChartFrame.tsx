import type { ReactNode } from 'react'
import { ContextBadge } from './Badge.tsx'
import { Card, SectionHeader } from './Card.tsx'
import { EmptyState } from './EmptyState.tsx'
import { ErrorMessage } from './ErrorMessage.tsx'
import { SkeletonBlock } from './LoadingSpinner.tsx'

export function ChartFrame({ title, description, context = 'Saved predictions', children, state = 'ready', onRetry, tableFallback }: { title: string; description: string; context?: string; children: ReactNode; state?: 'loading' | 'error' | 'empty' | 'ready'; onRetry?: () => void; tableFallback?: ReactNode }) {
  return <Card className="p-4 sm:p-5">
    <SectionHeader eyebrow="Analysis" title={title} description={description} action={<ContextBadge>{context}</ContextBadge>} />
    {state === 'loading' && <div className="chart-bed flex h-64 items-center justify-center"><SkeletonBlock className="h-40 w-4/5" /></div>}
    {state === 'error' && <ErrorMessage message="This visualization could not be loaded." onRetry={onRetry} />}
    {state === 'empty' && <EmptyState title="Not enough saved predictions" description="This view will appear once enough prediction records are available." />}
    {state === 'ready' && <div className="chart-bed min-h-64 p-2 sm:p-4">{children}</div>}
    {tableFallback}
  </Card>
}
