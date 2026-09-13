import { AlertTriangle, RefreshCw } from 'lucide-react'
import { Button } from './Button.tsx'

export function ErrorMessage({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div role="alert" className="flex flex-col gap-3 border border-negative/30 bg-negative/5 px-4 py-4 text-sm text-ink sm:flex-row sm:items-center sm:justify-between">
    <div className="flex items-start gap-3"><AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-negative" /><span>{message}</span></div>
    {onRetry && <Button variant="secondary" size="sm" onClick={onRetry}><RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />Retry</Button>}
  </div>
}
