import type { ReactNode } from 'react'
import { Inbox } from 'lucide-react'
import { Button } from './Button.tsx'

export function EmptyState({ title, description, action, icon = <Inbox aria-hidden="true" className="h-5 w-5" /> }: { title: string; description: string; action?: { label: string; onClick: () => void }; icon?: ReactNode }) {
  return <div className="flex min-h-48 flex-col items-center justify-center border-y border-line px-6 py-10 text-center">
    <div className="mb-4 text-cyan">{icon}</div>
    <h3 className="text-sm font-semibold text-ink">{title}</h3>
    <p className="mt-2 max-w-md text-sm leading-6 text-muted">{description}</p>
    {action && <Button variant="secondary" size="sm" className="mt-5" onClick={action.onClick}>{action.label}</Button>}
  </div>
}
