import type { ReactNode } from 'react'
import { ContextLabel } from '../common/Card.tsx'
import { ContextBadge } from '../common/Badge.tsx'

export function PageContainer({ title, description, context, children, actions }: { title: string; description: string; context?: string; children: ReactNode; actions?: ReactNode }) {
  return <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 py-7 sm:px-6 lg:px-8 lg:py-9">
    <div className="mb-8 flex flex-wrap items-end justify-between gap-5 border-b border-line pb-7">
      <div><div className="mb-3 flex items-center gap-3">{context && <ContextBadge>{context}</ContextBadge>}<span className="eyebrow">Student Placement Intelligence</span></div><h1 className="text-2xl font-extrabold tracking-[-0.04em] text-ink sm:text-3xl">{title}</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-muted">{description}</p></div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
    {children}
  </main>
}

export function CurrentSessionLabel() { return <ContextLabel>Current session</ContextLabel> }
