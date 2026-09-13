import type { HTMLAttributes, ReactNode } from 'react'

interface CardProps extends HTMLAttributes<HTMLElement> {
  variant?: 'panel' | 'solid' | 'recessed' | 'flat'
  children: ReactNode
}

export function Card({ variant = 'panel', className = '', children, ...props }: CardProps) {
  const styles = { panel: 'panel', solid: 'panel-solid', recessed: 'chart-bed', flat: 'border-y border-line bg-transparent' }
  return <section className={`${styles[variant]} ${className}`} {...props}>{children}</section>
}

export function SectionHeader({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: ReactNode }) {
  return <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
    <div>
      {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
      <h2 className="text-lg font-semibold tracking-[-0.02em] text-ink">{title}</h2>
      {description && <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">{description}</p>}
    </div>
    {action}
  </div>
}

export function ContextLabel({ children }: { children: ReactNode }) { return <span className="eyebrow inline-flex items-center gap-2 rounded-sm border border-line px-2 py-1">{children}</span> }
