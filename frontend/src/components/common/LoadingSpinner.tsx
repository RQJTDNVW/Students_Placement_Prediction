export function LoadingSpinner({ label = 'Loading', size = 'md' }: { label?: string; size?: 'sm' | 'md' | 'lg' }) {
  const sizes = { sm: 'h-4 w-4', md: 'h-6 w-6', lg: 'h-9 w-9' }
  return <span role="status" aria-label={label} className={`inline-block animate-spin rounded-full border-2 border-line border-t-cyan ${sizes[size]}`} />
}

export function SkeletonBlock({ className = '' }: { className?: string }) { return <div aria-hidden="true" className={`animate-pulse rounded-sm bg-raised/70 ${className}`} /> }
