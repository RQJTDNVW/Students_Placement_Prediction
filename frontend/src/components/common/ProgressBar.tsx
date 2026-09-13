export function ProgressBar({ value, tone = 'cyan', label, showValue = true }: { value: number; tone?: 'cyan' | 'positive' | 'moderate' | 'negative'; label?: string; showValue?: boolean }) {
  const color = { cyan: 'bg-cyan', positive: 'bg-positive', moderate: 'bg-moderate', negative: 'bg-negative' }[tone]
  const safeValue = Math.min(100, Math.max(0, value))
  return <div className="space-y-2">
    {(label || showValue) && <div className="flex items-center justify-between gap-3 text-xs"><span className="text-muted">{label}</span>{showValue && <span className="mono text-ink">{Math.round(safeValue)}%</span>}</div>}
    <div className="h-1.5 overflow-hidden rounded-full bg-recessed"><div className={`h-full rounded-full transition-[width] duration-300 ${color}`} style={{ width: `${safeValue}%` }} /></div>
  </div>
}
