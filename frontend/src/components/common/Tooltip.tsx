import { CircleHelp } from 'lucide-react'

export function HelpTooltip({ label }: { label: string }) {
  return <span className="group relative inline-flex align-middle">
    <button type="button" aria-label={label} className="text-meta hover:text-cyan"><CircleHelp aria-hidden="true" className="h-3.5 w-3.5" /></button>
    <span role="tooltip" className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 hidden w-56 -translate-x-1/2 rounded-[7px] border border-line bg-raised px-3 py-2 text-left text-xs leading-5 text-muted shadow-lg group-hover:block group-focus-within:block">{label}</span>
  </span>
}
