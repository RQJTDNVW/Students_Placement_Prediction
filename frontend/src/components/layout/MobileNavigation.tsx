import { useState } from 'react'
import { BarChart3, ChevronUp, Gauge, History, Info, LayoutDashboard, MoreHorizontal, Settings, UserRoundSearch, X } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const primary = [{ label: 'Dashboard', to: '/', icon: LayoutDashboard }, { label: 'Predict', to: '/predict', icon: UserRoundSearch }, { label: 'History', to: '/history', icon: History }]
const more = [{ label: 'Student analytics', to: '/analytics', icon: BarChart3 }, { label: 'Model performance', to: '/model-performance', icon: Gauge }, { label: 'About project', to: '/about', icon: Info }, { label: 'Settings', to: '/settings', icon: Settings }]

export function MobileNavigation() {
  const [open, setOpen] = useState(false)
  return <>
    {open && <div className="fixed inset-0 z-40 bg-page/70 backdrop-blur-sm" onClick={() => setOpen(false)} aria-hidden="true" />}
    {open && <div className="fixed inset-x-0 bottom-[60px] z-50 border-t border-line bg-raised p-4 shadow-[0_-18px_44px_rgba(0,0,0,.25)]"><div className="mb-4 flex items-center justify-between"><p className="eyebrow">More destinations</p><button type="button" aria-label="Close menu" onClick={() => setOpen(false)} className="text-meta hover:text-ink"><X aria-hidden="true" className="h-4 w-4" /></button></div><div className="grid grid-cols-2 gap-2">{more.map(({ label, to, icon: Icon }) => <NavLink key={to} to={to} onClick={() => setOpen(false)} className="flex min-h-11 items-center gap-3 border border-line px-3 text-sm text-muted hover:border-cyan hover:text-cyan"><Icon aria-hidden="true" className="h-4 w-4" />{label}</NavLink>)}</div></div>}
    <nav className="fixed inset-x-0 bottom-0 z-40 flex h-[60px] items-stretch justify-around border-t border-line bg-sidebar px-2 lg:hidden" aria-label="Mobile navigation">
      {primary.map(({ label, to, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `flex min-w-16 flex-1 flex-col items-center justify-center gap-1 text-[10px] font-semibold ${isActive ? 'text-cyan' : 'text-meta'}`}><Icon aria-hidden="true" className="h-4 w-4" />{label}</NavLink>)}
      <button type="button" onClick={() => setOpen((current) => !current)} className={`flex min-w-16 flex-1 flex-col items-center justify-center gap-1 text-[10px] font-semibold ${open ? 'text-cyan' : 'text-meta'}`}>{open ? <ChevronUp aria-hidden="true" className="h-4 w-4" /> : <MoreHorizontal aria-hidden="true" className="h-4 w-4" />}More</button>
    </nav>
  </>
}
