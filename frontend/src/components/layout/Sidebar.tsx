import { NavLink } from 'react-router-dom'
import { Activity, BarChart3, Gauge, History, Info, LayoutDashboard, PanelLeftClose, PanelLeftOpen, Settings, UserRoundSearch } from 'lucide-react'

const groups = [
  { label: 'Workflow', items: [{ label: 'Dashboard', to: '/', icon: LayoutDashboard }, { label: 'Predict student', to: '/predict', icon: UserRoundSearch }, { label: 'Prediction history', to: '/history', icon: History }] },
  { label: 'Analysis', items: [{ label: 'Student analytics', to: '/analytics', icon: BarChart3 }, { label: 'Model performance', to: '/model-performance', icon: Gauge }] },
  { label: 'Project', items: [{ label: 'About project', to: '/about', icon: Info }, { label: 'Settings', to: '/settings', icon: Settings }] },
]

export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return <aside className={`relative hidden shrink-0 border-r border-line bg-sidebar transition-[width] duration-200 lg:block ${collapsed ? 'w-[72px]' : 'w-[248px]'}`}>
    <div className="flex h-full min-h-screen flex-col px-3 py-5">
      <div className={`mb-10 flex items-center ${collapsed ? 'justify-center' : 'justify-between px-2'}`}>
        <NavLink to="/" className="flex items-center gap-3" aria-label="Student Placement Intelligence dashboard">
          <span className="grid h-9 w-9 shrink-0 place-items-center border border-cyan/50 bg-cyan/10 text-cyan"><Activity aria-hidden="true" className="h-5 w-5" /></span>
          {!collapsed && <span><span className="block text-sm font-extrabold tracking-tight text-ink">SPI<span className="text-cyan">.</span></span><span className="eyebrow mt-1 block text-[9px]">Placement intelligence</span></span>}
        </NavLink>
        {!collapsed && <button type="button" aria-label="Collapse sidebar" onClick={onToggle} className="rounded-sm p-2 text-meta hover:bg-raised hover:text-cyan"><PanelLeftClose aria-hidden="true" className="h-4 w-4" /></button>}
      </div>
      {collapsed && <button type="button" aria-label="Expand sidebar" onClick={onToggle} className="mb-7 self-center rounded-sm p-2 text-meta hover:bg-raised hover:text-cyan"><PanelLeftOpen aria-hidden="true" className="h-4 w-4" /></button>}
      <nav className="flex-1 space-y-8" aria-label="Primary navigation">
        {groups.map((group) => <div key={group.label}>
          {!collapsed && <p className="eyebrow mb-3 px-3 text-[9px]">{group.label}</p>}
          <div className="space-y-1">
            {group.items.map(({ label, to, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} title={collapsed ? label : undefined} className={({ isActive }) => `group relative flex min-h-10 items-center gap-3 rounded-sm px-3 text-sm font-semibold transition-colors ${collapsed ? 'justify-center' : ''} ${isActive ? 'bg-cyan/10 text-cyan' : 'text-muted hover:bg-raised/60 hover:text-ink'}`}>
              {({ isActive }) => <><span className={`absolute left-0 h-5 w-0.5 transition-transform ${isActive ? 'bg-cyan scale-y-100' : 'scale-y-0 bg-transparent'}`} /><Icon aria-hidden="true" className="h-[17px] w-[17px] shrink-0" /><span className={collapsed ? 'sr-only' : ''}>{label}</span>{isActive && !collapsed && <span aria-hidden="true" className="ml-auto h-1.5 w-1.5 bg-cyan" />}</>}
            </NavLink>)}
          </div>
        </div>)}
      </nav>
      {!collapsed && <div className="measure-ticks border-l border-line pl-3"><p className="eyebrow text-[9px] leading-5">V4 / 27 features</p><p className="mt-2 text-xs leading-5 text-meta">An estimate is a reference, not a guarantee.</p></div>}
    </div>
  </aside>
}
