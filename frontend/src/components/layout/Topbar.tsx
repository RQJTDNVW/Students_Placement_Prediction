import { Bell, Moon, Sun, UserRound } from 'lucide-react'
import { useLocation } from 'react-router-dom'
import { useApiActivity, usePreferences, useToasts } from '../../context/AppContext.tsx'
import { Breadcrumbs } from './Breadcrumbs.tsx'

export function Topbar() {
  const { pathname } = useLocation()
  const { theme, setTheme } = usePreferences()
  const { isApiLoading } = useApiActivity()
  const { pushToast } = useToasts()
  const isResult = pathname === '/predict/result'
  return <header className="relative flex h-[58px] shrink-0 items-center justify-between gap-4 border-b border-line bg-workspace/90 px-4 sm:px-6 lg:px-8">
    <Breadcrumbs />
    <div className="flex items-center gap-1.5">
      <span className="eyebrow mr-2 hidden text-[9px] sm:inline">{isResult ? 'Current session' : 'Student Placement Intelligence'}</span>
      <button type="button" aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="grid h-9 w-9 place-items-center rounded-sm text-meta hover:bg-raised hover:text-cyan">{theme === 'dark' ? <Sun aria-hidden="true" className="h-4 w-4" /> : <Moon aria-hidden="true" className="h-4 w-4" />}</button>
      <button type="button" aria-label="View notifications" onClick={() => pushToast({ tone: 'info', title: 'No new notifications', message: 'Prediction activity updates will appear here.' })} className="relative grid h-9 w-9 place-items-center rounded-sm text-meta hover:bg-raised hover:text-cyan"><Bell aria-hidden="true" className="h-4 w-4" /><span className="absolute right-2 top-2 h-1 w-1 rounded-full bg-cyan" /></button>
      <span className="ml-1 grid h-8 w-8 place-items-center rounded-full border border-line bg-raised text-meta" aria-label="Profile placeholder"><UserRound aria-hidden="true" className="h-4 w-4" /></span>
    </div>
    <div aria-hidden="true" className={`absolute inset-x-0 bottom-0 h-[2px] origin-left bg-cyan transition-transform duration-300 ${isApiLoading ? 'scale-x-100' : 'scale-x-0'}`} />
  </header>
}
