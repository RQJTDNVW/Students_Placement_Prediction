import { useState } from 'react'
import { motion } from 'framer-motion'
import { useLocation, Outlet } from 'react-router-dom'
import { ToastRegion } from '../common/Toast.tsx'
import { MobileNavigation } from './MobileNavigation.tsx'
import { Sidebar } from './Sidebar.tsx'
import { Topbar } from './Topbar.tsx'

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false)
  const { pathname } = useLocation()
  return <div className="flex min-h-screen bg-page text-ink">
    <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((current) => !current)} />
    <div className="flex min-w-0 flex-1 flex-col bg-workspace">
      <Topbar />
      <motion.div key={pathname} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .18 }} className="flex min-h-0 flex-1 flex-col"><Outlet /></motion.div>
    </div>
    <MobileNavigation />
    <ToastRegion />
  </div>
}
