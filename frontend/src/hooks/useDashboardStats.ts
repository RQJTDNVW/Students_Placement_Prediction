import { useCallback, useEffect, useState } from 'react'
import { mockDashboardStats } from '../data/mockData.ts'
import { useApiActivity, usePreferences } from '../context/AppContext.tsx'
import { getDashboardStatistics } from '../services/api.ts'
import type { DashboardStats } from '../types/dashboard.ts'

export function useDashboardStats() {
  const { dataMode } = usePreferences()
  const { startRequest, endRequest } = useApiActivity()
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const refresh = useCallback(async () => {
    setIsLoading(true); setError(null); startRequest()
    try { setStats(dataMode === 'demo' ? mockDashboardStats : await getDashboardStatistics()) }
    catch { setError('Dashboard statistics could not be loaded.'); setStats(null) }
    finally { setIsLoading(false); endRequest() }
  }, [dataMode, endRequest, startRequest])
  useEffect(() => { void refresh() }, [refresh])
  return { stats, isLoading, error, refresh }
}
