import { useCallback, useEffect, useMemo, useState } from 'react'
import { mockHistory } from '../data/mockData.ts'
import { useApiActivity, usePreferences } from '../context/AppContext.tsx'
import { deletePrediction, getPredictionHistory } from '../services/api.ts'
import type { PredictionRecord, PredictionStatus, RiskLevel } from '../types/prediction.ts'

export interface HistoryFilters { search: string; status: 'All' | PredictionStatus; risk: 'All' | RiskLevel; sort: 'probability' | 'date' }
const pageSize = 5

export function usePredictionHistory() {
  const { dataMode } = usePreferences()
  const { startRequest, endRequest } = useApiActivity()
  const [records, setRecords] = useState<PredictionRecord[]>([])
  const [filters, setFilters] = useState<HistoryFilters>({ search: '', status: 'All', risk: 'All', sort: 'date' })
  const [page, setPage] = useState(1)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const refresh = useCallback(async () => {
    setIsLoading(true); setError(null); startRequest()
    try { setRecords(dataMode === 'demo' ? mockHistory : await getPredictionHistory()) }
    catch { setError('Prediction history could not be loaded.'); setRecords([]) }
    finally { setIsLoading(false); endRequest() }
  }, [dataMode, endRequest, startRequest])
  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => { setPage(1) }, [filters])
  const filtered = useMemo(() => records.filter((record) => {
    const matchesSearch = !filters.search || record.prediction_id.toLowerCase().includes(filters.search.toLowerCase())
    const matchesStatus = filters.status === 'All' || record.status === filters.status
    const matchesRisk = filters.risk === 'All' || record.risk_level === filters.risk
    return matchesSearch && matchesStatus && matchesRisk
  }).sort((a, b) => filters.sort === 'probability' ? b.probability - a.probability : new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()), [filters, records])
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const pageRecords = filtered.slice((page - 1) * pageSize, page * pageSize)
  const remove = async (id: string) => { startRequest(); try { if (dataMode === 'production') await deletePrediction(id); setRecords((current) => current.filter((record) => record.prediction_id !== id)); return true } catch { return false } finally { endRequest() } }
  return { records: pageRecords, filteredCount: filtered.length, totalCount: records.length, page, pageSize, totalPages, filters, setFilters, setPage, isLoading, error, refresh, remove }
}
