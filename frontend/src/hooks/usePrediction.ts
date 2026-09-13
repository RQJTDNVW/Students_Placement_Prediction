import { useState } from 'react'
import { makeMockPrediction } from '../data/mockData.ts'
import { useApiActivity, usePreferences } from '../context/AppContext.tsx'
import { predictStudent } from '../services/api.ts'
import type { PredictionResponse, StudentInput } from '../types/prediction.ts'
import { isPredictionResponse } from '../utils/guards.ts'

export function usePrediction() {
  const { dataMode } = usePreferences()
  const { startRequest, endRequest } = useApiActivity()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const submit = async (input: StudentInput): Promise<PredictionResponse | null> => {
    setIsLoading(true); setError(null); startRequest()
    try {
      const result = dataMode === 'demo' ? makeMockPrediction(input) : await predictStudent(input)
      if (!isPredictionResponse(result)) throw new Error('The prediction service returned an unexpected response.')
      return result
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'The prediction service did not respond.'
      setError(dataMode === 'demo' ? message : 'The prediction service did not respond. Your entered values are still available.')
      return null
    } finally { setIsLoading(false); endRequest() }
  }
  return { submit, isLoading, error, clearError: () => setError(null) }
}
