import type { PredictionResponse, PredictionStatus, RiskLevel } from '../types/prediction.ts'

const statuses: PredictionStatus[] = ['Placed', 'Not Placed']
const risks: RiskLevel[] = ['Low Risk', 'Moderate Risk', 'High Risk', 'Not Available']

export function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  return statuses.includes(item.status as PredictionStatus)
    && (item.prediction === 0 || item.prediction === 1)
    && typeof item.probability === 'number'
    && item.probability >= 0 && item.probability <= 1
    && risks.includes(item.risk_level as RiskLevel)
}
