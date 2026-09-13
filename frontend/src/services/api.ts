import axios from 'axios'
import type { AnalyticsData, DashboardStats } from '../types/dashboard.ts'
import type { ModelEvaluation } from '../types/model.ts'
import type { PredictionRecord, PredictionResponse, StudentInput } from '../types/prediction.ts'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000', timeout: 10_000, headers: { 'Content-Type': 'application/json' } })

export async function checkHealth(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>('/health')
  return response.data
}

export async function predictStudent(input: StudentInput): Promise<PredictionResponse> {
  // This is the only production prediction boundary. FastAPI receives JSON and keeps model artifacts server-side.
  const response = await api.post<PredictionResponse>('/predict', input)
  return response.data
}

export async function getPredictionHistory(): Promise<PredictionRecord[]> {
  const response = await api.get<PredictionRecord[] | { predictions: PredictionRecord[] }>('/predictions')
  return Array.isArray(response.data) ? response.data : response.data.predictions
}

export async function savePrediction(prediction: PredictionRecord): Promise<PredictionRecord> {
  const response = await api.post<PredictionRecord>('/predictions', prediction)
  return response.data
}

export async function deletePrediction(id: string): Promise<void> {
  await api.delete(`/predictions/${encodeURIComponent(id)}`)
}

export async function getDashboardStatistics(): Promise<DashboardStats> {
  const response = await api.get<DashboardStats>('/dashboard/statistics')
  return response.data
}

export async function getModelMetrics(): Promise<ModelEvaluation> {
  const response = await api.get<ModelEvaluation>('/model/metrics')
  return response.data
}

export async function getAnalyticsData(): Promise<AnalyticsData> {
  const response = await api.get<AnalyticsData>('/analytics')
  return response.data
}
