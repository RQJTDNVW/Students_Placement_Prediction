import type { PredictionRecord, PredictionStatus, RiskLevel } from './prediction.ts'

export interface DistributionItem {
  name: string
  value: number
}

export interface ProbabilityBin {
  range: string
  count: number
}

export interface DashboardStats {
  total_students: number
  placement_rate: number
  average_probability: number
  high_risk_students: number
  placement_distribution: DistributionItem[]
  probability_distribution: ProbabilityBin[]
  risk_distribution: Array<{ name: RiskLevel; value: number }>
  recent_predictions: PredictionRecord[]
}

export interface AnalyticsData {
  probability_distribution: ProbabilityBin[]
  placement_distribution: Array<{ name: PredictionStatus; value: number }>
  risk_distribution: Array<{ name: RiskLevel; value: number }>
  probability_by_cgpa: Array<{ range: string; probability: number; sample_size: number }>
  probability_by_internships: Array<{ count: string; probability: number; sample_size: number }>
  probability_by_projects: Array<{ count: string; probability: number; sample_size: number }>
  profile_comparison: Array<{ label: string; value: number }>
}
