export type PredictionStatus = 'Placed' | 'Not Placed'
export type RiskLevel = 'Low Risk' | 'Moderate Risk' | 'High Risk' | 'Not Available'

export interface ShapFactor {
  feature: string
  impact: number
  description: string
}

export interface StudentInput {
  cgpa: number
  internships: number
  projects_count: number
  certifications: number
  aptitude_score: number
  communication_skills: number
  extracurricular_activities: number

  // Optional technical inputs (V4 Tech Model)
  coding_skills?: number
  dsa_score?: number
  backlogs?: number
  college_tier?: 'Tier-1' | 'Tier-2' | 'Tier-3'
  branch?: string
  hackathons?: number
  open_source_contributions?: number
  system_design?: number
  ml_knowledge?: number
  model_preference?: 'v3' | 'v4'
}

export interface PredictionResponse {
  status: PredictionStatus
  prediction: 0 | 1
  probability: number
  risk_level: RiskLevel
  experience_score?: number
  skill_score?: number
  prediction_id?: string
  timestamp?: string
  model_name?: string
  model_version?: string

  // Multi-Stage & Explainability Fields
  expected_salary_lpa?: number | null
  salary_range?: string | null
  top_positive_factors?: string[]
  top_negative_factors?: string[]
  shap_factors?: ShapFactor[]
}

export interface PredictionRecord extends StudentInput, PredictionResponse {
  prediction_id: string
  timestamp: string
  saved: boolean
}

