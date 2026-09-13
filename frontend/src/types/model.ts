export type ModelMetric = 'accuracy' | 'f1_score' | 'roc_auc' | 'pr_auc' | 'training_time'

export interface ModelComparisonRow {
  model: string
  accuracy: number
  f1_score: number
  roc_auc: number
  pr_auc: number
  training_time: number
}

export interface ModelEvaluation {
  model_name: string
  version: string
  feature_count: number
  training_time_seconds: number
  internal: Record<'accuracy' | 'f1_score' | 'roc_auc', number>
  external: Record<'accuracy' | 'f1_score' | 'roc_auc', number>
  comparison: ModelComparisonRow[]
}
