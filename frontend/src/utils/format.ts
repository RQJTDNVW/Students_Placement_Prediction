import type { ModelMetric } from '../types/model.ts'

export const formatProbability = (value: number) => `${Math.round(value * 100)}%`
export const formatMetric = (value: number, metric: ModelMetric) => metric === 'training_time' ? `${value.toFixed(2)} s` : `${(value * 100).toFixed(2)}%`
export const formatDateTime = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
export const formatCount = (value: number) => new Intl.NumberFormat().format(value)
export const clamp = (value: number, min = 0, max = 100) => Math.min(max, Math.max(min, value))
