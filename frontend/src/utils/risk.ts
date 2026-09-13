import type { RiskLevel } from '../types/prediction.ts'

export interface RiskMeta {
  risk: RiskLevel
  tone: 'positive' | 'moderate' | 'negative' | 'neutral'
  label: string
  explanation: string
}

export function getRiskMeta(probability: number): RiskMeta {
  if (probability >= 0.75) return { risk: 'Low Risk', tone: 'positive', label: 'Low risk', explanation: 'The model estimates a strong likelihood of placement based on the submitted profile.' }
  if (probability >= 0.5) return { risk: 'Moderate Risk', tone: 'moderate', label: 'Moderate risk', explanation: 'The model estimates an uncertain outcome. Strengthening relevant skills or experience may improve the profile.' }
  return { risk: 'High Risk', tone: 'negative', label: 'High risk', explanation: 'The model estimates a lower likelihood of placement. Focused improvement in the identified areas may help.' }
}

export const toneTextClass = (tone: RiskMeta['tone']) => ({ positive: 'text-positive', moderate: 'text-moderate', negative: 'text-negative', neutral: 'text-meta' }[tone])
export const toneDotClass = (tone: RiskMeta['tone']) => ({ positive: 'bg-positive', moderate: 'bg-moderate', negative: 'bg-negative', neutral: 'bg-meta' }[tone])
