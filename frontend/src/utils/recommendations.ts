import type { StudentInput } from '../types/prediction.ts'

export interface Recommendation {
  title: string
  detail: string
  value: string
}

export function getRecommendations(input: StudentInput): Recommendation[] {
  const recommendations: Recommendation[] = []
  if (input.internships < 2) recommendations.push({ title: 'Gain internship experience', detail: 'Add a practical placement or internship opportunity to build exposure to real work.', value: `${input.internships} internships` })
  if (input.projects_count < 3) recommendations.push({ title: 'Build practical projects', detail: 'Document projects that show how you apply classroom knowledge to a concrete problem.', value: `${input.projects_count} projects` })
  if (input.certifications < 2) recommendations.push({ title: 'Consider relevant certifications', detail: 'Select focused certifications that support the roles you are preparing for.', value: `${input.certifications} certifications` })
  if (input.aptitude_score < 65) recommendations.push({ title: 'Practice placement aptitude tests', detail: 'Use timed practice sessions to strengthen quantitative and reasoning fundamentals.', value: `${input.aptitude_score}/100 aptitude` })
  if (input.communication_skills < 6) recommendations.push({ title: 'Strengthen communication practice', detail: 'Practice structured introductions, interviews, and concise explanations of your work.', value: `${input.communication_skills}/10 communication` })
  if (input.extracurricular_activities < 4) recommendations.push({ title: 'Participate in extracurricular activities', detail: 'Choose activities that help you demonstrate collaboration, initiative, or leadership.', value: `${input.extracurricular_activities}/10 involvement` })
  return recommendations
}
