import type { AnalyticsData, DashboardStats } from '../types/dashboard.ts'
import type { ModelEvaluation } from '../types/model.ts'
import type { PredictionRecord, PredictionResponse, StudentInput } from '../types/prediction.ts'
import { getRiskMeta } from '../utils/risk.ts'

const now = Date.now()
const makeDate = (hoursAgo: number) => new Date(now - hoursAgo * 3_600_000).toISOString()

export const mockHistory: PredictionRecord[] = [
  { prediction_id: 'PR-1042', cgpa: 8.6, internships: 2, projects_count: 4, certifications: 3, aptitude_score: 84, communication_skills: 8, extracurricular_activities: 7, status: 'Placed', prediction: 1, probability: .91, risk_level: 'Low Risk', experience_score: 8.2, skill_score: 8.1, timestamp: makeDate(2), saved: true },
  { prediction_id: 'PR-1041', cgpa: 7.4, internships: 1, projects_count: 2, certifications: 1, aptitude_score: 68, communication_skills: 6, extracurricular_activities: 5, status: 'Placed', prediction: 1, probability: .72, risk_level: 'Moderate Risk', experience_score: 5.4, skill_score: 6.1, timestamp: makeDate(8), saved: true },
  { prediction_id: 'PR-1040', cgpa: 6.2, internships: 0, projects_count: 1, certifications: 0, aptitude_score: 52, communication_skills: 4, extracurricular_activities: 2, status: 'Not Placed', prediction: 0, probability: .31, risk_level: 'High Risk', experience_score: 2.1, skill_score: 3.8, timestamp: makeDate(22), saved: true },
  { prediction_id: 'PR-1039', cgpa: 8.1, internships: 3, projects_count: 5, certifications: 2, aptitude_score: 77, communication_skills: 7, extracurricular_activities: 8, status: 'Placed', prediction: 1, probability: .86, risk_level: 'Low Risk', experience_score: 9.1, skill_score: 7.8, timestamp: makeDate(30), saved: true },
  { prediction_id: 'PR-1038', cgpa: 7.0, internships: 1, projects_count: 3, certifications: 1, aptitude_score: 61, communication_skills: 5, extracurricular_activities: 4, status: 'Not Placed', prediction: 0, probability: .48, risk_level: 'High Risk', experience_score: 4.8, skill_score: 5.2, timestamp: makeDate(44), saved: true },
  { prediction_id: 'PR-1037', cgpa: 9.0, internships: 4, projects_count: 6, certifications: 4, aptitude_score: 93, communication_skills: 9, extracurricular_activities: 8, status: 'Placed', prediction: 1, probability: .97, risk_level: 'Low Risk', experience_score: 10, skill_score: 9.4, timestamp: makeDate(52), saved: true },
  { prediction_id: 'PR-1036', cgpa: 7.8, internships: 2, projects_count: 2, certifications: 2, aptitude_score: 73, communication_skills: 7, extracurricular_activities: 5, status: 'Placed', prediction: 1, probability: .79, risk_level: 'Low Risk', experience_score: 6.8, skill_score: 6.9, timestamp: makeDate(67), saved: true },
  { prediction_id: 'PR-1035', cgpa: 5.9, internships: 0, projects_count: 0, certifications: 1, aptitude_score: 47, communication_skills: 5, extracurricular_activities: 3, status: 'Not Placed', prediction: 0, probability: .24, risk_level: 'High Risk', experience_score: 1.7, skill_score: 3.9, timestamp: makeDate(79), saved: true },
]

export function makeMockPrediction(input: StudentInput): PredictionResponse {
  const academic = (input.cgpa / 10) * .3 + (input.aptitude_score / 100) * .25
  const experience = Math.min(1, (input.internships * .16) + (input.projects_count * .06) + (input.certifications * .04))
  const skills = (input.communication_skills / 10) * .15 + (input.extracurricular_activities / 10) * .1
  const probability = Math.min(.98, Math.max(.08, academic + experience + skills))
  const meta = getRiskMeta(probability)
  const isPlaced = probability >= .5
  const expectedSalary = isPlaced ? Number((12.5 + (input.cgpa * 0.6) + (input.internships * 1.5) + (input.projects_count * 0.4)).toFixed(2)) : null
  const salaryRange = expectedSalary ? `${(expectedSalary - 1.0).toFixed(1)} - ${(expectedSalary + 1.0).toFixed(1)} LPA` : null

  const topPos = isPlaced ? [
    `Strong CGPA standing (${input.cgpa.toFixed(1)})`,
    input.internships > 0 ? `${input.internships} Practical Internship(s)` : 'General Aptitude Performance',
    'Applied Project Portfolio'
  ] : []

  const topNeg = !isPlaced ? [
    input.internships === 0 ? 'Zero Internships Completed' : 'Aptitude Score Below Target',
    input.projects_count <= 1 ? 'Limited Practical Projects' : 'Academic Standing Margin'
  ] : []

  return {
    status: isPlaced ? 'Placed' : 'Not Placed',
    prediction: isPlaced ? 1 : 0,
    probability,
    risk_level: meta.risk,
    experience_score: Math.min(10, input.internships * 2.1 + input.projects_count * .8 + input.certifications * .55),
    skill_score: Math.min(10, input.communication_skills * .65 + input.extracurricular_activities * .35),
    prediction_id: `PR-${1043 + mockHistory.length}`,
    timestamp: new Date().toISOString(),
    model_name: 'XGBoost Tech',
    model_version: 'V4',
    expected_salary_lpa: expectedSalary,
    salary_range: salaryRange,
    top_positive_factors: topPos,
    top_negative_factors: topNeg,
  }
}

export const mockDashboardStats: DashboardStats = {
  total_students: 126,
  placement_rate: .68,
  average_probability: .71,
  high_risk_students: 29,
  placement_distribution: [{ name: 'Placed', value: 86 }, { name: 'Not Placed', value: 40 }],
  probability_distribution: [{ range: '0–20%', count: 7 }, { range: '21–40%', count: 18 }, { range: '41–60%', count: 15 }, { range: '61–80%', count: 34 }, { range: '81–100%', count: 52 }],
  risk_distribution: [{ name: 'Low Risk', value: 52 }, { name: 'Moderate Risk', value: 45 }, { name: 'High Risk', value: 29 }],
  recent_predictions: mockHistory.slice(0, 5),
}

export const mockAnalytics: AnalyticsData = {
  probability_distribution: mockDashboardStats.probability_distribution,
  placement_distribution: mockDashboardStats.placement_distribution as AnalyticsData['placement_distribution'],
  risk_distribution: mockDashboardStats.risk_distribution,
  probability_by_cgpa: [{ range: '5–6', probability: .34, sample_size: 16 }, { range: '6–7', probability: .49, sample_size: 29 }, { range: '7–8', probability: .68, sample_size: 38 }, { range: '8–9', probability: .82, sample_size: 31 }, { range: '9–10', probability: .93, sample_size: 12 }],
  probability_by_internships: [{ count: '0', probability: .42, sample_size: 37 }, { count: '1', probability: .61, sample_size: 43 }, { count: '2', probability: .76, sample_size: 31 }, { count: '3+', probability: .88, sample_size: 15 }],
  probability_by_projects: [{ count: '0–1', probability: .41, sample_size: 27 }, { count: '2–3', probability: .63, sample_size: 46 }, { count: '4–5', probability: .78, sample_size: 37 }, { count: '6+', probability: .91, sample_size: 16 }],
  profile_comparison: [{ label: 'Academic strength', value: 77 }, { label: 'Experience', value: 64 }, { label: 'Technical / aptitude', value: 73 }, { label: 'Communication', value: 69 }, { label: 'Extracurricular', value: 61 }],
}

export const mockModelMetrics: ModelEvaluation = {
  model_name: 'XGBoost Technical Classifier + Salary Regressor', version: 'V4', feature_count: 27, training_time_seconds: 3.10,
  internal: { accuracy: .6969, f1_score: .8069, roc_auc: .6822 },
  external: { accuracy: .783, f1_score: .7395, roc_auc: .8565 },
  comparison: [
    { model: 'Logistic Regression', accuracy: .641, f1_score: .712, roc_auc: .623, pr_auc: .648, training_time: 1.12 },
    { model: 'Random Forest', accuracy: .712, f1_score: .749, roc_auc: .733, pr_auc: .709, training_time: 2.42 },
    { model: 'Extra Trees', accuracy: .704, f1_score: .741, roc_auc: .721, pr_auc: .697, training_time: 1.98 },
    { model: 'Gradient Boosting', accuracy: .681, f1_score: .728, roc_auc: .694, pr_auc: .672, training_time: 2.61 },
    { model: 'Hist Gradient Boosting', accuracy: .697, f1_score: .735, roc_auc: .714, pr_auc: .688, training_time: 2.14 },
    { model: 'XGBoost', accuracy: .6663, f1_score: .7634, roc_auc: .6545, pr_auc: .721, training_time: 2.95 },
    { model: 'Support Vector Machine', accuracy: .657, f1_score: .702, roc_auc: .641, pr_auc: .629, training_time: 3.48 },
  ],
}
