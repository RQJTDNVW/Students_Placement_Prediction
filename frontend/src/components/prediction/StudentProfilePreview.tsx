import { Activity, BrainCircuit, BriefcaseBusiness, CircleHelp } from 'lucide-react'
import type { StudentInput } from '../../types/prediction.ts'
import { clamp } from '../../utils/format.ts'
import { Card } from '../common/Card.tsx'
import { ProgressBar } from '../common/ProgressBar.tsx'

function average(values: Array<number | undefined>) { const present = values.filter((value): value is number => typeof value === 'number'); return present.length ? present.reduce((sum, value) => sum + value, 0) / present.length : 0 }
function profileIndicators(values: Partial<StudentInput>) {
  const academic = average([typeof values.cgpa === 'number' ? values.cgpa / 10 * 100 : undefined, typeof values.aptitude_score === 'number' ? values.aptitude_score : undefined])
  const experience = average([typeof values.internships === 'number' ? clamp(values.internships / 3 * 100) : undefined, typeof values.projects_count === 'number' ? clamp(values.projects_count / 6 * 100) : undefined, typeof values.certifications === 'number' ? clamp(values.certifications / 4 * 100) : undefined])
  const skills = average([typeof values.communication_skills === 'number' ? values.communication_skills / 10 * 100 : undefined, typeof values.extracurricular_activities === 'number' ? values.extracurricular_activities / 10 * 100 : undefined])
  const entered = Object.values(values).filter((value) => typeof value === 'number').length
  return { academic, experience, skills, overall: average([academic, experience, skills]), entered }
}

export function StudentProfilePreview({ values }: { values: Partial<StudentInput> }) {
  const indicators = profileIndicators(values)
  return <Card variant="solid" className="lg:sticky lg:top-7"><div className="border-b border-line p-5"><div className="flex items-start justify-between gap-3"><div><p className="eyebrow mb-2">Live profile review</p><h2 className="text-lg font-semibold text-ink">Student profile preview</h2></div><CircleHelp aria-hidden="true" className="h-4 w-4 text-cyan" /></div><p className="mt-2 text-xs leading-5 text-muted">These indicators summarize the entered values. They are not a placement prediction or model explanation.</p></div><div className="space-y-6 p-5"><PreviewMetric label="Academic strength" value={indicators.academic} icon={<BrainCircuit aria-hidden="true" className="h-4 w-4" />} detail="CGPA + aptitude" /><PreviewMetric label="Experience level" value={indicators.experience} icon={<BriefcaseBusiness aria-hidden="true" className="h-4 w-4" />} detail="Internships + projects + certifications" /><PreviewMetric label="Skill level" value={indicators.skills} icon={<Activity aria-hidden="true" className="h-4 w-4" />} detail="Communication + involvement" /><div className="border-t border-line pt-5"><div className="flex items-end justify-between"><div><p className="eyebrow text-[9px]">Overall profile indicator</p><p className="mono mt-2 text-3xl font-semibold text-ink">{Math.round(indicators.overall)}<span className="text-base text-meta">/100</span></p></div><span className="mono text-xs text-meta">{indicators.entered}/7 fields</span></div><div className="mt-4"><ProgressBar value={indicators.overall} showValue={false} /></div></div></div></Card>
}

function PreviewMetric({ label, value, detail, icon }: { label: string; value: number; detail: string; icon: React.ReactNode }) { return <div><div className="mb-3 flex items-center justify-between gap-3"><div className="flex items-center gap-2 text-sm font-semibold text-ink"><span className="text-cyan">{icon}</span>{label}</div><span className="mono text-xs text-ink">{Math.round(value)}%</span></div><ProgressBar value={value} showValue={false} /><p className="mt-2 text-[11px] text-meta">{detail}</p></div> }
