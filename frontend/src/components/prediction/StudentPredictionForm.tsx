import { useState, useEffect } from 'react'
import { useForm, useWatch, type UseFormRegisterReturn } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { ArrowRight, ChevronDown, ChevronUp, Cpu, RotateCcw } from 'lucide-react'
import type { StudentInput } from '../../types/prediction.ts'
import { studentSchema } from '../../utils/validation.ts'
import { Button } from '../common/Button.tsx'
import { Card } from '../common/Card.tsx'
import { ErrorMessage } from '../common/ErrorMessage.tsx'
import { HelpTooltip } from '../common/Tooltip.tsx'

const initialValues: StudentInput = {
  cgpa: 7.5,
  aptitude_score: 70,
  internships: 1,
  projects_count: 2,
  certifications: 1,
  communication_skills: 6,
  extracurricular_activities: 5,
  coding_skills: 6.0,
  dsa_score: 5.5,
  backlogs: 0,
  college_tier: 'Tier-2',
  branch: 'CSE',
  hackathons: 1,
  open_source_contributions: 0,
}

type FieldName = keyof StudentInput
interface FieldProps {
  label: string
  description: string
  name: FieldName
  register: UseFormRegisterReturn
  error?: string
  min?: number
  max?: number
  step?: number
  tooltip: string
}

function Field({ label, description, name, register, error, min, max, step = 1, tooltip }: FieldProps) {
  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div>
          <label htmlFor={name} className="text-sm font-semibold text-ink">{label}</label>
          <p className="mt-1 text-xs leading-5 text-meta">{description}</p>
        </div>
        <HelpTooltip label={tooltip} />
      </div>
      <input
        id={name}
        type="number"
        inputMode={step < 1 ? 'decimal' : 'numeric'}
        min={min}
        max={max}
        step={step}
        aria-invalid={Boolean(error)}
        aria-describedby={`${name}-hint ${name}-error`}
        className={`mono h-10 w-full rounded-[7px] border bg-recessed px-3 text-sm text-ink outline-none transition-colors placeholder:text-meta focus:border-cyan focus:ring-2 focus:ring-cyan/20 ${error ? 'border-negative' : 'border-line'}`}
        {...register}
      />
      <div id={`${name}-hint`} className="mt-2 flex justify-between text-[10px] text-meta">
        <span>{min !== undefined ? `Min ${min}` : 'No minimum'}</span>
        <span>{max !== undefined ? `Max ${max}` : 'Whole number'}</span>
      </div>
      <p id={`${name}-error`} className="mt-1 min-h-4 text-xs text-negative">{error || ' '}</p>
    </div>
  )
}

export function StudentPredictionForm({
  onSubmit,
  onValuesChange,
  isLoading,
  error,
}: {
  onSubmit: (values: StudentInput) => Promise<void>
  onValuesChange?: (values: Partial<StudentInput>) => void
  isLoading: boolean
  error: string | null
}) {
  const [showTechFields, setShowTechFields] = useState(false)
  const { register, handleSubmit, reset, control, formState: { errors, isDirty } } = useForm<StudentInput>({
    resolver: zodResolver(studentSchema),
    defaultValues: initialValues,
    mode: 'onBlur',
  })
  const values = useWatch({ control })
  const numberRegister = (name: FieldName) =>
    register(name, { setValueAs: (value: string) => (value === '' ? undefined : Number(value)) })

  useEffect(() => {
    onValuesChange?.(values)
  }, [onValuesChange, values])

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate className="space-y-3">
      {error && <ErrorMessage message={error} />}
      <Card className="overflow-hidden">
        <Section title="Academic information" index="01" description="Foundational indicators used in the student profile.">
          <div className="grid gap-x-5 sm:grid-cols-2">
            <Field label="CGPA" description="Cumulative grade point average." name="cgpa" register={numberRegister('cgpa')} error={errors.cgpa?.message} min={0} max={10} step={0.1} tooltip="CGPA gives the model a normalized view of academic performance." />
            <Field label="Aptitude score" description="Placement aptitude assessment score." name="aptitude_score" register={numberRegister('aptitude_score')} error={errors.aptitude_score?.message} min={0} max={100} tooltip="Aptitude performance can indicate readiness for screening assessments." />
          </div>
        </Section>

        <Section title="Experience" index="02" description="Practical exposure and evidence of applied learning.">
          <div className="grid gap-x-5 sm:grid-cols-3">
            <Field label="Internships" description="Completed internships." name="internships" register={numberRegister('internships')} error={errors.internships?.message} min={0} tooltip="Internships represent exposure to real work environments." />
            <Field label="Projects count" description="Relevant practical projects." name="projects_count" register={numberRegister('projects_count')} error={errors.projects_count?.message} min={0} tooltip="Projects show how a student applies knowledge to concrete problems." />
            <Field label="Certifications" description="Relevant certifications." name="certifications" register={numberRegister('certifications')} error={errors.certifications?.message} min={0} tooltip="Certifications can signal focused preparation for a role or skill." />
          </div>
        </Section>

        <Section title="Skills" index="03" description="Communication and wider participation indicators.">
          <div className="grid gap-x-5 sm:grid-cols-2">
            <Field label="Communication skills" description="Self-assessed score from 0 to 10." name="communication_skills" register={numberRegister('communication_skills')} error={errors.communication_skills?.message} min={0} max={10} step={0.1} tooltip="Communication skills can support interviews, teamwork, and clear explanations." />
            <Field label="Extracurricular activities" description="Participation score from 0 to 10." name="extracurricular_activities" register={numberRegister('extracurricular_activities')} error={errors.extracurricular_activities?.message} min={0} max={10} step={0.1} tooltip="Activities can provide evidence of collaboration, initiative, or leadership." />
          </div>
        </Section>

        {/* Optional Technical Section */}
        <div className="border-b border-line bg-recessed/30 p-4 sm:p-5">
          <button
            type="button"
            onClick={() => setShowTechFields(!showTechFields)}
            className="flex w-full items-center justify-between text-left text-xs font-semibold text-ink transition-colors hover:text-cyan"
          >
            <span className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan" />
              Technical & Domain Profile (Advanced V4 Model)
            </span>
            <span className="flex items-center gap-1 text-[11px] text-muted">
              {showTechFields ? 'Hide technical fields' : 'Expand technical fields'}
              {showTechFields ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </span>
          </button>
        </div>

        {showTechFields && (
          <Section title="Technical & Domain Mastery" index="04" description="Specific technical metrics for software engineering and IT placements.">
            <div className="grid gap-x-5 sm:grid-cols-3">
              <Field label="Coding skills" description="Programming proficiency (1 to 10)." name="coding_skills" register={numberRegister('coding_skills')} min={1} max={10} step={0.1} tooltip="Proficiency in core language syntax and implementation." />
              <Field label="DSA score" description="Data Structures & Algorithms (1 to 10)." name="dsa_score" register={numberRegister('dsa_score')} min={1} max={10} step={0.1} tooltip="Problem-solving readiness for technical interviews." />
              <Field label="Active backlogs" description="Pending course backlogs." name="backlogs" register={numberRegister('backlogs')} min={0} max={10} tooltip="Active backlogs can affect eligibility for placement drives." />
            </div>
            <div className="mt-4 grid gap-x-5 sm:grid-cols-2">
              <Field label="Hackathons" description="Participated hackathons." name="hackathons" register={numberRegister('hackathons')} min={0} tooltip="Demonstrates rapid prototyping and teamwork." />
              <Field label="Open source" description="Contributions to OSS." name="open_source_contributions" register={numberRegister('open_source_contributions')} min={0} tooltip="Real-world collaborative code contributions." />
            </div>
          </Section>
        )}
      </Card>

      <div className="flex flex-wrap items-center justify-end gap-3 border-t border-line pt-5">
        <Button variant="text" onClick={() => reset(initialValues)} disabled={isLoading || !isDirty}>
          <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
          Reset form
        </Button>
        <Button variant="primary" type="submit" loading={isLoading}>
          {isLoading ? 'Requesting prediction…' : 'Predict placement'}
          {!isLoading && <ArrowRight aria-hidden="true" className="h-4 w-4" />}
        </Button>
      </div>
    </form>
  )
}

function Section({ title, index, description, children }: { title: string; index: string; description: string; children: React.ReactNode }) {
  return (
    <fieldset className="border-b border-line p-5 last:border-b-0 sm:p-6">
      <legend className="sr-only">{title}</legend>
      <div className="mb-6 flex gap-3">
        <span className="mono text-xs text-cyan">{index}</span>
        <div>
          <h2 className="text-base font-semibold text-ink">{title}</h2>
          <p className="mt-1 text-xs leading-5 text-muted">{description}</p>
        </div>
      </div>
      {children}
    </fieldset>
  )
}

