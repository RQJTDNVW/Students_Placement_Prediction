import { useState } from 'react'
import { FileDown, Printer, Save, ArrowLeft, ArrowRight } from 'lucide-react'
import { Navigate, Link, useLocation, useNavigate } from 'react-router-dom'
import { PageContainer } from '../components/layout/PageContainer.tsx'
import { Button } from '../components/common/Button.tsx'
import { Card } from '../components/common/Card.tsx'
import { StudentPredictionForm } from '../components/prediction/StudentPredictionForm.tsx'
import { StudentProfilePreview } from '../components/prediction/StudentProfilePreview.tsx'
import { PredictionResultCard } from '../components/prediction/PredictionResultCard.tsx'
import { RiskExplanation } from '../components/prediction/RiskExplanation.tsx'
import { ProfileIndicators } from '../components/prediction/ProfileIndicators.tsx'
import { ImprovementGuidance } from '../components/prediction/ImprovementGuidance.tsx'
import { StudentSummary } from '../components/prediction/StudentSummary.tsx'
import { usePrediction } from '../hooks/usePrediction.ts'
import { useSessionPrediction, usePreferences, useToasts } from '../context/AppContext.tsx'
import { savePrediction } from '../services/api.ts'
import type { PredictionRecord, StudentInput } from '../types/prediction.ts'

export default function PredictStudent() {
  const location = useLocation()
  const navigate = useNavigate()
  const [draftValues, setDraftValues] = useState<Partial<StudentInput>>({})
  const isResult = location.pathname.endsWith('/result')
  const { currentPrediction, setCurrentPrediction, markSaved } = useSessionPrediction()
  const { dataMode } = usePreferences()
  const { pushToast } = useToasts()
  const { submit, isLoading, error } = usePrediction()
  if (isResult && !currentPrediction) return <Navigate to="/predict" replace />
  const handleSubmit = async (values: StudentInput) => {
    const result = await submit(values)
    if (result) { setCurrentPrediction({ input: values, result, saved: false }); navigate('/predict/result') }
  }
  const handleSave = async () => {
    if (!currentPrediction || currentPrediction.saved) return
    const record: PredictionRecord = { ...currentPrediction.input, ...currentPrediction.result, prediction_id: currentPrediction.result.prediction_id || `PR-${Date.now()}`, timestamp: currentPrediction.result.timestamp || new Date().toISOString(), saved: true }
    try { if (dataMode === 'production') await savePrediction(record); markSaved(); pushToast({ tone: 'success', title: 'Prediction saved', message: `Prediction ${record.prediction_id} is now in history.` }) } catch { pushToast({ tone: 'error', title: 'Prediction could not be saved', message: 'The result is still available in the current session.' }) }
  }
  if (!isResult) return <PageContainer title="Predict student" description="Enter a student profile to request an estimate from the V4 technical placement model." context={dataMode === 'demo' ? 'Demo mode' : 'Production API'} actions={<Link to="/history" className="hidden text-xs font-semibold text-muted hover:text-cyan sm:inline-flex sm:items-center sm:gap-2"><ArrowLeft aria-hidden="true" className="h-3.5 w-3.5" />View history</Link>}><div className="grid items-start gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(300px,5fr)]"><StudentPredictionForm onSubmit={handleSubmit} onValuesChange={setDraftValues} isLoading={isLoading} error={error} /><StudentProfilePreview values={draftValues} /></div></PageContainer>
  const { input, result, saved } = currentPrediction!
  return <PageContainer title="Prediction result" description="A model estimate, profile indicators, and rule-based areas for improvement." context="Current session" actions={<Link to="/predict" className="inline-flex items-center gap-2 text-xs font-semibold text-muted hover:text-cyan"><ArrowLeft aria-hidden="true" className="h-3.5 w-3.5" />Predict another</Link>}><div className="space-y-3" id="prediction-result"><PredictionResultCard result={result} /><RiskExplanation result={result} /><div className="grid gap-3 lg:grid-cols-2"><ProfileIndicators input={input} /><Card variant="solid" className="p-5 sm:p-6"><div className="flex h-full flex-col justify-between"><div><p className="eyebrow mb-2">Returned scores</p><h2 className="text-lg font-semibold text-ink">Backend metadata</h2><p className="mt-2 text-sm leading-6 text-muted">These values were returned by the prediction service and are kept separate from the normalized profile indicators.</p></div><dl className="mt-8 grid grid-cols-2 gap-4 border-t border-line pt-5"><div><dt className="eyebrow text-[9px]">Experience score</dt><dd className="mono mt-2 text-xl text-ink">{result.experience_score ?? '—'}</dd></div><div><dt className="eyebrow text-[9px]">Skill score</dt><dd className="mono mt-2 text-xl text-ink">{result.skill_score ?? '—'}</dd></div></dl></div></Card></div><ImprovementGuidance input={input} result={result} /><StudentSummary input={input} result={result} /><Card variant="solid" className="flex flex-wrap items-center justify-between gap-3 p-4"><div><p className="eyebrow mb-1">Next action</p><p className="text-sm text-muted">Save this estimate or start a new student profile.</p></div><div className="flex flex-wrap gap-2"><Button variant={saved ? 'secondary' : 'primary'} onClick={() => void handleSave()} disabled={saved}><Save aria-hidden="true" className="h-4 w-4" />{saved ? 'Saved' : 'Save prediction'}</Button><Button variant="secondary" onClick={() => navigate('/predict')}><ArrowRight aria-hidden="true" className="h-4 w-4" />Predict another student</Button><Button variant="text" onClick={() => navigate('/history')}>View history</Button><Button variant="text" onClick={() => window.print()}><Printer aria-hidden="true" className="h-4 w-4" />Print</Button><Button variant="text" onClick={() => { window.print(); pushToast({ tone: 'info', title: 'Export PDF', message: 'Choose “Save as PDF” in the print dialog.' }) }}><FileDown aria-hidden="true" className="h-4 w-4" />Export PDF</Button></div></Card></div></PageContainer>
}
