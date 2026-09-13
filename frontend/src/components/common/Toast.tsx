import { CheckCircle2, Info, X, XCircle } from 'lucide-react'
import { useToasts } from '../../context/AppContext.tsx'

export function ToastRegion() {
  const { toasts, dismissToast } = useToasts()
  return <div className="fixed bottom-20 right-4 z-50 flex w-[min(360px,calc(100vw-2rem))] flex-col gap-2 md:bottom-5" aria-live="polite">
    {toasts.map((toast) => <div key={toast.id} className="panel-solid flex items-start gap-3 px-4 py-3 text-sm shadow-[0_16px_42px_rgba(0,0,0,.28)]">
      {toast.tone === 'success' ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-positive" /> : toast.tone === 'error' ? <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-negative" /> : <Info className="mt-0.5 h-4 w-4 shrink-0 text-cyan" />}
      <div className="min-w-0 flex-1"><p className="font-semibold text-ink">{toast.title}</p><p className="mt-1 leading-5 text-muted">{toast.message}</p></div>
      <button type="button" aria-label="Dismiss notification" className="text-meta hover:text-ink" onClick={() => dismissToast(toast.id)}><X aria-hidden="true" className="h-4 w-4" /></button>
    </div>)}
  </div>
}
