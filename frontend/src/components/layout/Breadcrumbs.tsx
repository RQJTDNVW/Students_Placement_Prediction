import { ChevronRight } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'

const labels: Record<string, string> = { '/': 'Dashboard', '/predict': 'Predict student', '/predict/result': 'Prediction result', '/history': 'Prediction history', '/analytics': 'Student analytics', '/model-performance': 'Model performance', '/about': 'About project', '/settings': 'Settings' }

export function Breadcrumbs() {
  const { pathname } = useLocation()
  const label = labels[pathname] || 'Dashboard'
  return <div className="flex items-center gap-2 text-xs text-meta"><Link to="/" className="hover:text-cyan">SPI</Link><ChevronRight aria-hidden="true" className="h-3 w-3" /><span className="text-muted">{label}</span></div>
}
