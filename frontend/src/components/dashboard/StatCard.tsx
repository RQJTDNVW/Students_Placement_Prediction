import { motion } from 'framer-motion'
import type { ReactNode } from 'react'
import { Card } from '../common/Card.tsx'

export function StatCard({ label, value, detail, icon, tone = 'cyan' }: { label: string; value: string; detail: string; icon?: ReactNode; tone?: 'cyan' | 'positive' | 'moderate' | 'negative' }) {
  const tones = { cyan: 'text-cyan', positive: 'text-positive', moderate: 'text-moderate', negative: 'text-negative' }
  return <Card className="relative overflow-hidden p-4 sm:p-5"><div className={`absolute right-4 top-4 ${tones[tone]}`}>{icon}</div><p className="eyebrow pr-8 text-[9px]">{label}</p><motion.p initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mono mt-4 text-2xl font-semibold tracking-[-0.04em] text-ink sm:text-3xl">{value}</motion.p><p className="mt-2 text-xs leading-5 text-meta">{detail}</p></Card>
}
