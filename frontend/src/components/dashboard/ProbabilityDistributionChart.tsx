import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ProbabilityBin } from '../../types/dashboard.ts'

export function ProbabilityDistributionChart({ data }: { data: ProbabilityBin[] }) {
  return <div className="h-64 w-full"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 8, right: 10, left: -18, bottom: 0 }}>
    <CartesianGrid vertical={false} stroke="var(--line)" strokeDasharray="2 5" />
    <XAxis dataKey="range" tick={{ fill: 'var(--meta)', fontSize: 10 }} tickLine={false} axisLine={false} />
    <YAxis allowDecimals={false} tick={{ fill: 'var(--meta)', fontSize: 10 }} tickLine={false} axisLine={false} />
    <Tooltip cursor={{ fill: 'rgba(69,207,245,.06)' }} contentStyle={{ background: '#172b3e', border: '1px solid #294257', borderRadius: 7 }} labelStyle={{ color: '#b5c5d2' }} formatter={(value) => [Number(value || 0), 'Predictions']} />
    <Bar dataKey="count" fill="#45cff5" radius={[3, 3, 0, 0]} maxBarSize={48} />
  </BarChart></ResponsiveContainer></div>
}
