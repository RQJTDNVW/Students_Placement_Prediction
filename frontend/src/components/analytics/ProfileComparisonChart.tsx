import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { AnalyticsData } from '../../types/dashboard.ts'
import { Card, SectionHeader } from '../common/Card.tsx'

export function ProfileComparisonChart({ data }: { data: AnalyticsData['profile_comparison'] }) {
  return <Card className="p-4 sm:p-5"><SectionHeader eyebrow="Profile review" title="Student profile score comparison" description="Normalized indicator averages across the saved prediction history." /><div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} layout="vertical" margin={{ left: 8, right: 22 }}><CartesianGrid horizontal={false} stroke="var(--line)" strokeDasharray="2 5" /><XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fill: 'var(--meta)', fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="label" width={130} tick={{ fill: 'var(--muted)', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: '#172b3e', border: '1px solid #294257', borderRadius: 7 }} formatter={(value) => [`${Math.round(Number(value || 0))}%`, 'Indicator average']} /><Bar dataKey="value" fill="#45cff5" radius={[0, 3, 3, 0]} barSize={20} /></BarChart></ResponsiveContainer></div></Card>
}
