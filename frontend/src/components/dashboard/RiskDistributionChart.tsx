import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { RiskLevel } from '../../types/prediction.ts'

const colors: Record<RiskLevel, string> = { 'Low Risk': '#43ce96', 'Moderate Risk': '#f0ae4e', 'High Risk': '#f06d72', 'Not Available': '#8297a8' }
export function RiskDistributionChart({ data }: { data: Array<{ name: RiskLevel; value: number }> }) {
  return <div className="h-64 w-full"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} layout="vertical" margin={{ top: 8, right: 18, left: 10, bottom: 0 }}>
    <XAxis type="number" allowDecimals={false} tick={{ fill: 'var(--meta)', fontSize: 10 }} tickLine={false} axisLine={false} />
    <YAxis type="category" dataKey="name" width={88} tick={{ fill: 'var(--muted)', fontSize: 10 }} tickLine={false} axisLine={false} />
    <Tooltip cursor={{ fill: 'rgba(69,207,245,.06)' }} contentStyle={{ background: '#172b3e', border: '1px solid #294257', borderRadius: 7 }} formatter={(value) => [Number(value || 0), 'Predictions']} />
    <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={22}>{data.map((item) => <Cell key={item.name} fill={colors[item.name]} />)}</Bar>
  </BarChart></ResponsiveContainer></div>
}
