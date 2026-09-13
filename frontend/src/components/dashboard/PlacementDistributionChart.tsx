import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import type { DistributionItem } from '../../types/dashboard.ts'
import { formatCount } from '../../utils/format.ts'

const colors = ['#43ce96', '#f06d72']
export function PlacementDistributionChart({ data }: { data: DistributionItem[] }) {
  const total = data.reduce((sum, item) => sum + item.value, 0)
  return <div className="h-64 w-full"><ResponsiveContainer width="100%" height="100%"><PieChart>
    <Pie data={data} dataKey="value" nameKey="name" innerRadius={62} outerRadius={88} paddingAngle={3} stroke="none" isAnimationActive>
      {data.map((item, index) => <Cell key={item.name} fill={colors[index % colors.length]} />)}
    </Pie>
    <Tooltip contentStyle={{ background: '#172b3e', border: '1px solid #294257', borderRadius: 7, color: '#f2f7fb' }} formatter={(value) => [formatCount(Number(value || 0)), 'Predictions']} />
    <Legend iconType="circle" formatter={(value) => <span className="text-xs text-muted">{value}</span>} />
    <text x="50%" y="45%" textAnchor="middle" dominantBaseline="middle" fill="var(--ink)" fontSize="25" fontFamily="IBM Plex Mono">{total}</text>
    <text x="50%" y="55%" textAnchor="middle" dominantBaseline="middle" fill="var(--meta)" fontSize="10">saved</text>
  </PieChart></ResponsiveContainer></div>
}
