// pages/DataExplorer.tsx — Dataset metadata explorer
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Database } from 'lucide-react'
import { api } from '../lib/api'

export default function DataExplorer() {
  const { data, isLoading } = useQuery({ queryKey: ['data-summary'], queryFn: api.dataSummary })

  const splitData = data?.splits
    ? Object.entries(data.splits).map(([k, v]) => ({ name: k, count: v }))
    : []

  const labelData = data?.label_counts
    ? [
        { name: 'Normal',  value: data.label_counts.normal,  fill: '#22d3ee' },
        { name: 'Anomaly', value: data.label_counts.anomaly, fill: '#ef4444' },
      ]
    : []

  if (isLoading) {
    return (
      <div className="p-6 space-y-4">
        {Array(4).fill(0).map((_, i) => (
          <div key={i} className="h-20 rounded-lg bg-white/3 animate-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-2">
        <Database className="w-5 h-5 text-cyan-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">Data Explorer</h1>
      </div>

      {/* Dataset info */}
      <div className="rounded-xl border border-white/5 bg-white/2 p-5 space-y-4">
        <h2 className="text-xs text-slate-500 uppercase tracking-widest">Dataset</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Name',         value: 'NAB machine_temperature' },
            { label: 'Total rows',   value: data?.total_rows?.toLocaleString() ?? '—' },
            { label: 'Features',     value: data?.feature_count ?? '—' },
            { label: 'Anomaly rate', value: `${data?.anomaly_rate_pct ?? '—'}%` },
          ].map(({ label, value }) => (
            <div key={label} className="rounded-md bg-white/3 p-3">
              <div className="text-xs text-slate-500 mb-1">{label}</div>
              <div className="text-sm font-semibold text-slate-100">{value}</div>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-500">Source:</span>
          <code className="text-cyan-400 bg-cyan-900/20 px-1 py-0.5 rounded">{data?.source}</code>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Split bar chart */}
        <div className="rounded-xl border border-white/5 bg-white/2 p-5">
          <h2 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Split Distribution</h2>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={splitData}>
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
              <Tooltip
                contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.05)', fontSize: 11 }}
              />
              <Bar dataKey="count" fill="#22d3ee" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Label pie */}
        <div className="rounded-xl border border-white/5 bg-white/2 p-5">
          <h2 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Label Balance</h2>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={labelData} dataKey="value" nameKey="name" outerRadius={65} label={({ name, percent }: { name?: string; percent?: number }) => `${name ?? ''} ${((percent ?? 0) * 100).toFixed(1)}%`} labelLine={false}>
                {labelData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.05)', fontSize: 11 }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature list */}
      {data?.feature_columns && (
        <div className="rounded-xl border border-white/5 bg-white/2 p-5">
          <h2 className="text-xs text-slate-500 uppercase tracking-widest mb-3">
            Feature Contract ({data.feature_columns.length} columns)
          </h2>
          <div className="flex flex-wrap gap-2">
            {data.feature_columns.map((f) => (
              <motion.span
                key={f}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-xs font-mono px-2 py-1 rounded bg-white/3 border border-white/5 text-slate-400"
              >
                {f}
              </motion.span>
            ))}
          </div>
        </div>
      )}

      {/* Preprocessing notes */}
      <div className="rounded-xl border border-white/5 bg-white/2 p-5 space-y-2">
        <h2 className="text-xs text-slate-500 uppercase tracking-widest">Preprocessing Steps</h2>
        <ul className="space-y-1 text-xs text-slate-400">
          {[
            'Parse timestamp, drop duplicates, sort chronologically',
            'Fill missing values via linear interpolation (flag imputed rows)',
            'Chronological 70/15/15 train/val/test split (no leakage)',
            'StandardScaler fit on train set only — applied to val/test',
            '22 causal features: lags, rolling stats, EWMA, z-scores for windows 5/15/60',
            'feature_columns.json frozen as canonical feature contract',
          ].map((step, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-cyan-600 shrink-0">{i + 1}.</span>
              <span>{step}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
