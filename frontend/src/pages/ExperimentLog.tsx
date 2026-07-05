// pages/ExperimentLog.tsx — Experiment history table with export
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Beaker, Download } from 'lucide-react'
import { api } from '../lib/api'

function pct(v: unknown) {
  const n = parseFloat(String(v))
  return isNaN(n) ? '—' : `${(n * 100).toFixed(1)}%`
}

export default function ExperimentLog() {
  const { data: experiments = [], isLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: api.experiments,
  })

  function exportCSV() {
    if (!experiments.length) return
    const keys = Object.keys(experiments[0])
    const csv  = [keys.join(','), ...experiments.map(e =>
      keys.map(k => JSON.stringify(e[k] ?? '')).join(',')
    )].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = Object.assign(document.createElement('a'), { href: url, download: 'experiments.csv' })
    document.body.appendChild(a); a.click(); a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-5">
      <div className="flex items-center gap-2">
        <Beaker className="w-5 h-5 text-cyan-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">Experiment Results</h1>
        <button
          onClick={exportCSV}
          className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1.5 rounded border border-white/10 text-slate-400 hover:border-cyan-500/30 hover:text-cyan-300 transition-colors"
        >
          <Download className="w-3.5 h-3.5" /> Export CSV
        </button>
      </div>

      <div className="rounded-xl border border-white/5 overflow-x-auto">
        <table className="w-full text-xs min-w-[800px]">
          <thead className="bg-white/3 text-slate-500 uppercase tracking-widest">
            <tr>
              <th className="px-4 py-3 text-left">#</th>
              <th className="px-4 py-3 text-left">Model</th>
              <th className="px-4 py-3 text-left">Split</th>
              <th className="px-4 py-3 text-left">Threshold</th>
              <th className="px-4 py-3 text-left">PR-AUC</th>
              <th className="px-4 py-3 text-left">F1</th>
              <th className="px-4 py-3 text-left">Precision</th>
              <th className="px-4 py-3 text-left">Recall</th>
              <th className="px-4 py-3 text-left">FAR</th>
              <th className="px-4 py-3 text-left">Date</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array(5).fill(0).map((_, i) => (
                <tr key={i} className="border-t border-white/5">
                  <td colSpan={10} className="px-4 py-3">
                    <div className="h-4 bg-white/5 rounded animate-pulse" />
                  </td>
                </tr>
              ))
            ) : experiments.length === 0 ? (
              <tr className="border-t border-white/5">
                <td colSpan={10} className="px-4 py-8 text-center text-slate-600">
                  No experiments yet. Run evaluate_all.py to populate.
                </td>
              </tr>
            ) : (
              experiments.map((e, i) => (
                <motion.tr
                  key={e.id ?? i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.02 }}
                  className="border-t border-white/5 hover:bg-white/3 transition-colors"
                >
                  <td className="px-4 py-2.5 text-slate-600">{e.id}</td>
                  <td className="px-4 py-2.5 text-cyan-300 font-medium">{String(e.model)}</td>
                  <td className="px-4 py-2.5 text-slate-500">{String(e.split)}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-400 tabular-nums">
                    {parseFloat(String(e.threshold)).toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-emerald-400 tabular-nums">{pct(e['PR-AUC'])}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{pct(e['F1'])}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{pct(e['Precision (Point)'])}</td>
                  <td className="px-4 py-2.5 font-mono tabular-nums">{pct(e['Recall (Point)'])}</td>
                  <td className="px-4 py-2.5 font-mono text-red-400 tabular-nums">{pct(e['False Alarm Rate'])}</td>
                  <td className="px-4 py-2.5 text-slate-500 font-mono">
                    {e.created_at ? new Date(String(e.created_at)).toLocaleDateString() : '—'}
                  </td>
                </motion.tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
