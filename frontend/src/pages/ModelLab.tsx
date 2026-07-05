// pages/ModelLab.tsx — Model comparison, ranking, promote-to-production
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { Trophy, FlaskConical, ArrowUpCircle } from 'lucide-react'
import { api, type ModelComparison } from '../lib/api'

const METRIC_LABELS: Record<string, string> = {
  pr_auc:            'PR-AUC',
  f1:                'F1',
  precision:         'Precision',
  recall:            'Recall',
  false_alarm_rate:  'False Alarm Rate',
  roc_auc:           'ROC-AUC',
}

function ModelCard({ model, onPromote, promoting, isActive }: {
  model: ModelComparison
  onPromote: (name: string) => void
  promoting: boolean
  isActive: boolean
}) {
  const isTop  = model.rank === 1
  const barColor = isTop ? '#22d3ee' : '#64748b'

  const registryNameMap: Record<string, string> = {
    'Isolation Forest': 'isolation_forest',
    'One-Class SVM': 'one_class_svm',
    'LOF': 'lof',
    'Elliptic Envelope': 'elliptic_envelope',
    'Rolling Z-score (Baseline)': 'rolling_zscore',
    'River HST (Online)': 'river_hst',
    'LSTM Autoencoder': 'lstm_autoencoder'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: model.rank * 0.05 }}
      className={`rounded-xl border p-5 space-y-4 ${
        isActive ? 'border-cyan-500/30 bg-cyan-500/5' : 'border-white/5 bg-white/2'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          {isActive && <Trophy className="w-4 h-4 text-amber-400" />}
          <h3 className="font-semibold text-slate-100 text-sm">{model.name}</h3>
        </div>
        <div className="flex items-center gap-1.5">
          {isActive && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/20 uppercase tracking-wide">
              Production
            </span>
          )}
          <span className="text-[10px] text-slate-600 border border-white/5 px-1.5 py-0.5 rounded">
            #{model.rank}
          </span>
        </div>
      </div>

      {/* Metric mini-table */}
      <div className="grid grid-cols-3 gap-2 text-center">
        {(['pr_auc','f1','false_alarm_rate'] as const).map(k => (
          <div key={k} className="bg-white/3 rounded-md py-2">
            <div className="text-base font-bold text-slate-100 tabular-nums">
              {(model[k] * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-slate-500">{METRIC_LABELS[k]}</div>
          </div>
        ))}
      </div>

      {/* Metric bar */}
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={['precision','recall','f1','pr_auc','roc_auc'].map(k => ({
              m: k.toUpperCase().replace('_','-'),
              v: +(model[k as keyof ModelComparison] as number * 100).toFixed(1),
            }))}
            margin={{ top: 2, right: 4, bottom: 2, left: -20 }}
          >
            <CartesianGrid stroke="#1e2a3a" strokeDasharray="2 2" />
            <XAxis dataKey="m" tick={{ fontSize: 8, fill: '#64748b' }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 8, fill: '#64748b' }} />
            <Tooltip
              contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.05)', fontSize: 10 }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(v: any) => [`${Number(v).toFixed(1)}%`]}
            />
            <Bar dataKey="v" fill={barColor} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center gap-3 text-xs text-slate-500">
        <span>{model.avg_inference_ms.toFixed(3)} ms</span>
        <span>·</span>
        <span>{model.detection_latency_steps.toFixed(0)} steps latency</span>
      </div>

      {!isActive && (
        <button
          onClick={() => onPromote(registryNameMap[model.name] || model.name.toLowerCase().replace(/\s+/g,'_'))}
          disabled={promoting}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-md text-xs border border-slate-600 text-slate-400 hover:border-cyan-500/40 hover:text-cyan-300 disabled:opacity-40 transition-colors"
        >
          <ArrowUpCircle className="w-3.5 h-3.5" />
          Promote to production
        </button>
      )}
    </motion.div>
  )
}

export default function ModelLab() {
  const qc = useQueryClient()
  const [toast, setToast] = useState<string | null>(null)

  const { data: models, isLoading } = useQuery({
    queryKey: ['comparison'],
    queryFn: api.modelComparison,
  })

  const { data: sysStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: api.systemStatus,
    refetchInterval: 5000
  })

  const activeModel = sysStatus?.production_model || 'isolation_forest'

  const promote = useMutation({
    mutationFn: (name: string) => api.selectModel(name),
    onSuccess: (_, name) => {
      setToast(`Promoted '${name}' to production`)
      qc.invalidateQueries({ queryKey: ['comparison'] })
      setTimeout(() => setToast(null), 3500)
    },
    onError: (e: Error) => {
      setToast(`Error: ${e.message}`)
      setTimeout(() => setToast(null), 4000)
    },
  })

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Toast */}
      {toast && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="fixed top-4 right-4 z-50 px-4 py-2 rounded-lg bg-cyan-900/80 border border-cyan-500/30 text-sm text-cyan-200 shadow-xl"
        >
          {toast}
        </motion.div>
      )}

      <div className="flex items-center gap-2">
        <FlaskConical className="w-5 h-5 text-cyan-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">Model Lab</h1>
        <span className="ml-auto text-xs text-slate-600">Ranked by PR-AUC · test split</span>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {[1,2,3].map(i => <div key={i} className="h-64 rounded-xl bg-white/3 animate-pulse" />)}
        </div>
      ) : models?.length ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {models.map(m => {
            const registryMap: Record<string, string> = {
              'Isolation Forest': 'isolation_forest',
              'One-Class SVM': 'one_class_svm',
              'LOF': 'lof',
              'Elliptic Envelope': 'elliptic_envelope',
              'Rolling Z-score (Baseline)': 'rolling_zscore',
              'River HST (Online)': 'river_hst',
              'LSTM Autoencoder': 'lstm_autoencoder'
            }
            return (
              <ModelCard
                key={m.name}
                model={m}
                onPromote={promote.mutate}
                promoting={promote.isPending}
                isActive={activeModel === (registryMap[m.name] || m.name.toLowerCase().replace(/\s+/g,'_'))}
              />
            )
          })}
        </div>
      ) : (
        <div className="text-center py-16 text-slate-500 text-sm">
          No models found. Run <code className="text-cyan-400">python -m src.models.evaluate_all</code> first.
        </div>
      )}

      {/* Comparison bar chart */}
      {models?.length && (
        <div className="rounded-xl border border-white/5 bg-white/2 p-5">
          <h2 className="text-xs text-slate-500 uppercase tracking-widest mb-4">PR-AUC Comparison</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              data={models.map(m => ({ name: m.name.replace(' (Baseline)','').replace(' (MVP)',''), pr_auc: +(m.pr_auc*100).toFixed(1) }))}
            >
              <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} />
              <Tooltip
                contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.05)', fontSize: 11 }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(v: any) => [`${Number(v).toFixed(1)}%`, 'PR-AUC']}
              />
              <Bar dataKey="pr_auc" fill="#22d3ee" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
