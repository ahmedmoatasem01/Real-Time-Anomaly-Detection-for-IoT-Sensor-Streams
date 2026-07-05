// pages/Overview.tsx — Platform Overview
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Activity, AlertTriangle, Zap, CheckCircle2, TrendingUp } from 'lucide-react'
import { api } from '../lib/api'

function StatCard({ label, value, unit = '', color = 'cyan', icon: Icon }: {
  label: string; value: string | number; unit?: string; color?: string; icon: React.ElementType
}) {
  const ring = {
    cyan:  'ring-cyan-500/20 bg-cyan-500/5',
    amber: 'ring-amber-500/20 bg-amber-500/5',
    red:   'ring-red-500/20 bg-red-500/5',
    green: 'ring-emerald-500/20 bg-emerald-500/5',
  }[color] ?? 'ring-cyan-500/20 bg-cyan-500/5'

  const iconColor = {
    cyan:  'text-cyan-400', amber: 'text-amber-400',
    red:   'text-red-400',  green: 'text-emerald-400',
  }[color] ?? 'text-cyan-400'

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-lg border border-white/5 p-5 ring-1 ${ring} flex flex-col gap-3`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500 uppercase tracking-widest">{label}</span>
        <Icon className={`w-4 h-4 ${iconColor}`} />
      </div>
      <motion.div
        className="text-3xl font-semibold text-slate-100 tabular-nums"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        {value}
        {unit && <span className="text-sm text-slate-500 ml-1">{unit}</span>}
      </motion.div>
    </motion.div>
  )
}

export default function Overview() {
  const { data: comparison } = useQuery({ queryKey: ['comparison'], queryFn: api.modelComparison })
  const { data: status }     = useQuery({ queryKey: ['system-status'], queryFn: api.systemStatus, refetchInterval: 8000 })

  const { data: assetSummary, isLoading: sLoading } = useQuery({ 
    queryKey: ['asset-summary'], 
    queryFn: async () => {
      const res = await fetch("http://localhost:8000/assets/summary");
      return res.json();
    }, 
    refetchInterval: 8000 
  })

  const top = comparison?.[0]

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-8">
      {/* Hero banner */}
      <div className="rounded-xl border border-white/5 bg-gradient-to-r from-cyan-900/20 via-slate-900/20 to-slate-800/20 px-8 py-6 flex items-center gap-6">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${status?.api === 'ok' ? 'bg-emerald-400' : 'bg-red-400'}`} />
            <span className="text-xs text-slate-400 uppercase tracking-widest">
              System {status?.api === 'ok' ? 'Operational' : 'Degraded'}
            </span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100">Multi-Modal Anomaly Detection Platform</h1>
          <p className="text-sm text-slate-400">
            Real-time insights across Time-Series, Vibration, and Vision assets.
          </p>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <CheckCircle2 className="w-8 h-8 text-emerald-400" />
        </div>
      </div>

      {/* Cross-Modal KPI cards */}
      {sLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array(4).fill(0).map((_, i) => (
            <div key={i} className="h-28 rounded-lg bg-white/3 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Monitored Assets"  value={assetSummary?.total_assets ?? '—'} icon={Activity}       color="cyan" />
          <StatCard label="Active Modalities" value={assetSummary?.active_modalities ?? '—'} icon={Zap}       color="green" />
          <StatCard label="Trained Models"    value={assetSummary?.total_models ?? '—'} icon={TrendingUp} color="amber" />
          <StatCard label="Total Anomalies"   value={assetSummary?.total_anomalies_detected ?? '—'} icon={AlertTriangle} color="red" />
        </div>
      )}

      {/* Top model + quick guide */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {top && (
          <div className="rounded-lg border border-white/5 bg-white/2 p-5 space-y-4">
            <h2 className="text-xs uppercase tracking-widest text-slate-500">Top Performing Model</h2>
            <div className="flex items-center gap-3">
              <span className="text-base font-semibold text-cyan-300">{top.name}</span>
              <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Production
              </span>
            </div>
            <div className="grid grid-cols-4 gap-3 text-center">
              {(['pr_auc','f1','precision','recall'] as const).map(k => (
                <div key={k} className="rounded-md bg-white/3 p-3">
                  <div className="text-lg font-bold text-slate-100">{(top[k] * 100).toFixed(1)}%</div>
                  <div className="text-xs text-slate-500">{k.replace(/_/g,' ').toUpperCase()}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="rounded-lg border border-white/5 bg-white/2 p-5 space-y-3">
          <h2 className="text-xs uppercase tracking-widest text-slate-500">Quick Start</h2>
          <ol className="space-y-2 text-sm text-slate-300">
            {[
              'Run: uvicorn src.api.main:app --reload',
              'Run: python -m src.streaming.stream_simulator --speed 50',
              'Navigate to Live Monitor to see data flow',
              'Visit Model Lab to compare all 5 trained models',
            ].map((step, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-cyan-500 font-mono shrink-0">{i + 1}.</span>
                <span className="text-slate-400 font-mono text-xs">{step}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  )
}
