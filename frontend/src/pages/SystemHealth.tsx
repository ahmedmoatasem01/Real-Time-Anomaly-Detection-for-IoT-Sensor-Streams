// pages/SystemHealth.tsx — Live system health dashboard
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { HeartPulse, Server, Database, Wifi, Cpu, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${ok ? 'text-emerald-400' : 'text-red-400'}`}>
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-red-400'}`} />
      {ok ? 'OK' : 'ERROR'}
    </span>
  )
}

function MetricRow({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-white/5 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <div className="text-right">
        <span className="text-sm font-mono text-slate-200">{value}</span>
        {sub && <span className="text-xs text-slate-600 ml-1">{sub}</span>}
      </div>
    </div>
  )
}

export default function SystemHealth() {
  const qc = useQueryClient()
  const retrain = useMutation({
    mutationFn: () => api.triggerRetrain(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['system-status'] }),
  })

  const { data, isLoading, dataUpdatedAt } = useQuery({
    queryKey: ['system-status'],
    queryFn:  api.systemStatus,
    refetchInterval: 5000,
  })

  const { data: driftStatus, isLoading: driftLoading } = useQuery({
    queryKey: ['drift-status'],
    queryFn: api.driftStatus,
    refetchInterval: 5000,
  })

  const checkDrift = useMutation({
    mutationFn: () => api.driftCheck(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['drift-status'] }),
  })

  const lastChecked = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—'

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-2">
        <HeartPulse className="w-5 h-5 text-emerald-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">System Health</h1>
        <span className="ml-auto text-xs text-slate-600">Auto-refresh every 5s · Last: {lastChecked}</span>
        <button 
          onClick={() => retrain.mutate()} 
          disabled={retrain.isPending} 
          className="ml-4 px-3 py-1 flex items-center gap-1.5 bg-indigo-500/20 text-indigo-300 border border-indigo-500/50 rounded hover:bg-indigo-500/30 text-xs transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${retrain.isPending ? 'animate-spin' : ''}`} />
          {retrain.isPending ? 'Triggering...' : 'Retrain Pipeline'}
        </button>
      </div>

      {/* Status cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'API',       icon: Server,   ok: data?.api === 'ok' },
          { label: 'Database',  icon: Database, ok: data?.database === 'ok' },
          { label: 'WebSocket', icon: Wifi,     ok: (data?.websocket_clients ?? 0) >= 0 },
          { label: 'Model',     icon: Cpu,      ok: data?.production_model !== 'none' },
        ].map(({ label, icon: Icon, ok }) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`rounded-xl border p-4 flex flex-col gap-3 ${
              ok ? 'border-emerald-500/20 bg-emerald-500/5' : 'border-red-500/20 bg-red-500/5'
            }`}
          >
            <div className="flex items-center justify-between">
              <Icon className={`w-4 h-4 ${ok ? 'text-emerald-400' : 'text-red-400'}`} />
              <StatusDot ok={ok} />
            </div>
            <span className="text-xs text-slate-400 uppercase tracking-widest">{label}</span>
          </motion.div>
        ))}
      </div>

      {/* Metrics */}
      <div className="rounded-xl border border-white/5 bg-white/2 p-5">
        <h2 className="text-xs text-slate-500 uppercase tracking-widest mb-1">Runtime Metrics</h2>
        {isLoading ? (
          <div className="space-y-2 mt-3">
            {Array(6).fill(0).map((_, i) => <div key={i} className="h-8 bg-white/3 rounded animate-pulse" />)}
          </div>
        ) : data ? (
          <>
            <MetricRow label="Production Model"   value={data.production_model} />
            <MetricRow label="Threshold"          value={data.threshold.toFixed(6)} />
            <MetricRow label="Total Readings"     value={data.total_readings.toLocaleString()} />
            <MetricRow label="Stream Rate"        value={data.stream_rate_rpm} sub="readings/min" />
            <MetricRow label="Avg Inference"      value={`${data.avg_inference_ms.toFixed(3)}`} sub="ms" />
            <MetricRow label="WS Clients"         value={data.websocket_clients} />
          </>
        ) : null}
      </div>

      {/* Drift Detection */}
      <div className="rounded-xl border border-white/5 bg-white/2 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xs text-slate-500 uppercase tracking-widest mb-1">Data Drift Status</h2>
          <button
            onClick={() => checkDrift.mutate()}
            disabled={checkDrift.isPending}
            className="px-3 py-1 flex items-center gap-1.5 bg-white/5 border border-white/10 rounded hover:bg-white/10 text-[10px] text-slate-400 uppercase tracking-widest transition-colors disabled:opacity-50"
          >
            Force Drift Check
          </button>
        </div>
        
        {driftLoading ? (
          <div className="h-8 bg-white/3 rounded animate-pulse" />
        ) : driftStatus ? (
          <div className={`p-4 rounded-lg border ${
            driftStatus.status === 'critical' ? 'border-red-500/30 bg-red-500/10' :
            driftStatus.status === 'warning' ? 'border-amber-500/30 bg-amber-500/10' :
            'border-emerald-500/30 bg-emerald-500/10'
          }`}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <span className="block text-[10px] text-slate-400 uppercase tracking-widest">Status</span>
                <span className={`text-sm font-semibold uppercase ${
                  driftStatus.status === 'critical' ? 'text-red-400' :
                  driftStatus.status === 'warning' ? 'text-amber-400' :
                  'text-emerald-400'
                }`}>{driftStatus.status}</span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-400 uppercase tracking-widest">Max PSI</span>
                <span className="text-sm font-mono text-slate-200">{driftStatus.psi.toFixed(4)}</span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-400 uppercase tracking-widest">Mean Shift (σ)</span>
                <span className="text-sm font-mono text-slate-200">{driftStatus.mean_shift_sigma.toFixed(2)}</span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-400 uppercase tracking-widest">Recommendation</span>
                <span className="text-sm font-medium text-slate-200">{driftStatus.recommendation}</span>
              </div>
            </div>
            {driftStatus.affected_features && driftStatus.affected_features.length > 0 && (
              <div className="mt-3 pt-3 border-t border-white/10">
                <span className="text-[10px] text-slate-400 uppercase tracking-widest mr-2">Affected Features:</span>
                <span className="text-xs text-slate-300 font-mono">{driftStatus.affected_features.join(', ')}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-sm text-slate-500">Drift status unavailable</div>
        )}
      </div>
    </div>
  )
}
