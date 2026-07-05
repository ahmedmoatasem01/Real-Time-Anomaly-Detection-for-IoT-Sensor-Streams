// pages/LiveMonitor.tsx — Real-time sensor data stream dashboard
import { useState, useCallback, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid
} from 'recharts'
import { AlertTriangle, CheckCircle2, Activity } from 'lucide-react'
import { api, type SensorReading } from '../lib/api'
import { useWebSocket } from '../lib/ws'

const MAX_POINTS = 120

function SeverityBadge({ s }: { s: string }) {
  const map: Record<string, string> = {
    high:   'bg-red-500/20 text-red-300 border-red-500/30',
    medium: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    low:    'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    none:   'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    critical: 'bg-red-700/30 text-red-200 border-red-700/40',
  }
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium uppercase tracking-wide ${map[s] ?? map.none}`}>
      {s}
    </span>
  )
}

function formatTime(ts: string) {
  try { return new Date(ts).toLocaleTimeString() } catch { return ts }
}

const globalReadings: SensorReading[] = []
const globalAlerts: SensorReading[] = []

export default function LiveMonitor() {
  const [readings, setReadings]   = useState<SensorReading[]>(globalReadings)
  const [alerts, setAlerts]       = useState<SensorReading[]>(globalAlerts)
  const { data: historical } = useQuery({
    queryKey: ['readings-init'],
    queryFn: () => api.readings(MAX_POINTS),
    staleTime: 0,
    refetchOnMount: 'always'
  })

  useEffect(() => {
    if (historical) {
      setReadings(historical.slice(-MAX_POINTS))
      setAlerts(historical.filter(r => r.is_anomaly).slice(-20))
    }
  }, [historical])

  const handleMsg = useCallback((data: SensorReading) => {
    setReadings(prev => {
      const next = [...prev, data]
      const final = next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next
      globalReadings.splice(0, globalReadings.length, ...final)
      return final
    })
    if (data.is_anomaly) {
      setAlerts(prev => {
        const final = [data, ...prev].slice(0, 30)
        globalAlerts.splice(0, globalAlerts.length, ...final)
        return final
      })
    }
  }, [])

  const { status } = useWebSocket<SensorReading>(handleMsg)

  const latest   = readings[readings.length - 1]
  const anomalyCount = readings.filter(r => r.is_anomaly).length
  const avgScore = readings.length
    ? (readings.reduce((s, r) => s + r.anomaly_score, 0) / readings.length).toFixed(4)
    : '—'

  const chartData = readings.map(r => ({
    t:     formatTime(r.timestamp),
    value: r.value,
    score: r.anomaly_score,
    anomaly: r.is_anomaly ? r.value : null,
  }))

  return (
    <div className="p-5 space-y-5 max-w-7xl mx-auto">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" /> Live Monitor
        </h1>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${
            status === 'connected' ? 'bg-emerald-400' :
            status === 'connecting' ? 'bg-amber-400 animate-pulse' : 'bg-red-400'
          }`} />
          <span className="text-slate-500 capitalize">{status}</span>
        </div>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Latest Value',   value: latest?.value?.toFixed(2) ?? '—',       unit: '°F', color: 'text-slate-100' },
          { label: 'Anomaly Score',  value: latest?.anomaly_score?.toFixed(4) ?? '—', unit: '',  color: latest?.is_anomaly ? 'text-red-400' : 'text-emerald-400' },
          { label: 'Anomalies / window', value: anomalyCount, unit: '',              color: anomalyCount > 0 ? 'text-amber-400' : 'text-slate-400' },
          { label: 'Avg Score',      value: avgScore,         unit: '',              color: 'text-slate-300' },
        ].map(({ label, value, unit, color }) => (
          <div key={label} className="rounded-lg border border-white/5 bg-white/2 p-4">
            <div className="text-xs text-slate-500 mb-1">{label}</div>
            <div className={`text-2xl font-semibold tabular-nums ${color}`}>
              {value}<span className="text-sm text-slate-500 ml-1">{unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Temperature chart */}
        <div className="rounded-lg border border-white/5 bg-white/2 p-4">
          <h3 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Temperature Stream</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#64748b' }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 9, fill: '#64748b' }} width={40} />
              <Tooltip
                contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.05)', fontSize: 11 }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Line dataKey="value" stroke="#22d3ee" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              {/* Anomaly markers */}
              {chartData.map((d, i) =>
                d.anomaly !== null ? (
                  <ReferenceLine key={i} x={d.t} stroke="#ef4444" strokeWidth={1} strokeOpacity={0.5} />
                ) : null
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Anomaly score chart */}
        <div className="rounded-lg border border-white/5 bg-white/2 p-4">
          <h3 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Anomaly Score</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#64748b' }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 9, fill: '#64748b' }} width={45} />
              <Tooltip
                contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.05)', fontSize: 11 }}
              />
              <Line dataKey="score" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Alert log */}
      <div className="rounded-lg border border-white/5 bg-white/2 p-4">
        <h3 className="text-xs text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> Recent Anomalies
        </h3>
        {alerts.length === 0 ? (
          <div className="flex items-center gap-2 text-xs text-emerald-400 py-2">
            <CheckCircle2 className="w-4 h-4" /> No anomalies in current window
          </div>
        ) : (
          <div className="space-y-1 max-h-52 overflow-y-auto">
            <AnimatePresence>
              {alerts.map((a, i) => (
                <motion.div
                  key={a.timestamp + i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center gap-3 text-xs py-1.5 px-2 rounded hover:bg-white/3"
                >
                  <SeverityBadge s={a.severity} />
                  <span className="text-slate-500 font-mono">{formatTime(a.timestamp)}</span>
                  <span className="text-slate-300 truncate">{a.reason || 'Anomaly detected'}</span>
                  <span className="ml-auto text-slate-500 tabular-nums">{a.anomaly_score.toFixed(4)}</span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  )
}
