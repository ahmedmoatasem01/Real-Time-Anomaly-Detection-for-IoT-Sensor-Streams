// pages/DemoPanel.tsx — Demo Control Panel with copy-pasteable commands
import { useState } from 'react'
import { Terminal, Copy, Check, PlayCircle, Zap, Ban } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'

function CodeBlock({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false)
  function copy() {
    navigator.clipboard.writeText(cmd).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000) })
  }
  return (
    <div className="relative group rounded-md bg-black/40 border border-white/5">
      <pre className="px-4 py-3 text-xs font-mono text-cyan-300 overflow-x-auto whitespace-pre-wrap">{cmd}</pre>
      <button
        onClick={copy}
        className="absolute top-2 right-2 p-1 rounded bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
      </button>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/2 p-5 space-y-3">
      <h2 className="text-xs text-slate-500 uppercase tracking-widest flex items-center gap-2">
        <PlayCircle className="w-3.5 h-3.5 text-cyan-400" /> {title}
      </h2>
      {children}
    </div>
  )
}

const STEPS = [
  { label: '1. Install deps',             cmd: 'pip install -r requirements.txt' },
  { label: '2. Download data + engineer features',
    cmd: 'python -m src.data.data_loader\npython -m src.data.preprocessing\npython -m src.features.feature_engineering' },
  { label: '3. Train all models',
    cmd: 'python -m src.models.train_baseline\npython -m src.models.train_isolation_forest\npython -m src.models.train_one_class_svm\npython -m src.models.train_lof\npython -m src.models.train_elliptic_envelope' },
  { label: '4. Evaluate (produces CSV + registry)',  cmd: 'python -m src.models.evaluate_all' },
  { label: '5. Start backend API',         cmd: 'uvicorn src.api.main:app --reload' },
  { label: '6. Start frontend dev server', cmd: 'cd frontend && npm run dev' },
  { label: '7. Run stream simulator (normal rate)',
    cmd: 'python -m src.streaming.stream_simulator --speed 50' },
  { label: '8. Jump to anomaly window (machine failure ~index 4700)',
    cmd: 'python -m src.streaming.stream_simulator --speed 30 --start-index 4700' },
]

const DOCKER_CMD = `docker compose up --build`

const FAULT_BUTTONS = [
  { type: 'spike_anomaly', label: 'Spike Anomaly', desc: 'Sudden huge value jump' },
  { type: 'gradual_drift', label: 'Gradual Drift', desc: 'Slowly increasing values' },
  { type: 'sensor_stuck', label: 'Sensor Stuck', desc: 'Value freezes entirely' },
  { type: 'missing_values', label: 'Missing Values', desc: 'Drops data packets (zeroes)' },
  { type: 'noise_burst', label: 'Noise Burst', desc: 'High variance static' },
  { type: 'overheating', label: 'Overheating', desc: 'Value rises and stays high' },
]

export default function DemoPanel() {
  const qc = useQueryClient()
  
  const { data: faultStatus } = useQuery({
    queryKey: ['fault-status'],
    queryFn: api.faultsStatus,
    refetchInterval: 2000,
  })

  const injectFault = useMutation({
    mutationFn: (type: string) => api.faultsInject({ fault_type: type, duration_steps: 100 }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['fault-status'] })
  })

  const stopFault = useMutation({
    mutationFn: () => api.faultsStop(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['fault-status'] })
  })

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-2">
        <Terminal className="w-5 h-5 text-cyan-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">Demo Control Panel</h1>
      </div>

      {/* Fault Injection UI */}
      <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-rose-300 uppercase tracking-widest flex items-center gap-2">
            <Zap className="w-4 h-4" /> Synthetic Fault Injection
          </h2>
          {faultStatus?.fault_type ? (
            <span className="px-2.5 py-1 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-mono animate-pulse">
              ACTIVE FAULT: {faultStatus.fault_type.toUpperCase()}
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono">
              STREAM NORMAL
            </span>
          )}
        </div>
        
        <p className="text-xs text-rose-200/60">
          Inject simulated hardware failures directly into the live data stream. 
          The machine learning models will detect these anomalies in real-time.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {FAULT_BUTTONS.map(f => (
            <button
              key={f.type}
              onClick={() => injectFault.mutate(f.type)}
              disabled={injectFault.isPending}
              className="flex flex-col items-start p-3 rounded bg-black/40 border border-rose-500/20 hover:border-rose-400/50 hover:bg-rose-500/10 transition-colors text-left"
            >
              <span className="text-sm font-medium text-rose-200">{f.label}</span>
              <span className="text-[10px] text-rose-400/60 mt-1">{f.desc}</span>
            </button>
          ))}
        </div>

        <div className="pt-2 border-t border-rose-500/10 flex justify-end">
          <button
            onClick={() => stopFault.mutate()}
            disabled={!faultStatus?.fault_type || stopFault.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-rose-500/20 text-rose-300 border border-rose-500/40 rounded hover:bg-rose-500/30 disabled:opacity-30 transition-colors text-xs font-medium uppercase tracking-widest"
          >
            <Ban className="w-3.5 h-3.5" /> Stop Active Fault
          </button>
        </div>
      </div>

      {/* Step-by-step commands */}
      <Section title="Step-by-Step Setup">
        <div className="space-y-4">
          {STEPS.map(({ label, cmd }) => (
            <div key={label} className="space-y-1">
              <p className="text-xs text-slate-400 font-medium">{label}</p>
              <CodeBlock cmd={cmd} />
            </div>
          ))}
        </div>
      </Section>

      {/* Docker alternative */}
      <Section title="Docker Alternative (one command)">
        <CodeBlock cmd={DOCKER_CMD} />
        <p className="text-xs text-slate-500">
          Starts both API (:8000) and frontend (:3000) via docker-compose.
        </p>
      </Section>

      {/* Simulator flags */}
      <Section title="Stream Simulator Flags">
        <div className="text-xs text-slate-400 space-y-2">
          <div className="flex gap-3">
            <code className="text-cyan-400 bg-cyan-900/20 px-1.5 py-0.5 rounded">--speed N</code>
            <span>Readings per second (default 10, max ~500)</span>
          </div>
          <div className="flex gap-3">
            <code className="text-cyan-400 bg-cyan-900/20 px-1.5 py-0.5 rounded">--start-index N</code>
            <span>Skip to index N in the test set (use ~4700 to jump to the failure window)</span>
          </div>
          <div className="flex gap-3">
            <code className="text-cyan-400 bg-cyan-900/20 px-1.5 py-0.5 rounded">--loop</code>
            <span>Loop the stream continuously (default: single pass)</span>
          </div>
        </div>
      </Section>

    </div>
  )
}
