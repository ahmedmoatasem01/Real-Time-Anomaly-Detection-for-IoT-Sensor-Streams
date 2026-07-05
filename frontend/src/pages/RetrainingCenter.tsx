import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { RefreshCw, PlayCircle, Star, ArrowUpRight } from 'lucide-react'
import { api } from '../lib/api'

export default function RetrainingCenter() {
  const qc = useQueryClient()
  
  const { data: models = [], isLoading: modelsLoading } = useQuery({
    queryKey: ['model-registry'],
    queryFn: api.modelRegistry,
    refetchInterval: 5000,
  })

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: api.metrics,
    refetchInterval: 5000,
  })

  const { data: driftStatus } = useQuery({
    queryKey: ['drift-status'],
    queryFn: api.driftStatus,
    refetchInterval: 5000,
  })

  const retrain = useMutation({
    mutationFn: () => api.triggerRetrain(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['model-registry'] }),
  })

  const promote = useMutation({
    mutationFn: (name: string) => api.selectModel(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['model-registry'] })
      qc.invalidateQueries({ queryKey: ['system-status'] })
    },
  })

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center gap-2">
        <RefreshCw className="w-5 h-5 text-indigo-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">Retraining Center</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Retraining Pipeline Control */}
        <div className="md:col-span-1 rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-6 space-y-6">
          <div>
            <h2 className="text-sm font-medium text-indigo-300 uppercase tracking-widest">Automated Pipeline</h2>
            <p className="text-xs text-indigo-200/60 mt-2">
              Trigger a complete background re-training of all offline models using the latest telemetry data. 
              Candidate models will appear in the registry below for manual promotion.
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-black/20 p-3 rounded border border-white/5 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Total Readings Collected:</span>
                <span className="text-slate-300 font-mono">{metrics?.total_readings.toLocaleString() || '...'}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Data Drift Status:</span>
                <span className={`font-medium ${driftStatus?.status === 'critical' ? 'text-red-400' : driftStatus?.status === 'warning' ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {driftStatus?.status ? driftStatus.status.toUpperCase() : '...'}
                </span>
              </div>
            </div>

            <button 
              onClick={() => retrain.mutate()} 
              disabled={retrain.isPending} 
              className="w-full flex justify-center items-center gap-2 px-6 py-2.5 bg-indigo-600/80 text-indigo-100 rounded border border-indigo-500/50 hover:bg-indigo-500 transition-colors disabled:opacity-50 font-medium text-sm"
            >
              <PlayCircle className={`w-4 h-4 ${retrain.isPending ? 'animate-spin' : ''}`} />
              {retrain.isPending ? 'Pipeline Running...' : 'Start Retraining Job'}
            </button>
            
            {retrain.isSuccess && (
              <div className="text-center text-[10px] text-emerald-400 bg-emerald-500/10 py-1 rounded">
                Job started in background!
              </div>
            )}
          </div>
        </div>

        {/* Model Registry & Promotion */}
        <div className="md:col-span-2 rounded-xl border border-white/5 bg-white/2 p-6 space-y-4">
          <h2 className="text-sm font-medium text-slate-300 uppercase tracking-widest mb-4">Model Candidate Registry</h2>
          
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500 uppercase tracking-widest bg-white/5">
                <tr>
                  <th className="px-3 py-2 font-medium">Model Name</th>
                  <th className="px-3 py-2 font-medium">F1 Score</th>
                  <th className="px-3 py-2 font-medium">PR AUC</th>
                  <th className="px-3 py-2 font-medium">Created</th>
                  <th className="px-3 py-2 text-right font-medium">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {modelsLoading ? (
                  <tr><td colSpan={5} className="p-4 text-center text-slate-500">Loading registry...</td></tr>
                ) : models.length === 0 ? (
                  <tr><td colSpan={5} className="p-4 text-center text-slate-500">No models found</td></tr>
                ) : (
                  [...models].sort((a, b) => new Date(b.registered_at).getTime() - new Date(a.registered_at).getTime()).map(m => (
                    <tr key={m.name} className={`hover:bg-white/5 transition-colors ${m.is_production ? 'bg-emerald-500/5' : ''}`}>
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-slate-300">{m.name}</span>
                          {m.is_production && <span className="bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wider border border-emerald-500/30 flex items-center gap-1"><Star className="w-2.5 h-2.5" fill="currentColor"/> PROD</span>}
                        </div>
                        {m.type !== m.name && <div className="text-[10px] text-slate-500 mt-0.5">{m.type}</div>}
                      </td>
                      <td className="px-3 py-3 font-mono text-slate-400">
                        {m.test_metrics?.f1 ? m.test_metrics.f1.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-3 font-mono text-slate-400">
                        {m.test_metrics?.pr_auc ? m.test_metrics.pr_auc.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-3 text-slate-500">
                        {new Date(m.registered_at).toLocaleString()}
                      </td>
                      <td className="px-3 py-3 text-right">
                        {!m.is_production && m.name !== 'rolling_zscore' && (
                          <button
                            onClick={() => promote.mutate(m.name)}
                            disabled={promote.isPending}
                            className="inline-flex items-center gap-1 px-3 py-1 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded hover:bg-cyan-500/20 transition-colors text-[10px] uppercase tracking-widest disabled:opacity-50"
                          >
                            <ArrowUpRight className="w-3 h-3" /> Promote
                          </button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
