// pages/AlertCenter.tsx — Sortable, filterable alert table with ack
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { Bell, CheckCheck, Filter, ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '../lib/api'

type SortKey = 'ts' | 'severity' | 'score'

const SEVERITY_ORDER: Record<string, number> = { critical: 4, high: 3, medium: 2, low: 1, none: 0 }

function SeverityBadge({ s }: { s: string }) {
  const map: Record<string, string> = {
    high:     'bg-red-500/20 text-red-300 border-red-500/30',
    critical: 'bg-red-700/30 text-red-200 border-red-700/40',
    medium:   'bg-amber-500/20 text-amber-300 border-amber-500/30',
    low:      'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    none:     'bg-slate-500/10 text-slate-400 border-slate-500/20',
  }
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium uppercase tracking-wide ${map[s] ?? map.none}`}>
      {s}
    </span>
  )
}

export default function AlertCenter() {
  const qc = useQueryClient()
  const [filterSev, setFilterSev] = useState<string>('all')
  const [filterAck, setFilterAck] = useState<'all' | 'unacked'>('unacked')
  const [sortKey, setSortKey]     = useState<SortKey>('ts')
  const [sortAsc, setSortAsc]     = useState(false)

  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => api.alerts(200),
    refetchInterval: 5000,
  })

  const ack = useMutation({
    mutationFn: (id: number) => api.ackAlert(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
  })

  const updateStatus = useMutation({
    mutationFn: ({ id, status, operator_note }: { id: number, status: string, operator_note?: string }) => api.updateAlertStatus(id, { status, operator_note }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
  })

  const updateFeedback = useMutation({
    mutationFn: ({ id, feedback }: { id: number, feedback: string }) => api.updateAlertFeedback(id, { feedback }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
  })

  const sorted = [...alerts]
    .filter(a => filterSev === 'all' || a.severity === filterSev)
    .filter(a => filterAck === 'all' || !a.acknowledged)
    .sort((a, b) => {
      let cmp = 0
      if (sortKey === 'ts')       cmp = a.ts.localeCompare(b.ts)
      if (sortKey === 'severity') cmp = (SEVERITY_ORDER[a.severity] ?? 0) - (SEVERITY_ORDER[b.severity] ?? 0)
      if (sortKey === 'score')    cmp = a.score - b.score
      return sortAsc ? cmp : -cmp
    })

  function toggleSort(k: SortKey) {
    if (sortKey === k) setSortAsc(!sortAsc)
    else { setSortKey(k); setSortAsc(false) }
  }

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k
      ? (sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)
      : null

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-5">
      <div className="flex items-center gap-2">
        <Bell className="w-5 h-5 text-amber-400" />
        <h1 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">Alert Center</h1>
        <span className="ml-2 text-xs text-slate-600">{sorted.length} alerts</span>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Filter className="w-3.5 h-3.5" /> Filter:
        </div>
        <select
          value={filterSev}
          onChange={e => setFilterSev(e.target.value)}
          className="text-xs bg-white/5 border border-white/10 rounded px-2 py-1 text-slate-300 focus:outline-none"
        >
          <option value="all" className="bg-slate-900 text-slate-300">All Severities</option>
          <option value="critical" className="bg-slate-900 text-slate-300">Critical</option>
          <option value="high" className="bg-slate-900 text-slate-300">High</option>
          <option value="medium" className="bg-slate-900 text-slate-300">Medium</option>
          <option value="low" className="bg-slate-900 text-slate-300">Low</option>
        </select>
        <select
          value={filterAck}
          onChange={e => setFilterAck(e.target.value as 'all' | 'unacked')}
          className="text-xs bg-white/5 border border-white/10 rounded px-2 py-1 text-slate-300 focus:outline-none"
        >
          <option value="unacked" className="bg-slate-900 text-slate-300">Unacknowledged</option>
          <option value="all" className="bg-slate-900 text-slate-300">All</option>
        </select>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-white/5 overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-white/3 text-slate-500 uppercase tracking-widest">
            <tr>
              <th className="px-4 py-3 text-left">
                <button onClick={() => toggleSort('ts')} className="flex items-center gap-1">
                  Time <SortIcon k="ts" />
                </button>
              </th>
              <th className="px-4 py-3 text-left">
                <button onClick={() => toggleSort('severity')} className="flex items-center gap-1">
                  Severity <SortIcon k="severity" />
                </button>
              </th>
              <th className="px-4 py-3 text-left">
                <button onClick={() => toggleSort('score')} className="flex items-center gap-1">
                  Score <SortIcon k="score" />
                </button>
              </th>
              <th className="px-4 py-3 text-left">Reason</th>
              <th className="px-4 py-3 text-left">Status</th>
              <th className="px-4 py-3 text-left">Feedback</th>
              <th className="px-4 py-3 text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array(5).fill(0).map((_, i) => (
                <tr key={i} className="border-t border-white/5">
                  <td colSpan={5} className="px-4 py-3">
                    <div className="h-4 bg-white/5 rounded animate-pulse" />
                  </td>
                </tr>
              ))
            ) : sorted.length === 0 ? (
              <tr className="border-t border-white/5">
                <td colSpan={7} className="px-4 py-8 text-center text-slate-600">No alerts match filters</td>
              </tr>
            ) : (
              <AnimatePresence>
                {sorted.map(a => (
                  <motion.tr
                    key={a.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className={`border-t border-white/5 hover:bg-white/3 transition-colors ${
                      a.acknowledged ? 'opacity-40' : ''
                    }`}
                  >
                    <td className="px-4 py-2.5 font-mono text-slate-400">
                      {new Date(a.ts).toLocaleString()}
                    </td>
                    <td className="px-4 py-2.5"><SeverityBadge s={a.severity} /></td>
                    <td className="px-4 py-2.5 font-mono text-slate-300 tabular-nums">
                      {a.score.toFixed(4)}
                    </td>
                    <td className="px-4 py-2.5 text-slate-400 max-w-xs truncate">
                      {a.reason || '—'}
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex flex-col gap-1">
                        <select
                          value={a.status || 'new'}
                          onChange={(e) => updateStatus.mutate({ id: a.id, status: e.target.value, operator_note: a.operator_note })}
                          className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300"
                        >
                          <option value="new">New</option>
                          <option value="investigating">Investigating</option>
                          <option value="resolved">Resolved</option>
                        </select>
                        <input
                          type="text"
                          placeholder="Operator note..."
                          defaultValue={a.operator_note || ''}
                          onBlur={(e) => updateStatus.mutate({ id: a.id, status: a.status || 'new', operator_note: e.target.value })}
                          className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-[10px] text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/50"
                        />
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <select
                        value={a.feedback || ''}
                        onChange={(e) => updateFeedback.mutate({ id: a.id, feedback: e.target.value })}
                        className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300"
                      >
                        <option value="">None</option>
                        <option value="true_positive">True Positive</option>
                        <option value="false_positive">False Positive</option>
                      </select>
                    </td>
                    <td className="px-4 py-2.5 text-center flex items-center justify-center gap-2 h-full">
                      {a.acknowledged ? (
                        <CheckCheck className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <button
                          onClick={() => ack.mutate(a.id)}
                          disabled={ack.isPending}
                          className="px-2 py-1 rounded text-[10px] border border-slate-600 text-slate-400 hover:border-cyan-500/40 hover:text-cyan-300 transition-colors disabled:opacity-40"
                        >
                          Ack
                        </button>
                      )}
                      <a
                        href={`http://localhost:8000/reports/incident/${a.id}`}
                        target="_blank" rel="noreferrer"
                        className="px-2 py-1 rounded text-[10px] border border-slate-600 text-slate-400 hover:border-cyan-500/40 hover:text-cyan-300 transition-colors"
                      >
                        PDF
                      </a>
                    </td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
