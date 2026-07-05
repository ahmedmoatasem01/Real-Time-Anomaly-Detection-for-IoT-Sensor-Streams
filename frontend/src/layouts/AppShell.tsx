// src/layouts/AppShell.tsx
// Persistent left sidebar + top status bar + <Outlet/>

import { useState } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity, LayoutDashboard, FlaskConical, Database,
  Bell, Beaker, HeartPulse, Terminal, ChevronLeft, ChevronRight,
  Wifi, WifiOff, Loader2, Camera, Server, RefreshCw
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { useWebSocket, type WsStatus } from '../lib/ws'

const NAV = [
  { to: '/',           icon: LayoutDashboard, label: 'Overview'      },
  { to: '/live',       icon: Activity,        label: 'Live Monitor'  },
  { to: '/models',     icon: FlaskConical,    label: 'Model Lab'     },
  { to: '/data',       icon: Database,        label: 'Data Explorer' },
  { to: '/alerts',     icon: Bell,            label: 'Alert Center'  },
  { to: '/experiments',icon: Beaker,          label: 'Experiments'   },
  { to: '/retrain',    icon: RefreshCw,       label: 'Retraining'    },
  { to: '/health',     icon: HeartPulse,      label: 'System Health' },
  { to: '/assets',     icon: Server,          label: 'Asset Center'  },
  { to: '/vibration',  icon: Activity,        label: 'Vibration Lab' },
  { to: '/vision',     icon: Camera,          label: 'Vision Lab'    },
  { to: '/demo',       icon: Terminal,        label: 'Demo Panel'    },
]

function WsDot({ status }: { status: WsStatus }) {
  const color =
    status === 'connected'    ? 'bg-emerald-400' :
    status === 'connecting'   ? 'bg-amber-400 animate-pulse' :
                                'bg-red-500'
  return (
    <span className="flex items-center gap-1.5 text-xs text-slate-400">
      <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
      {status === 'connected' ? (
        <Wifi className="w-3 h-3" />
      ) : status === 'connecting' ? (
        <Loader2 className="w-3 h-3 animate-spin" />
      ) : (
        <WifiOff className="w-3 h-3 text-red-400" />
      )}
    </span>
  )
}

export default function AppShell() {
  const [collapsed, setCollapsed] = useState(false)
  const [wsStatus, setWsStatus] = useState<WsStatus>('connecting')
  const location = useLocation()

  const { data: systemData } = useQuery({
    queryKey: ['system-status'],
    queryFn: api.systemStatus,
    refetchInterval: 10000,
  })

  // Lightweight WS to derive connection status for top bar
  const { status } = useWebSocket<unknown>(() => {})
  // expose status update
  if (status !== wsStatus) setWsStatus(status)

  const activeRoute = NAV.find(n => n.to === location.pathname)

  return (
    <div className="flex h-screen bg-[#0A0E14] text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`flex flex-col border-r border-white/5 bg-[#0d1117] transition-all duration-200 ${
          collapsed ? 'w-14' : 'w-52'
        }`}
      >
        {/* Logo */}
        <div className="flex items-center gap-2 px-3 py-4 border-b border-white/5">
          <Activity className="w-5 h-5 text-cyan-400 shrink-0" />
          {!collapsed && (
            <span className="text-xs font-semibold text-cyan-300 tracking-widest uppercase truncate">
              AnomalyOps
            </span>
          )}
        </div>

        {/* Nav links */}
        <nav className="flex-1 py-3 space-y-0.5 px-1.5 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-2 py-2 rounded-md text-xs transition-colors ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20'
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center h-10 border-t border-white/5 text-slate-500 hover:text-slate-300 transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top status bar */}
        <header className="flex items-center justify-between px-5 py-2.5 border-b border-white/5 bg-[#0d1117] shrink-0">
          <span className="text-xs text-slate-400 font-semibold tracking-wide uppercase">
            {activeRoute?.label ?? 'Dashboard'}
          </span>
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <WsDot status={wsStatus} />
            {systemData && (
              <>
                <span className="text-slate-600">|</span>
                <span className="text-cyan-400">{systemData.production_model}</span>
                <span className="text-slate-600">|</span>
                <span>{systemData.total_readings.toLocaleString()} readings</span>
                <span className="text-slate-600">|</span>
                <span>{systemData.avg_inference_ms.toFixed(2)} ms</span>
              </>
            )}
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.15 }}
              className="h-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}
