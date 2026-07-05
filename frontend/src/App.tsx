// App.tsx — route configuration with lazy-loaded pages
import { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import AppShell from './layouts/AppShell'

const Overview      = lazy(() => import('./pages/Overview'))
const LiveMonitor   = lazy(() => import('./pages/LiveMonitor'))
const ModelLab      = lazy(() => import('./pages/ModelLab'))
const DataExplorer  = lazy(() => import('./pages/DataExplorer'))
const AlertCenter   = lazy(() => import('./pages/AlertCenter'))
const ExperimentLog = lazy(() => import('./pages/ExperimentLog'))
const SystemHealth  = lazy(() => import('./pages/SystemHealth'))
const RetrainingCenter = lazy(() => import('./pages/RetrainingCenter'))
const DemoPanel     = lazy(() => import('./pages/DemoPanel'))
const VibrationLab  = lazy(() => import('./pages/VibrationLab'))
const VisionLab     = lazy(() => import('./pages/VisionLab'))
const AssetCenter   = lazy(() => import('./pages/AssetCenter'))

function PageSkeleton() {
  return (
    <div className="flex items-center justify-center h-full min-h-[200px]">
      <div className="flex gap-1.5">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="w-2 h-2 bg-cyan-500/60 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.12}s` }}
          />
        ))}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index           element={<Suspense fallback={<PageSkeleton />}><Overview /></Suspense>} />
        <Route path="live"     element={<Suspense fallback={<PageSkeleton />}><LiveMonitor /></Suspense>} />
        <Route path="models"   element={<Suspense fallback={<PageSkeleton />}><ModelLab /></Suspense>} />
        <Route path="data"     element={<Suspense fallback={<PageSkeleton />}><DataExplorer /></Suspense>} />
        <Route path="alerts"   element={<Suspense fallback={<PageSkeleton />}><AlertCenter /></Suspense>} />
        <Route path="experiments" element={<Suspense fallback={<PageSkeleton />}><ExperimentLog /></Suspense>} />
        <Route path="health"   element={<Suspense fallback={<PageSkeleton />}><SystemHealth /></Suspense>} />
        <Route path="retrain"  element={<Suspense fallback={<PageSkeleton />}><RetrainingCenter /></Suspense>} />
        <Route path="demo"     element={<Suspense fallback={<PageSkeleton />}><DemoPanel /></Suspense>} />
        <Route path="vibration" element={<Suspense fallback={<PageSkeleton />}><VibrationLab /></Suspense>} />
        <Route path="vision"   element={<Suspense fallback={<PageSkeleton />}><VisionLab /></Suspense>} />
        <Route path="assets"   element={<Suspense fallback={<PageSkeleton />}><AssetCenter /></Suspense>} />
      </Route>
    </Routes>
  )
}
