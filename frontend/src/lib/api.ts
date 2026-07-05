// src/lib/api.ts — typed client for every API endpoint

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function req<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${path})`)
  return res.json()
}

// ── Types ──────────────────────────────────────────────────────────────────
export interface SensorReading {
  timestamp: string
  sensor_id: string
  value: number
  anomaly_score: number
  is_anomaly: boolean
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical'
  reason?: string
  model: string
  inference_ms?: number
  reading_id?: number
  alert_id?: number
}

export interface Alert {
  id: number
  reading_id: number
  ts: string
  sensor_id: string
  severity: string
  score: number
  reason?: string
  acknowledged: boolean
  status?: string
  operator_note?: string
  resolved_at?: string
  feedback?: string
}

export interface MetricsResponse {
  total_readings: number
  total_anomalies: number
  anomaly_rate: number
  avg_inference_latency_ms: number
  current_model: string
  stream_rate_rpm: number
  latest_run: {
    precision: number
    recall: number
    f1: number
    pr_auc: number
    roc_auc: number
    threshold: number
  } | null
}

export interface ModelComparison {
  rank: number
  name: string
  split: string
  precision: number
  recall: number
  f1: number
  roc_auc: number
  pr_auc: number
  false_alarm_rate: number
  detection_latency_steps: number
  avg_inference_ms: number
  throughput_rps: number
  figures: { pr: string; roc: string; confusion: string }
}

export interface RegistryEntry {
  name: string
  type: string
  artifact_path?: string
  threshold: number
  feature_set_hash: string
  test_metrics: Record<string, number>
  is_production: boolean
  notes?: string
  registered_at: string
}

export interface Experiment {
  id: number
  model: string
  split: string
  feature_set_hash: string
  threshold: number
  created_at: string
  [key: string]: unknown
}

export interface SystemStatus {
  api: string
  database: string
  websocket_clients: number
  production_model: string
  threshold: number
  total_readings: number
  stream_rate_rpm: number
  avg_inference_ms: number
  timestamp: string
}

export interface DataSummary {
  dataset: string
  source: string
  raw_exists: boolean
  processed_exists: boolean
  feature_columns?: string[]
  feature_count?: number
  total_rows?: number
  splits?: Record<string, number>
  label_counts?: { normal: number; anomaly: number }
  anomaly_rate_pct?: number
}

// ── API calls ─────────────────────────────────────────────────────────────
export const api = {
  health:           ()              => req<{ status: string; model: string }>('/health'),
  metrics:          ()              => req<MetricsResponse>('/metrics'),
  readings:         (limit = 80)   => req<SensorReading[]>(`/readings?limit=${limit}`),
  alerts:           (limit = 100)  => req<Alert[]>(`/alerts?limit=${limit}`),
  ackAlert:         (id: number)   => req<{ status: string }>(`/alerts/${id}/ack`, { method: 'POST' }),
  updateAlertStatus:(id: number, payload: { status: string; operator_note?: string }) => req<{ status: string }>(`/alerts/${id}/status`, { method: 'PUT', body: JSON.stringify(payload), headers: { 'Content-Type': 'application/json' } }),
  updateAlertFeedback:(id: number, payload: { feedback: string }) => req<{ status: string }>(`/alerts/${id}/feedback`, { method: 'PUT', body: JSON.stringify(payload), headers: { 'Content-Type': 'application/json' } }),
  models:           ()             => req<RegistryEntry[]>('/models'),
  modelRegistry:    ()             => req<RegistryEntry[]>('/models/registry'),
  modelComparison:  ()             => req<ModelComparison[]>('/models/comparison'),
  selectModel:      (name: string) => req<{ status: string; active_model: string; threshold: number }>(`/models/select/${name}`, { method: 'POST' }),
  experiments:      ()             => req<Experiment[]>('/experiments'),
  dataSummary:      ()             => req<DataSummary>('/data/summary'),
  systemStatus:     ()             => req<SystemStatus>('/system/status'),
  triggerRetrain:   ()             => req<{ status: string }>('/models/retrain', { method: 'POST' }),

  // Fault Injection
  faultsTypes:      ()             => req<{ types: string[]; descriptions: Record<string, string> }>('/faults/types'),
  faultsStatus:     ()             => req<{ fault_type?: string }>('/faults/status'),
  faultsInject:     (payload: { fault_type: string; duration_steps?: number; magnitude?: number; sensor_id?: string }) => req<{ fault_type?: string }>('/faults/inject', { method: 'POST', body: JSON.stringify(payload), headers: { 'Content-Type': 'application/json' } }),
  faultsStop:       ()             => req<{ fault_type?: string }>('/faults/stop', { method: 'POST' }),

  // Drift Detection
  driftStatus:      ()             => req<{ status: string; psi: number; mean_shift_sigma: number; recommendation: string; affected_features: string[] }>('/drift/status'),
  driftCheck:       ()             => req<{ status: string; psi: number; mean_shift_sigma: number; recommendation: string; affected_features: string[] }>('/drift/check', { method: 'POST' }),
}
