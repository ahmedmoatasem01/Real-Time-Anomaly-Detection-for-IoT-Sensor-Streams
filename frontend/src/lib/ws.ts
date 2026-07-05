// src/lib/ws.ts — reusable WebSocket hook with exponential backoff

import { useEffect, useRef, useState, useCallback } from 'react'

export type WsStatus = 'connecting' | 'connected' | 'disconnected'

const WS_BASE = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/stream'
const RECONNECT_BASE_MS = 1500
const RECONNECT_MAX_MS  = 30000

export function useWebSocket<T>(onMessage: (data: T) => void) {
  const [status, setStatus] = useState<WsStatus>('connecting')
  const wsRef       = useRef<WebSocket | null>(null)
  const delayRef    = useRef(RECONNECT_BASE_MS)
  const timerRef    = useRef<ReturnType<typeof setTimeout> | null>(null)
  const stableMsg   = useRef(onMessage)

  useEffect(() => { stableMsg.current = onMessage }, [onMessage])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    const ws = new WebSocket(WS_BASE)
    wsRef.current = ws
    setStatus('connecting')

    ws.onopen = () => {
      setStatus('connected')
      delayRef.current = RECONNECT_BASE_MS
    }
    ws.onmessage = (e) => {
      try { stableMsg.current(JSON.parse(e.data) as T) } catch { /* ignore bad frames */ }
    }
    ws.onerror = () => setStatus('disconnected')
    ws.onclose = () => {
      setStatus('disconnected')
      const delay = Math.min(delayRef.current, RECONNECT_MAX_MS)
      delayRef.current = delay * 2
      timerRef.current = setTimeout(connect, delay)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { status }
}
