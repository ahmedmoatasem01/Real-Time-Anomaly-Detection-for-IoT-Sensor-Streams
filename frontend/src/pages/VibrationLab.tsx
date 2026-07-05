import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, Zap, Play, Pause, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface VibrationSample {
  file_index: number;
  timestamp: string;
  features: any;
  anomaly_score: number;
  is_anomaly: boolean;
  waveform: number[];
  fft: {
    frequencies: number[];
    magnitudes: number[];
  };
}

export default function VibrationLab() {
  const [sample, setSample] = useState<VibrationSample | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [history, setHistory] = useState<any[]>([]);

  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/vibration/ws/stream");
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setSample(data);
      
      // Add to history for degradation trend
      setHistory(prev => {
        const newHistory = [...prev, {
          index: data.file_index,
          timestamp: data.timestamp,
          score: data.anomaly_score,
          rms: data.features.rms
        }];
        // Keep last 100 for the trend chart
        return newHistory.slice(-100);
      });
    };

    socket.onopen = () => {
      console.log("Vibration WS Connected");
    };

    socket.onerror = (error) => {
      console.error("Vibration WS Error:", error);
    };
    
    setWs(socket);

    return () => {
      socket.close();
    };
  }, []);

  const togglePlay = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      const newPlayState = !isPlaying;
      ws.send(JSON.stringify({ command: newPlayState ? "play" : "pause" }));
      setIsPlaying(newPlayState);
    }
  };

  const waveformChartData = sample?.waveform.map((val, idx) => ({ idx, val })) || [];
  const fftChartData = sample?.fft.frequencies.map((freq, idx) => ({ 
    freq, 
    mag: sample.fft.magnitudes[idx] 
  })) || [];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Vibration Health Lab</h1>
          <p className="text-slate-400 mt-2">NASA Bearing Run-to-Failure Analysis</p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant={sample?.is_anomaly ? "destructive" : "default"} className="px-4 py-1 text-sm">
            {sample?.is_anomaly ? <><AlertTriangle className="w-4 h-4 mr-2"/> CRITICAL</> : "NORMAL"}
          </Badge>
          <Button onClick={togglePlay} variant={isPlaying ? "destructive" : "default"}>
            {isPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isPlaying ? "Pause Timeline" : "Start Timeline"}
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="pt-6 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Snapshot Time</p>
              <h3 className="text-xl font-bold text-slate-100 mt-1">{sample?.timestamp || '--'}</h3>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="pt-6 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">RMS Vibration</p>
              <h3 className="text-2xl font-bold text-cyan-400 mt-1">
                {sample?.features?.rms?.toFixed(4) || '0.0000'}
              </h3>
            </div>
            <Activity className="w-8 h-8 text-cyan-500 opacity-50" />
          </CardContent>
        </Card>
        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="pt-6 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Kurtosis</p>
              <h3 className="text-2xl font-bold text-amber-400 mt-1">
                {sample?.features?.kurtosis?.toFixed(2) || '0.00'}
              </h3>
            </div>
            <Activity className="w-8 h-8 text-amber-500 opacity-50" />
          </CardContent>
        </Card>
        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="pt-6 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Anomaly Score</p>
              <h3 className="text-2xl font-bold text-rose-400 mt-1">
                {sample?.anomaly_score?.toFixed(2) || '0.00'}
              </h3>
            </div>
            <Zap className="w-8 h-8 text-rose-500 opacity-50" />
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Waveform Chart */}
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-200">Raw Time-Domain Waveform</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="idx" hide />
                  <YAxis stroke="#475569" domain={[-1, 1]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}
                  />
                  <Line type="monotone" dataKey="val" stroke="#22d3ee" dot={false} strokeWidth={1} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* FFT Chart */}
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-200">Frequency Spectrum (FFT)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={fftChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="freq" stroke="#475569" tickFormatter={(v) => Math.round(v) + 'Hz'} />
                  <YAxis stroke="#475569" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}
                  />
                  <Line type="monotone" dataKey="mag" stroke="#fbbf24" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Degradation Trend */}
        <Card className="bg-slate-900 border-slate-800 md:col-span-2">
          <CardHeader>
            <CardTitle className="text-slate-200">Run-to-Failure Degradation Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="index" stroke="#475569" />
                  <YAxis yAxisId="left" stroke="#475569" />
                  <YAxis yAxisId="right" orientation="right" stroke="#475569" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}
                  />
                  <Line yAxisId="left" type="monotone" dataKey="score" name="Anomaly Score" stroke="#f43f5e" dot={false} strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="rms" name="RMS" stroke="#22d3ee" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
