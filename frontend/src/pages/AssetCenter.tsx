import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, Camera, ArrowRight, Loader2, Cpu, CheckCircle2, AlertTriangle, Info } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function AssetCenter() {
  const navigate = useNavigate();
  
  const { data, isLoading } = useQuery({
    queryKey: ['assets'],
    queryFn: async () => {
      const res = await fetch("http://localhost:8000/assets/");
      if (!res.ok) throw new Error("Failed to fetch assets");
      return res.json();
    },
    refetchInterval: 10000
  });

  const getModalityConfig = (modality: string) => {
    switch (modality) {
      case 'time_series': return { icon: Cpu, route: '/live', label: 'Time-Series (NAB)' };
      case 'vibration': return { icon: Activity, route: '/vibration', label: 'Vibration (NASA Bearing)' };
      case 'vision': return { icon: Camera, route: '/vision', label: 'Vision (MVTec)' };
      default: return { icon: Info, route: '/', label: 'Unknown' };
    }
  };

  const getStatusBadge = (status: string) => {
    switch(status) {
      case 'operational': 
        return <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30"><CheckCircle2 className="w-3 h-3 mr-1"/> Operational</Badge>;
      case 'warning':
        return <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30"><AlertTriangle className="w-3 h-3 mr-1"/> Warning</Badge>;
      case 'critical':
        return <Badge variant="destructive"><AlertTriangle className="w-3 h-3 mr-1"/> Critical</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Asset Center</h1>
          <p className="text-slate-400 mt-2">Unified management of multi-modal industrial assets</p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12"><Loader2 className="w-8 h-8 text-cyan-400 animate-spin" /></div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {data?.assets?.map((asset: any) => {
            const config = getModalityConfig(asset.modality);
            const Icon = config.icon;
            
            return (
              <Card key={asset.id} className="bg-slate-900 border-slate-800 hover:border-slate-700 transition-colors">
                <CardHeader className="pb-3 border-b border-slate-800">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-slate-800 rounded-lg">
                        <Icon className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <CardTitle className="text-lg text-slate-100">{asset.name}</CardTitle>
                        <CardDescription className="text-slate-400 font-mono text-xs mt-1">{asset.id}</CardDescription>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-4 space-y-4">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">Modality</span>
                    <span className="text-slate-300 font-medium">{config.label}</span>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">Status</span>
                    {getStatusBadge(asset.status)}
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">Last Anomaly</span>
                    <span className="text-slate-300">{asset.last_anomaly_at ? new Date(asset.last_anomaly_at).toLocaleString() : 'None'}</span>
                  </div>
                  
                  <div className="pt-4">
                    <Button 
                      className="w-full bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700" 
                      onClick={() => navigate(config.route)}
                    >
                      Open Monitoring Lab <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
