import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Camera, RefreshCw, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function VisionLab() {
  const [gallery, setGallery] = useState<any[]>([]);
  const [selectedImage, setSelectedImage] = useState<any | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const fetchGallery = async () => {
    try {
      const res = await fetch("http://localhost:8000/image/gallery");
      if (res.ok) {
        const data = await res.json();
        setGallery(data.images || []);
      }
    } catch (e) {
      console.error("Failed to fetch gallery:", e);
    }
  };

  useEffect(() => {
    fetchGallery();
  }, []);

  const analyzeImage = async (imageObj: any) => {
    setSelectedImage(imageObj);
    setIsAnalyzing(true);
    setAnalysisResult(null);
    try {
      const res = await fetch("http://localhost:8000/image/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: imageObj.data })
      });
      if (res.ok) {
        const data = await res.json();
        setAnalysisResult(data);
      }
    } catch (e) {
      console.error("Analysis failed:", e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Visual Inspection Lab</h1>
          <p className="text-slate-400 mt-2">MVTec-style ResNet18 Embeddings + Isolation Forest</p>
        </div>
        <Button onClick={fetchGallery} variant="outline" className="border-slate-700 text-slate-300">
          <RefreshCw className="w-4 h-4 mr-2" /> Refresh Gallery
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main Inspection View */}
        <Card className="bg-slate-900 border-slate-800 md:col-span-2">
          <CardHeader>
            <CardTitle className="text-slate-200 flex items-center gap-2">
              <Camera className="w-5 h-5 text-cyan-400" />
              Inspection Camera
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            {selectedImage ? (
              <div className="relative border-4 border-slate-800 rounded-lg overflow-hidden mb-6">
                <img 
                  src={selectedImage.data} 
                  alt="Selected" 
                  className="w-full max-w-md object-contain"
                />
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center backdrop-blur-sm">
                    <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
                  </div>
                )}
              </div>
            ) : (
              <div className="border-4 border-dashed border-slate-800 rounded-lg w-full max-w-md h-64 flex flex-col items-center justify-center text-slate-500 mb-6">
                <Camera className="w-12 h-12 mb-2 opacity-50" />
                <p>Select an image from the gallery</p>
              </div>
            )}

            {/* Analysis Result */}
            {analysisResult && (
              <div className="w-full max-w-md bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-sm text-slate-400">Analysis Status</span>
                  {analysisResult.is_anomaly ? (
                    <Badge variant="destructive" className="px-3 py-1">
                      <AlertTriangle className="w-4 h-4 mr-1"/> {analysisResult.message}
                    </Badge>
                  ) : (
                    <Badge className="bg-emerald-500 hover:bg-emerald-600 px-3 py-1">
                      <CheckCircle2 className="w-4 h-4 mr-1"/> {analysisResult.message}
                    </Badge>
                  )}
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Anomaly Score</span>
                  <span className={`text-xl font-mono font-bold ${analysisResult.is_anomaly ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {analysisResult.anomaly_score.toFixed(3)}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Gallery */}
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-slate-200">Test Gallery</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
              {gallery.map((img, idx) => (
                <div 
                  key={idx} 
                  className={`cursor-pointer rounded-md overflow-hidden border-2 transition-all ${
                    selectedImage?.filename === img.filename ? 'border-cyan-500 scale-95' : 'border-transparent hover:border-slate-600'
                  }`}
                  onClick={() => analyzeImage(img)}
                >
                  <img src={img.data} alt={img.filename} className="w-full h-auto object-cover" />
                  <div className="bg-slate-800 text-xs text-center py-1 text-slate-300 truncate px-1">
                    {img.filename}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
