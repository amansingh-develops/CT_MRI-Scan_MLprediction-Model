import React, { useState, useEffect, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useScanStore } from '../stores/scanStore';
import { useAuthStore } from '../stores/authStore';
import { useUiStore } from '../stores/uiStore';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Download, ZoomIn, Layers, Moon, Sun, Bell,
  AlertTriangle, CheckCircle, Info, ChevronDown, ChevronUp,
  Activity, Shield, Clock, Cpu, Eye, Grid3X3
} from 'lucide-react';
import { cn } from '../lib/utils';

const Results = () => {
  const { scanId } = useParams();
  const { scans, fetchAndUpdateScan } = useScanStore();
  const { user } = useAuthStore();
  const { theme, toggleTheme } = useUiStore();

  const scan = scans.find(s => s.scanId === scanId);
  
  // The backend wraps result in { success, scanId, result: { ... } }
  // Unwrap: if scan.result.result exists, use it; otherwise use scan.result directly
  const rawResult = scan?.result || null;
  const r = rawResult?.result || rawResult;

  const [sliceIdx, setSliceIdx] = useState(0);
  const [viewMode, setViewMode] = useState('overlay');
  const [showModelInfo, setShowModelInfo] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isLoading, setIsLoading] = useState(!r);

  const handleDownloadReport = async () => {
    setIsDownloading(true);
    try {
      const apiUrl = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/report/${scanId}`);
      if (!response.ok) throw new Error('Report generation failed');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ScanSight_Report_${scanId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();
    } catch (err) {
      console.error('PDF download failed:', err);
      alert('Failed to generate report. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  // Fetch from FastAPI if data isn't in the store
  useEffect(() => {
    if (!r && scanId) {
      setIsLoading(true);
      fetchAndUpdateScan(scanId).finally(() => setIsLoading(false));
    }
  }, [scanId, r, fetchAndUpdateScan]);

  const slices = useMemo(() => (r?.slices || []).filter(s => !s.error), [r]);
  const currentSlice = slices[sliceIdx] || null;
  const hasAnomaly = r?.hasAnomaly || false;
  const tumorIndices = useMemo(() => new Set(slices.filter(s => s.hasTumor).map(s => s.sliceIndex)), [slices]);

  const imageUrl = currentSlice
    ? viewMode === 'overlay' ? currentSlice.overlayUrl
    : viewMode === 'mask' ? currentSlice.maskUrl
    : currentSlice.originalUrl
    : '';

  if (!scan || !r || isLoading) {
    return (
      <div className="flex-1 p-8 bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-on-surface-variant">Loading analysis results...</p>
          <p className="text-xs text-on-surface-variant/60 mt-2 font-mono">SCAN: {scanId}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 md:p-8 bg-background overflow-y-auto min-h-full pb-16">
      {/* Header */}
      <header className="flex justify-between items-center mb-6">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-3">
          <Link to="/dashboard" className="p-2 hover:bg-surface-container-high rounded-full transition-colors">
            <ArrowLeft className="w-5 h-5 text-on-surface" />
          </Link>
          <div>
            <h1 className="syne text-xl font-bold text-on-surface">Diagnostic Report</h1>
            <div className="flex items-center gap-3 mt-0.5">
              <span className="font-mono text-[10px] text-on-surface-variant">{scan.scanId}</span>
              <span className="text-[10px] text-on-surface-variant">•</span>
              <span className="font-mono text-[10px] text-on-surface-variant">{scan.uploadedAt || 'N/A'}</span>
            </div>
          </div>
        </motion.div>
        <div className="flex gap-3">
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={handleDownloadReport}
            disabled={isDownloading}
            className={cn(
              "h-10 px-4 flex items-center gap-2 rounded-lg font-bold text-xs transition-all",
              isDownloading
                ? "bg-surface-container-high text-on-surface-variant cursor-wait"
                : "bg-primary text-on-primary hover:shadow-lg hover:shadow-primary/20"
            )}
          >
            {isDownloading ? (
              <><div className="w-4 h-4 border-2 border-on-surface-variant border-t-transparent rounded-full animate-spin"></div> Generating...</>
            ) : (
              <><Download className="w-4 h-4" /> Download PDF</>
            )}
          </motion.button>
          <button onClick={toggleTheme} className="w-10 h-10 flex items-center justify-center rounded-lg bg-surface-container-high hover:bg-surface-bright transition-colors">
            {theme === 'dark' ? <Moon className="w-4 h-4 text-on-surface-variant" /> : <Sun className="w-4 h-4 text-on-surface-variant" />}
          </button>
          <button className="w-10 h-10 flex items-center justify-center rounded-lg bg-surface-container-high hover:bg-surface-bright transition-colors relative">
            <Bell className="w-4 h-4 text-on-surface-variant" />
          </button>
        </div>
      </header>

      {/* Banner */}
      <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }}
        className={cn(
          "rounded-xl p-5 mb-6 border-l-4 flex flex-col md:flex-row md:items-center justify-between gap-4",
          hasAnomaly
            ? "bg-surface-container-low border-orange-500 shadow-[0_0_30px_-10px_rgba(249,115,22,0.2)]"
            : "bg-surface-container-low border-emerald-500 shadow-[0_0_30px_-10px_rgba(16,185,129,0.15)]"
        )}
      >
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className={cn("w-2 h-2 rounded-full animate-pulse", hasAnomaly ? "bg-orange-500" : "bg-emerald-500")}></span>
            <span className={cn("text-[10px] font-bold uppercase tracking-widest", hasAnomaly ? "text-orange-500" : "text-emerald-500")}>
              {hasAnomaly ? 'Anomaly Detected' : 'Clear — No Anomaly'}
            </span>
          </div>
          <h2 className="syne text-lg font-bold text-on-surface mb-1">
            {hasAnomaly
              ? `AI detected abnormal tissue in ${r.anomalySlices} of ${r.totalSlices} slices`
              : `All ${r.totalSlices} slices analyzed — no abnormal tissue detected`}
          </h2>
          {r.estimatedStage && (
            <span className="inline-block mt-1 px-3 py-1 rounded-full bg-orange-500/10 text-orange-400 text-xs font-bold">{r.estimatedStage}</span>
          )}
        </div>
        <div className="text-right shrink-0">
          <p className="text-[10px] uppercase tracking-widest text-on-surface-variant font-bold mb-0.5">Confidence</p>
          <p className={cn("syne text-4xl font-extrabold tracking-tighter", hasAnomaly ? "text-orange-500" : "text-emerald-500")}>
            {r.confidence}<span className="text-lg">%</span>
          </p>
        </div>
      </motion.div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Left: Slice Viewer */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="lg:col-span-7 space-y-4">
          {/* Viewer */}
          <div className="bg-black rounded-xl overflow-hidden relative" style={{ minHeight: 400 }}>
            {imageUrl ? (
              <img src={imageUrl} alt={`Slice ${sliceIdx}`} className="w-full h-full object-contain" style={{ minHeight: 400 }} />
            ) : (
              <div className="flex items-center justify-center h-[400px] text-on-surface-variant text-sm">No image available</div>
            )}
            {/* HUD */}
            <div className="absolute top-3 left-3 flex flex-col gap-1.5 z-10">
              <div className="px-2.5 py-1 rounded bg-black/60 backdrop-blur-md text-[10px] font-mono text-white flex items-center gap-2">
                SLICE {sliceIdx + 1} / {slices.length}
              </div>
              {currentSlice && (
                <>
                  <div className="px-2.5 py-1 rounded bg-black/60 backdrop-blur-md text-[10px] font-mono text-white flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span> LIVER: {currentSlice.liverPercent}%
                  </div>
                  <div className={cn("px-2.5 py-1 rounded bg-black/60 backdrop-blur-md text-[10px] font-mono text-white flex items-center gap-2", currentSlice.hasTumor && "text-red-400")}>
                    <span className={cn("w-1.5 h-1.5 rounded-full", currentSlice.hasTumor ? "bg-red-500" : "bg-emerald-500")}></span>
                    TUMOR: {currentSlice.tumorPercent}%
                  </div>
                </>
              )}
            </div>
            {currentSlice?.hasTumor && (
              <div className="absolute top-3 right-3 px-2.5 py-1 rounded bg-red-500/80 text-[10px] font-bold text-white uppercase tracking-wider animate-pulse">
                Anomaly
              </div>
            )}
            {/* View mode toggle */}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur-md rounded-full px-1 py-1 flex gap-1 z-10">
              {[{k:'original',l:'Original',i:Eye},{k:'overlay',l:'Overlay',i:Layers},{k:'mask',l:'Mask',i:Grid3X3}].map(({k,l,i:Icon})=>(
                <button key={k} onClick={()=>setViewMode(k)}
                  className={cn("px-3 py-1.5 rounded-full text-[10px] font-bold flex items-center gap-1.5 transition-all",
                    viewMode===k ? "bg-primary text-on-primary" : "text-white/70 hover:text-white")}>
                  <Icon className="w-3 h-3"/>{l}
                </button>
              ))}
            </div>
          </div>

          {/* Slider */}
          <div className="bg-surface-container rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Slice Navigator</span>
              <span className="font-mono text-xs text-on-surface">{sliceIdx + 1} / {slices.length}</span>
            </div>
            {/* Tumor markers */}
            <div className="relative h-2 mb-1">
              {slices.map((s, i) => s.hasTumor && (
                <div key={i} className="absolute top-0 w-1 h-2 bg-red-500/60 rounded-full"
                  style={{ left: `${(i / Math.max(slices.length - 1, 1)) * 100}%` }} />
              ))}
            </div>
            <input type="range" min={0} max={Math.max(slices.length - 1, 0)} value={sliceIdx}
              onChange={e => setSliceIdx(Number(e.target.value))}
              className="w-full accent-primary cursor-pointer" />
          </div>
        </motion.div>

        {/* Right: Metrics */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="lg:col-span-5 space-y-4">

          {/* Confidence */}
          <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/10">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4 text-primary" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">AI Diagnostic Confidence</span>
            </div>
            <div className="flex items-baseline gap-1 mb-3">
              <span className="text-4xl font-mono font-bold text-primary">{r.confidence}</span>
              <span className="text-lg font-mono text-primary/60">%</span>
            </div>
            <div className="w-full bg-surface-container-highest h-1.5 rounded-full overflow-hidden">
              <div className="bg-primary h-full rounded-full transition-all" style={{ width: `${Math.min(r.confidence, 100)}%` }}></div>
            </div>
            {hasAnomaly && (
              <div className="mt-3 flex items-center gap-2 text-xs text-orange-400">
                <AlertTriangle className="w-3.5 h-3.5" /> Clinical verification recommended
              </div>
            )}
          </div>

          {/* Volumetric Analysis */}
          <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/10">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-secondary-fixed-dim" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Volumetric Analysis</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Liver Coverage', value: `${r.liverVolumePercent || 0}%` },
                { label: 'Tumor:Liver Ratio', value: `${r.tumorToLiverRatio || 0}%` },
                { label: 'Total Liver (px)', value: (r.totalLiverPx || 0).toLocaleString() },
                { label: 'Total Tumor (px)', value: (r.totalTumorPx || 0).toLocaleString() },
              ].map((m, i) => (
                <div key={i} className="bg-surface-container-high rounded-lg p-3">
                  <p className="text-[10px] text-on-surface-variant mb-1">{m.label}</p>
                  <p className="font-mono text-sm font-bold text-on-surface">{m.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Staging (only if anomaly) */}
          {hasAnomaly && r.estimatedStage && (
            <div className="bg-surface-container rounded-xl p-5 border border-orange-500/20">
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Staging Assessment</span>
              <div className="flex items-center gap-3 mt-2">
                <span className="text-xl syne font-bold text-on-surface">{r.estimatedStage}</span>
                <span className="px-2 py-0.5 rounded bg-orange-500/10 text-orange-400 text-[10px] font-bold">
                  {r.anomalySlices} affected slice{r.anomalySlices !== 1 ? 's' : ''}
                </span>
              </div>
              {r.affectedSliceRange && (
                <p className="text-xs text-on-surface-variant mt-2 font-mono">
                  Range: Slice {r.affectedSliceRange.start + 1} → {r.affectedSliceRange.end + 1}
                </p>
              )}
            </div>
          )}

          {/* Per-Slice Table */}
          <div className="bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10">
            <div className="px-5 py-3 border-b border-outline-variant/10 flex justify-between items-center">
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Slice Findings</span>
              <span className="text-[10px] font-mono text-on-surface-variant bg-surface-container-highest px-2 py-0.5 rounded">
                {slices.length} slices
              </span>
            </div>
            <div className="max-h-[220px] overflow-y-auto">
              <table className="w-full text-left">
                <thead className="sticky top-0 bg-surface-container/95 backdrop-blur text-[10px] font-bold uppercase tracking-widest text-on-surface-variant z-10">
                  <tr>
                    <th className="px-4 py-2.5">#</th>
                    <th className="px-4 py-2.5 text-right">Liver%</th>
                    <th className="px-4 py-2.5 text-right">Tumor%</th>
                    <th className="px-4 py-2.5 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/5">
                  {slices.map((s, i) => (
                    <tr key={i} onClick={() => setSliceIdx(i)}
                      className={cn("hover:bg-surface-container-high transition-colors cursor-pointer text-xs",
                        i === sliceIdx && "bg-primary/5")}>
                      <td className="px-4 py-2 font-mono">{i + 1}</td>
                      <td className="px-4 py-2 text-right font-mono">{s.liverPercent}</td>
                      <td className="px-4 py-2 text-right font-mono">{s.tumorPercent}</td>
                      <td className="px-4 py-2 text-right">
                        {s.hasTumor
                          ? <span className="text-[9px] font-bold text-red-400 bg-red-400/10 px-1.5 py-0.5 rounded">ANOMALY</span>
                          : <span className="text-[9px] font-bold text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">CLEAR</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Clinical Interpretation */}
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="mt-8">
        <div className="bg-surface-container rounded-xl p-6 border border-outline-variant/10">
          <h3 className="syne font-bold text-lg text-on-surface mb-3 flex items-center gap-2">
            <Info className="w-5 h-5 text-primary" /> Clinical Interpretation
          </h3>
          <div className="text-sm text-on-surface-variant leading-relaxed space-y-3">
            <p>
              The AI model analyzed <strong className="text-on-surface">{r.totalSlices} CT scan slices</strong>.
              {hasAnomaly ? (
                <> <strong className="text-orange-400">{r.anomalySlices} out of {r.totalSlices} slices</strong> showed
                  signs of abnormal tissue growth. The overall anomaly detection confidence is <strong className="text-on-surface">{r.confidence}%</strong>.</>
              ) : (
                <> <strong className="text-emerald-400">No slices</strong> showed signs of abnormal tissue growth.
                  The model confidence for a clear scan is <strong className="text-on-surface">{r.confidence}%</strong>.</>
              )}
            </p>
            {hasAnomaly && r.maxTumorSlice && (
              <p>
                The largest tumor concentration was found in <strong className="text-on-surface">slice "{r.maxTumorSlice.sliceName}"</strong> with
                tumor coverage of <strong className="text-orange-400">{r.maxTumorSlice.tumorPercent}%</strong> of that slice.
                {r.estimatedStage && <> The estimated clinical staging is <strong className="text-orange-400">{r.estimatedStage}</strong>.</>}
              </p>
            )}
            <p>
              Liver parenchyma was detected covering <strong className="text-on-surface">{r.liverVolumePercent || 0}%</strong> of the total scan area.
              {r.tumorToLiverRatio > 0 && <> The tumor-to-liver volume ratio is <strong className="text-on-surface">{r.tumorToLiverRatio}%</strong>.</>}
            </p>
            <div className="mt-4 p-3 bg-surface-container-high rounded-lg border border-outline-variant/10">
              <p className="text-[11px] text-on-surface-variant/80 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-on-surface-variant/50" />
                <span><strong>Disclaimer:</strong> This analysis is generated by an AI model ({r.modelVersion || 'U-Net v1.0'}) and is intended for informational purposes only. It does not constitute a medical diagnosis. Please consult a licensed radiologist or hepatologist for clinical confirmation.</span>
              </p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Model Info (Collapsible) */}
      <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-4">
        <button onClick={() => setShowModelInfo(!showModelInfo)}
          className="w-full flex items-center justify-between px-5 py-3 bg-surface-container rounded-xl border border-outline-variant/10 hover:bg-surface-container-high transition-colors">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-on-surface-variant" />
            <span className="text-xs font-bold uppercase tracking-widest text-on-surface-variant">Model & Processing Info</span>
          </div>
          {showModelInfo ? <ChevronUp className="w-4 h-4 text-on-surface-variant" /> : <ChevronDown className="w-4 h-4 text-on-surface-variant" />}
        </button>
        <AnimatePresence>
          {showModelInfo && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden">
              <div className="bg-surface-container-low rounded-b-xl p-5 border-x border-b border-outline-variant/10 grid grid-cols-2 md:grid-cols-4 gap-3 -mt-1">
                {[
                  { l: 'Model', v: r.modelVersion || 'U-Net v1.0' },
                  { l: 'Processing Time', v: r.processingTimeMs ? `${(r.processingTimeMs / 1000).toFixed(1)}s` : 'N/A' },
                  { l: 'Input Resolution', v: '256 × 256 px' },
                  { l: 'Slices Processed', v: r.totalSlices },
                ].map((m, i) => (
                  <div key={i}>
                    <p className="text-[10px] text-on-surface-variant mb-0.5">{m.l}</p>
                    <p className="font-mono text-xs font-bold text-on-surface">{m.v}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>
    </div>
  );
};

export default Results;
