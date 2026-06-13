import React, { useEffect, useState, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Terminal, X, Moon, Sun, Bell } from 'lucide-react';
import { useScanStore } from '../stores/scanStore';
import { useUiStore } from '../stores/uiStore';
import { cn } from '../lib/utils';

const CLINICAL_LOGS = [
  { t: 0,  msg: "Establishing secure connection to ML inference engine...", color: "text-on-surface-variant" },
  { t: 5,  msg: "Decompressing CT study archive...", color: "text-on-surface-variant" },
  { t: 15, msg: "Normalizing contrast levels via CLAHE algorithm...", color: "text-[#89ceff]" },
  { t: 30, msg: "Isolating hepatic boundary structures (FasNet U-Net)...", color: "text-on-surface-variant" },
  { t: 45, msg: "Applying semantic segmentation masks across all slices...", color: "text-[#89ceff]" },
  { t: 65, msg: "Scanning for potential neoplastic lesions...", color: "text-orange-400 font-bold" },
  { t: 80, msg: "Calculating volumetric density & confidence scores...", color: "text-on-surface-variant" },
  { t: 95, msg: "Aggregating clinical findings into diagnostic report...", color: "text-emerald-400" },
];

const Analysis = () => {
  const { scanId } = useParams();
  const navigate = useNavigate();
  const { scans, setAnalysisState, fetchAndUpdateScan } = useScanStore();
  const { theme, toggleTheme } = useUiStore();
  const [simulatedProgress, setSimulatedProgress] = useState(0);
  const [isBackendComplete, setIsBackendComplete] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const pollingRef = useRef(null);

  // Poll FastAPI every 3 seconds until status === 'complete'
  useEffect(() => {
    const pollScan = async () => {
      const data = await fetchAndUpdateScan(scanId);
      if (data && data.status === 'complete' && data.result) {
        setIsBackendComplete(true);
        setScanResult(data);
        // Stop polling
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
      }
    };

    // Start polling immediately
    pollScan();
    pollingRef.current = setInterval(pollScan, 3000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [scanId, fetchAndUpdateScan]);

  // Simulate smooth progress up to 95%, then wait for backend
  useEffect(() => {
    const iv = setInterval(() => {
      setSimulatedProgress(p => {
        if (p >= 95) {
          return isBackendComplete ? 100 : 95;
        }
        const increment = isBackendComplete ? 4 : 2; 
        return Math.min(p + increment, 95);
      });
    }, 150);

    return () => clearInterval(iv);
  }, [isBackendComplete]);

  // Navigate when fully 100%
  useEffect(() => {
    if (simulatedProgress === 100 && isBackendComplete && scanResult) {
      const hasAnomaly = scanResult.result?.hasAnomaly || scanResult.result?.result?.hasAnomaly;
      setAnalysisState(hasAnomaly ? 'anomaly' : 'clear');
      const t = setTimeout(() => navigate(`/results/${scanId}`), 1000);
      return () => clearTimeout(t);
    }
  }, [simulatedProgress, isBackendComplete, scanId, navigate, scanResult, setAnalysisState]);

  const activeLogs = CLINICAL_LOGS.filter(l => simulatedProgress >= l.t);

  return (
    <div className="flex-1 bg-background min-h-full flex flex-col">
      <style dangerouslySetInnerHTML={{__html: `
        .scan-line { animation: sweep 1.5s linear infinite; }
        @keyframes sweep { 0%{top:0%;opacity:0} 5%{opacity:1} 95%{opacity:1} 100%{top:100%;opacity:0} }
      `}} />

      <header className="flex justify-between items-center px-8 py-4">
        <span className="syne font-bold text-lg text-[#89ceff]">Analyzing Scan...</span>
        <div className="flex items-center gap-3">
          <button onClick={toggleTheme} className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-surface-container-high transition-colors">
            {theme === 'dark' ? <Moon className="w-5 h-5 text-on-surface-variant" /> : <Sun className="w-5 h-5 text-on-surface-variant" />}
          </button>
          <button className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-surface-container-high transition-colors relative">
            <Bell className="w-5 h-5 text-on-surface-variant" /><span className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full"></span>
          </button>
        </div>
      </header>

      <section className="flex-1 p-8 flex flex-col items-center justify-center pb-24">
        <div className="w-full max-w-[800px] bg-surface-container border-2 border-tertiary/20 rounded-xl overflow-hidden shadow-[0_0_40px_-10px_rgba(245,158,11,0.25)]">
          {/* Progress */}
          <div className="p-6 bg-surface-container-low flex flex-col gap-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 bg-tertiary rounded-full animate-pulse"></span>
                <h2 className="syne font-bold text-lg text-on-surface">
                  {isBackendComplete && simulatedProgress === 100 ? 'Analysis Complete!' : 'Processing CT Slices...'}
                </h2>
              </div>
              <span className="font-mono text-sm text-tertiary">{simulatedProgress}%</span>
            </div>
            <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
              <div className={cn("h-full transition-all duration-300 rounded-full", isBackendComplete && simulatedProgress === 100 ? "bg-emerald-500" : "bg-tertiary shadow-[0_0_10px_rgba(255,185,95,0.5)]")}
                style={{ width: `${simulatedProgress}%` }}></div>
            </div>
          </div>

          {/* Viewer Area */}
          <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
            <img 
              className="w-full h-full object-contain opacity-80" 
              alt="CT scan analysis view" 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuCdOZg6EMwrEbUdWa2OWTNs33BHFaYycu_N9sZK8iwb7yBmIO_1BwyfdRZWdTX1XZjKbVu9Eq5EyFRTkvjFsY6dj8SFAE2npKjxRTkeQd1UAIDezDHp-jRulayLZs_ktzVV-GdoHZIUC5x6fqAtBuPgYr9CYYJP29Lxno5keGwV8wJCGJI0eRvZ9ENlsNMhIldzGtaVU2ZtWXmSyQwL5i4H6VMx0cyzue5H4TTljMbUK9roEt-q2U80gqpmgZI6LEZYL_a8-RjqPhU" 
            />
            {/* Scan Line Overlay */}
            <div className="absolute left-0 right-0 h-0.5 bg-cyan-400 scan-line shadow-[0_0_15px_#22d3ee] z-10"></div>
            
            {/* HUD Overlays */}
            <div className="absolute top-4 left-4 flex flex-col gap-1 z-20">
              <span className="font-mono text-[10px] text-cyan-400 bg-black/40 px-2 py-0.5 rounded backdrop-blur-md">
                SLICE: {Math.min(Math.floor(simulatedProgress / 2) + 1, 47)} / 47
              </span>
            </div>
            <div className="absolute bottom-4 right-4 z-20">
              <div className="bg-black/60 border border-tertiary/30 backdrop-blur-md p-3 rounded-lg flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-tertiary rounded-full animate-pulse"></span>
                  <span className="font-mono text-[11px] text-on-surface">Anomaly probability: {(simulatedProgress / 100 * 0.82).toFixed(2)}</span>
                </div>
                <div className="font-mono text-[10px] text-on-surface-variant">Voxel Analysis: Active</div>
              </div>
            </div>
          </div>

          {/* Logs */}
          <div className="p-6 bg-surface-container-lowest border-t border-outline-variant/15">
            <div className="flex items-center gap-2 mb-4 text-xs font-bold uppercase tracking-widest text-on-surface-variant">
              <Terminal className="w-4 h-4" /> Processing Logs
            </div>
            <div className="space-y-2 font-mono text-[12px] h-40 overflow-y-auto flex flex-col justify-end">
              {activeLogs.length === 0 && (
                <div className="text-on-surface-variant opacity-50">Initializing inference engine...</div>
              )}
              {activeLogs.map((log, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                  className={cn("flex gap-4", log.color)}>
                  <span className="text-outline">[{String(Math.floor(i * 1.5)).padStart(2,'0')}:{String((i * 23) % 60).padStart(2,'0')}]</span>
                  <span>{log.msg}</span>
                </motion.div>
              ))}
              {/* Auto-scroll anchor */}
              <div ref={(el) => el?.scrollIntoView({ behavior: 'smooth' })} />
            </div>
          </div>
        </div>

        <div className="mt-8 flex gap-4">
          <button onClick={() => navigate('/upload')}
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-surface-container-high border border-outline-variant/30 text-sm font-medium text-on-surface hover:border-error/50 transition-all">
            <X className="w-4 h-4" /> Abort Analysis
          </button>
        </div>
      </section>
    </div>
  );
};

export default Analysis;
