import React, { useState } from 'react';
import { useScanStore } from '../stores/scanStore';
import { useUiStore } from '../stores/uiStore';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ArrowLeft, RefreshCcw, Moon, Sun, Bell, TrendingUp, BarChart2, 
  ArrowUpRight, Download
} from 'lucide-react';
import { cn } from '../lib/utils';

const Compare = () => {
  const { theme, toggleTheme } = useUiStore();
  const [syncSlices, setSyncSlices] = useState(true);

  return (
    <div className="flex-1 bg-background relative min-h-full flex flex-col">
      <style dangerouslySetInnerHTML={{__html: `
        .violet-glow { filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.15)); }
        .ambient-violet { background: radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.08) 0%, transparent 70%); }
      `}} />

      {/* TopAppBar */}
      <header className="bg-[#0f131d] flex justify-between items-center w-full px-6 py-3 h-16 sticky top-0 z-40 border-b border-outline-variant/5">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="p-2 hover:bg-[#262a34] rounded-full transition-colors text-slate-400">
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <h1 className="syne font-bold text-lg tracking-tight text-[#89ceff]">Scan Comparison</h1>
        </div>
        <div className="flex items-center gap-4">
          <div 
            onClick={() => setSyncSlices(!syncSlices)}
            className="flex items-center bg-surface-container-high rounded-full px-4 py-1.5 gap-2 border border-outline-variant/10 cursor-pointer hover:border-primary/50 transition-colors"
          >
            <RefreshCcw className="w-4 h-4 text-tertiary" />
            <span className="text-xs font-bold tracking-wider uppercase text-on-surface">Sync Slices</span>
            <div className={cn("w-8 h-4 rounded-full relative transition-colors duration-300", syncSlices ? "bg-primary-container" : "bg-surface-variant")}>
              <div className={cn("absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all duration-300", syncSlices ? "right-0.5" : "left-0.5")}></div>
            </div>
          </div>
          <button onClick={toggleTheme} className="text-slate-400 hover:bg-[#262a34] p-2 rounded-full transition-colors">
            {theme === 'dark' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
          </button>
          <button className="text-slate-400 hover:bg-[#262a34] p-2 rounded-full transition-colors relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-2 right-2 w-2 h-2 bg-error rounded-full"></span>
          </button>
        </div>
      </header>

      {/* Canvas Area */}
      <div className="flex-1 p-6 space-y-6 overflow-y-auto ambient-violet">
        
        {/* Comparison Viewer Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[614px]">
          
          {/* Scan A Card */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-surface-container rounded-xl overflow-hidden flex flex-col violet-glow border border-outline-variant/10"
          >
            <div className="px-4 py-3 bg-surface-container-high flex justify-between items-center border-b border-outline-variant/5">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Baseline Scan</span>
                <span className="font-mono text-sm font-medium text-primary">JAN 15, 2024</span>
              </div>
              <span className="text-[10px] font-mono text-on-surface-variant">UID: 29.102.5.12</span>
            </div>
            <div className="flex-1 relative group bg-black">
              <img 
                alt="Baseline Scan" 
                className="w-full h-full object-contain mix-blend-screen opacity-90" 
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuDc7Oq5RQ8HzkfuDv4Nm5xpVJyhc2o7hD5avaDbYuK0bsGBszYhGywagSsKC8huipog9lJQdUqTRYiQKHaMbL_OFJ2tVKsHwRpskb3EdopD4WVPspYFb1JuUU_kVzvYz9M_iFI8P7on3xgcWzscRgiZgaOzBLR_X_zi950xHeVyHQvscZl2rSqbbYO0mrfFOmDVIptfmOtVIPPKsudJ_xxRg2xwfuPfHb2TsuIceDnuiM2ygwcGnnYumlGd-mGd39LwVOzfYyzmvv4" 
              />
              <div className="absolute inset-0 border-[0.5px] border-primary/20 pointer-events-none"></div>
              {/* Crosshair Overlay */}
              <div className="absolute top-1/2 left-0 w-full h-[1px] bg-primary/30"></div>
              <div className="absolute top-0 left-1/2 w-[1px] h-full bg-primary/30"></div>
              <div className="absolute bottom-4 left-4 flex gap-2">
                <div className="px-2 py-1 bg-surface-container-lowest/80 backdrop-blur-md rounded border border-outline-variant/20 text-[10px] font-mono text-white">Z: 142.5mm</div>
              </div>
            </div>
          </motion.div>

          {/* Scan B Card */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-surface-container rounded-xl overflow-hidden flex flex-col violet-glow border border-outline-variant/10"
          >
            <div className="px-4 py-3 bg-surface-container-high flex justify-between items-center border-b border-outline-variant/5">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Follow-up Scan</span>
                <span className="font-mono text-sm font-medium text-primary">MAR 20, 2024</span>
              </div>
              <span className="text-[10px] font-mono text-on-surface-variant">UID: 31.405.2.08</span>
            </div>
            <div className="flex-1 relative bg-black">
              <img 
                alt="Follow-up Scan" 
                className="w-full h-full object-contain mix-blend-screen" 
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuAqSYKWjStpogmSjFRfjMY-97_YUPgvPPnWKVU2XYCy9oCGDUA69DalTqotUxxS5aNzyqhYf-gQpxIK9yB81D3Bd5-ECJ4dDPSGn9uxlm-Aac5JbFwG8B0uNMow5UW5MU5ZVvtQbLTtPiClURcrFIePwP6WjRSbamhOeXsSaA8jAruL2P3--DVHU4DwlKd5MFqkOLiAh4rCNUmUgQ4yaTDQQK32e2uVY4TOtdUmTSHep-uBhm_QGI98CJ62eTwBUb9dfm2qAr_EWAc" 
              />
              <div className="absolute inset-0 border-[0.5px] border-tertiary/20 pointer-events-none"></div>
              {/* Analysis Overlays */}
              <div className="absolute top-[45%] left-[55%] w-12 h-12 border-2 border-dashed border-error rounded-full flex items-center justify-center shadow-[0_0_15px_rgba(255,180,171,0.3)]">
                <span className="absolute -top-6 bg-error text-on-error px-1.5 py-0.5 rounded text-[10px] font-bold">ANOMALY+</span>
              </div>
              {/* Crosshair Overlay */}
              <div className="absolute top-1/2 left-0 w-full h-[1px] bg-tertiary/30"></div>
              <div className="absolute top-0 left-1/2 w-[1px] h-full bg-tertiary/30"></div>
              <div className="absolute bottom-4 right-4 flex gap-2">
                <div className="px-2 py-1 bg-surface-container-lowest/80 backdrop-blur-md rounded border border-outline-variant/20 text-[10px] font-mono text-white">Z: 142.5mm</div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Analysis & Controls Bottom Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
        >
          {/* Change Summary Card */}
          <div className="lg:col-span-2 bg-surface-container rounded-xl p-6 relative overflow-hidden border border-outline-variant/10 shadow-lg">
            <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
              <TrendingUp className="w-32 h-32 text-error" />
            </div>
            <h3 className="syne text-xl font-bold mb-4 flex items-center gap-2 text-on-surface">
              <BarChart2 className="w-6 h-6 text-error" />
              Quantitative Analysis
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-surface-container-high p-4 rounded-lg border-l-4 border-error shadow-sm">
                <p className="text-xs text-on-surface-variant uppercase tracking-wider mb-1 font-bold">Primary Lesion Change</p>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-2xl font-bold text-error">Tumor Size: +0.7 cm²</span>
                  <ArrowUpRight className="text-error w-5 h-5" />
                </div>
              </div>
              <div className="bg-surface-container-high p-4 rounded-lg border-l-4 border-primary shadow-sm">
                <p className="text-xs text-on-surface-variant uppercase tracking-wider mb-1 font-bold">Volumetric Progression</p>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-2xl font-bold text-on-surface">Affected Liver: +1.4%</span>
                  <span className="text-xs text-on-surface-variant">Total Vol.</span>
                </div>
              </div>
            </div>
          </div>

          {/* Action Panel */}
          <div className="bg-surface-container-high rounded-xl p-6 flex flex-col justify-between border border-outline-variant/10 shadow-lg">
            <div>
              <h4 className="text-xs font-extrabold uppercase tracking-widest text-on-surface-variant mb-4">Export Comparison</h4>
              <p className="text-sm text-on-surface-variant leading-relaxed mb-6">
                Generate a full clinical report including side-by-side snapshots, progression metrics, and AI-predicted prognosis based on the delta.
              </p>
            </div>
            <button className="w-full py-4 rounded-xl font-bold text-sm syne tracking-wide flex items-center justify-center gap-2 bg-gradient-to-b from-[#8b5cf6] to-[#7c3aed] text-white shadow-lg shadow-violet-900/20 active:scale-95 transition-all">
              <Download className="w-5 h-5" />
              Download Comparison Report
            </button>
          </div>
        </motion.div>

        {/* Timeline/Metadata Section (Minimalist) */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-surface-container-lowest rounded-xl p-4 flex items-center justify-between border border-outline-variant/10 shadow-sm"
        >
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <span className="text-[10px] text-on-surface-variant uppercase tracking-tighter font-bold">Patient ID</span>
              <span className="font-mono text-xs font-medium text-on-surface">PX-9928-ALPHA</span>
            </div>
            <div className="w-px h-6 bg-outline-variant/20"></div>
            <div className="flex flex-col">
              <span className="text-[10px] text-on-surface-variant uppercase tracking-tighter font-bold">Modality</span>
              <span className="font-mono text-xs font-medium text-on-surface">CT Contrast Enhanced</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-error animate-pulse"></div>
            <span className="text-[10px] text-error uppercase font-bold tracking-widest">Growth Detected</span>
          </div>
        </motion.div>

      </div>
    </div>
  );
};

export default Compare;
