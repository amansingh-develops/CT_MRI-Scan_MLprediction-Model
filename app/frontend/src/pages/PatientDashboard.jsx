import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { Activity, Bell, Moon, Sun, History, Calendar, HeartPulse, UploadCloud, FileText, Loader2 } from 'lucide-react';
import { useAuthStore } from '../stores/authStore';
import { useUiStore } from '../stores/uiStore';
import { useScanStore } from '../stores/scanStore';
import { cn } from '../lib/utils';

const PatientDashboard = () => {
  const { user } = useAuthStore();
  const { theme, toggleTheme } = useUiStore();
  const { scans, loadScans, isLoadingScans } = useScanStore();
  const navigate = useNavigate();
  const [filter, setFilter] = useState('all'); // 'all' | 'reports'

  // Fetch scans from FastAPI on mount
  useEffect(() => {
    if (user?.uid) {
      loadScans(user.uid);
    }
  }, [user, loadScans]);

  // Computed stats
  const totalScans = scans.length;
  const completedScans = scans.filter(s => s.status === 'complete' || s.status === 'completed').length;
  const hasAnomaly = scans.some(s => s.analysisState === 'anomaly' || s.result?.hasAnomaly);
  const healthStatus = hasAnomaly ? 'Needs Review' : 'Stable';

  const filteredScans = filter === 'reports' 
    ? scans.filter(s => s.status === 'complete' || s.status === 'completed') 
    : scans;

  return (
    <div className="flex-1 p-8 bg-background overflow-y-auto">
      {/* Top Navigation / Header Section */}
      <header className="flex justify-between items-center mb-10">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h2 className="syne text-3xl font-bold tracking-tight text-on-surface">My Health Dashboard</h2>
          <p className="text-on-surface-variant text-sm mt-1">Welcome back, {user?.name || 'Patient'}. Here is your health overview.</p>
        </motion.div>
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-4">
          <button onClick={toggleTheme} className="w-12 h-12 flex items-center justify-center rounded-lg bg-surface-container-high hover:bg-surface-bright transition-colors">
            {theme === 'dark' ? <Moon className="w-5 h-5 text-on-surface-variant" /> : <Sun className="w-5 h-5 text-on-surface-variant" />}
          </button>
          <button className="w-12 h-12 flex items-center justify-center rounded-lg bg-surface-container-high hover:bg-surface-bright transition-colors relative">
            <Bell className="w-5 h-5 text-on-surface-variant" />
            <span className="absolute top-3 right-3 w-2 h-2 bg-primary rounded-full"></span>
          </button>
        </motion.div>
      </header>

      {/* Stats Grid — dynamic */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="bg-surface-container rounded-xl p-6 flex flex-col justify-between h-32 hover:bg-surface-container-high transition-colors group">
          <div className="flex justify-between items-start">
            <Activity className="w-6 h-6 text-primary-container" />
            <span className="text-[10px] font-bold text-on-surface-variant tracking-widest uppercase">My Scans</span>
          </div>
          <div className="font-mono text-3xl font-bold text-on-surface group-hover:text-primary transition-colors">
            {isLoadingScans ? <Loader2 className="w-6 h-6 animate-spin" /> : totalScans}
          </div>
        </motion.div>
        
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="bg-surface-container rounded-xl p-6 flex flex-col justify-between h-32 hover:bg-surface-container-high transition-colors group">
          <div className="flex justify-between items-start">
            <Calendar className="w-6 h-6 text-tertiary" />
            <span className="text-[10px] font-bold text-on-surface-variant tracking-widest uppercase">Last Upload</span>
          </div>
          <div className="font-mono text-lg font-medium text-on-surface">
            {scans.length > 0 ? (scans[0].uploadedAt || 'Recent') : 'No scans yet'}
          </div>
        </motion.div>
        
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="bg-surface-container rounded-xl p-6 flex flex-col justify-between h-32 hover:bg-surface-container-high transition-colors group">
          <div className="flex justify-between items-start">
            <FileText className="w-6 h-6 text-emerald-500" />
            <span className="text-[10px] font-bold text-on-surface-variant tracking-widest uppercase">Completed Reports</span>
          </div>
          <div className="font-mono text-3xl font-bold text-emerald-400">
            {isLoadingScans ? <Loader2 className="w-6 h-6 animate-spin" /> : completedScans}
          </div>
        </motion.div>
        
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="bg-surface-container rounded-xl p-6 flex flex-col justify-between h-32 border-b-2 border-primary/20 hover:bg-surface-container-high transition-colors group relative overflow-hidden">
          <div className="absolute -right-4 -bottom-4 w-16 h-16 bg-primary/5 blur-2xl"></div>
          <div className="flex justify-between items-start">
            <HeartPulse className="w-6 h-6 text-primary" />
            <span className="text-[10px] font-bold text-on-surface-variant tracking-widest uppercase">Health Status</span>
          </div>
          <div className={cn("font-mono text-xl font-bold", hasAnomaly ? "text-orange-400" : "text-primary")}>{healthStatus}</div>
        </motion.div>
      </section>

      {/* Scan History Section */}
      <section className="mb-12">
        <div className="flex items-center justify-between mb-8">
          <h3 className="syne text-xl font-bold text-on-surface">Your Recent Scans</h3>
          <div className="flex bg-surface-container-lowest p-1 rounded-lg">
            {['all', 'reports'].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  "px-4 py-1.5 text-xs font-bold rounded-md transition-colors capitalize",
                  filter === f
                    ? "bg-surface-container-high text-primary"
                    : "text-on-surface-variant hover:text-on-surface"
                )}
              >
                {f === 'all' ? 'All' : 'Reports'}
              </button>
            ))}
          </div>
        </div>

        {/* Dynamically rendered scan cards */}
        <div className="space-y-4">
          {filteredScans.length === 0 && (
            <div className="text-center py-12 text-on-surface-variant">
              {isLoadingScans ? 'Loading your scans...' : 'No scans uploaded yet. Upload your first scan to get started!'}
            </div>
          )}

          {filteredScans.map((scan, idx) => {
            const isCompleted = scan.status === 'complete' || scan.status === 'completed';
            return (
              <motion.div
                key={scan.scanId || scan.id || idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * idx }}
                className="bg-surface-container rounded-xl p-4 flex flex-col sm:flex-row items-center gap-6 hover:bg-surface-container-high transition-all duration-200"
              >
                <div className="w-20 h-20 rounded-lg overflow-hidden bg-black flex-shrink-0 relative group flex items-center justify-center">
                  {scan.result?.overlayImage || scan.originalFile ? (
                    <img alt="Scan" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" src={scan.result?.overlayImage || scan.originalFile} />
                  ) : (
                    <Activity className="w-8 h-8 text-on-surface-variant/30" />
                  )}
                </div>
                <div className="flex-1 grid grid-cols-2 md:grid-cols-4 items-center gap-4 w-full">
                  <div>
                    <p className="text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">Scan Reference</p>
                    <p className="font-mono text-sm text-on-surface">{scan.scanRef || `#${scan.scanId || scan.id}`}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">Upload Date</p>
                    <p className="font-mono text-sm text-on-surface">{scan.uploadedAt || 'N/A'}</p>
                  </div>
                  <div>
                    <span className={cn(
                      "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider",
                      isCompleted 
                        ? "bg-emerald-500/10 text-emerald-400"
                        : "bg-surface-container-highest text-on-surface-variant"
                    )}>
                      <span className={cn("w-1.5 h-1.5 rounded-full", isCompleted ? "bg-emerald-500" : "bg-outline animate-pulse")}></span>
                      {isCompleted ? 'Analysis Complete' : 'Processing...'}
                    </span>
                  </div>
                  <div className="flex justify-end">
                    <button 
                      onClick={() => isCompleted ? navigate(`/results/${scan.scanId || scan.id}`) : null}
                      disabled={!isCompleted}
                      className={cn(
                        "px-5 py-2 rounded-lg text-xs font-bold uppercase tracking-tight transition-all",
                        isCompleted
                          ? "bg-primary-container text-on-primary-container hover:brightness-110 active:scale-95"
                          : "bg-surface-container-highest text-on-surface-variant cursor-not-allowed"
                      )}
                    >
                      {isCompleted ? 'View Report' : 'Pending'}
                    </button>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Bottom CTA */}
      <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
        <div 
          onClick={() => navigate('/upload')}
          className="w-full h-48 border-2 border-dashed border-outline-variant/30 rounded-2xl flex flex-col items-center justify-center gap-4 bg-surface-container-low/30 hover:bg-surface-container-low/60 hover:border-primary/40 transition-all cursor-pointer group"
        >
          <div className="w-12 h-12 rounded-full bg-primary-container/20 flex items-center justify-center group-hover:scale-110 transition-transform">
            <UploadCloud className="w-6 h-6 text-primary" />
          </div>
          <div className="text-center">
            <h4 className="syne text-lg font-bold text-on-surface">Upload a new scan</h4>
            <p className="text-on-surface-variant text-sm">Drag and drop DICOM or JPEG files here</p>
          </div>
        </div>
      </motion.section>
    </div>
  );
};

export default PatientDashboard;
