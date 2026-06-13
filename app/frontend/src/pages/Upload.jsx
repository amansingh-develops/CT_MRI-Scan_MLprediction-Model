import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, Rocket, Moon, Sun, Bell, AlertCircle, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuthStore } from '../stores/authStore';
import { useUiStore } from '../stores/uiStore';
import { useScanStore } from '../stores/scanStore';
import { cn } from '../lib/utils';
import { predictScan } from '../lib/api';

// Allowed file types
const ACCEPTED_TYPES = ['.nii', '.nii.gz', '.dcm', '.zip', '.png', '.jpg', '.jpeg'];
const MAX_SIZE_MB = 500;

const Upload = () => {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const { theme, toggleTheme } = useUiStore();
  const { setAnalysisState, addScan } = useScanStore();
  
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0); // 0-100
  const [uploadStage, setUploadStage] = useState('idle'); // 'idle' | 'validating' | 'uploading' | 'saving' | 'done' | 'error'
  const [errorMessage, setErrorMessage] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    setErrorMessage(null);
    const name = selectedFile.name.toLowerCase();
    const isValid = ACCEPTED_TYPES.some(ext => name.endsWith(ext));
    if (!isValid) {
      setErrorMessage(`Unsupported file type. Accepted: ${ACCEPTED_TYPES.join(', ')}`);
      return;
    }
    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMessage(`File too large. Max: ${MAX_SIZE_MB}MB`);
      return;
    }
    setFile(selectedFile);
  };

  const handleBeginAnalysis = async () => {
    if (!file || !user) return;
    
    setUploadStage('validating');
    setUploadProgress(5);

    try {
      const timestamp = Date.now();
      const scanId = `scan-${timestamp}`;
      
      // Optimistic local entry so the Analysis page can reference it immediately
      const scanRef = `#SCN-${scanId.slice(-6).toUpperCase()}`;
      addScan({
        scanId,
        scanRef,
        userId: user.uid,
        fileName: file.name,
        label: file.name, // backward compat for UI components
        fileSize: file.size,
        scanType: file.name.endsWith('.zip') ? 'CT Archive (ZIP)' : 'Image Slices',
        modality: 'CT',
        status: 'processing',
        analysisState: 'uploading',
        uploadedAt: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
        completedAt: null,
        result: null,
        confidence: null,
        totalSlices: 1,
        hasAnomaly: false,
        estimatedStage: null,
      });

      setUploadStage('uploading');
      setUploadProgress(20);

      // Send file to FastAPI — it creates the Firestore doc and runs inference
      predictScan(file, scanId, user.uid, (percentComplete) => {
        const scaledProgress = 20 + (percentComplete * 0.8);
        setUploadProgress(scaledProgress);

        if (percentComplete >= 100) {
          setUploadStage('done');
          setTimeout(() => {
            navigate(`/analysis/${scanId}`);
          }, 800);
        }
      }).catch(mlErr => {
        console.warn('ML analysis deferred or failed:', mlErr.message);
        setUploadStage('done');
        setTimeout(() => {
          navigate(`/analysis/${scanId}`);
        }, 800);
      });

    } catch (err) {
      console.error('Upload failed:', err);
      setUploadStage('error');
      setErrorMessage(`Upload failed: ${err.message}`);
    }
  };

  const getStageLabel = () => {
    switch (uploadStage) {
      case 'validating': return 'Validating file...';
      case 'uploading': return 'Uploading to cloud...';
      case 'saving': return 'Saving scan metadata...';
      case 'analyzing': return 'Running AI analysis...';
      case 'done': return 'Upload complete! Redirecting...';
      case 'error': return 'Upload failed';
      default: return 'Begin Analysis';
    }
  };

  const isUploading = uploadStage !== 'idle' && uploadStage !== 'error';

  return (
    <div className="flex-1 p-8 bg-background overflow-y-auto relative pb-32 min-h-full">
      {/* Top Navigation / Header Section */}
      <header className="flex justify-between items-center mb-10">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h2 className="syne text-3xl font-bold tracking-tight text-on-surface">DICOM Acquisition</h2>
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

      {/* Step Indicator */}
      <div className="max-w-4xl mx-auto mb-10">
        <div className="flex items-center justify-center space-x-4 md:space-x-12">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs",
              uploadStage === 'done' ? "bg-emerald-500 text-white" : "bg-primary text-on-primary"
            )}>
              {uploadStage === 'done' ? <CheckCircle2 className="w-4 h-4" /> : '1'}
            </div>
            <span className="text-sm font-bold text-primary">Upload</span>
          </div>
          <div className={cn("w-8 md:w-16 h-px", isUploading ? "bg-primary/50" : "bg-outline-variant/30")}></div>
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs",
              isUploading ? "bg-primary/50 text-on-primary" : "bg-surface-container-high text-on-surface-variant"
            )}>2</div>
            <span className="text-sm font-medium text-on-surface-variant hidden md:inline">Process</span>
          </div>
          <div className="w-8 md:w-16 h-px bg-outline-variant/30"></div>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-surface-container-high text-on-surface-variant flex items-center justify-center font-bold text-xs">3</div>
            <span className="text-sm font-medium text-on-surface-variant hidden md:inline">Results</span>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto space-y-8">
        {/* Error Banner */}
        <AnimatePresence>
          {errorMessage && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="flex items-center gap-3 p-4 bg-error/10 border border-error/20 rounded-xl text-error text-sm"
            >
              <AlertCircle className="w-5 h-5 shrink-0" />
              {errorMessage}
              <button onClick={() => { setErrorMessage(null); setUploadStage('idle'); }} className="ml-auto text-xs font-bold hover:text-on-surface">Dismiss</button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Primary Upload Zone */}
        {!file && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative group"
          >
            <div 
              className={cn(
                "w-full h-[300px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center transition-all duration-300 overflow-hidden",
                isDragging 
                  ? "border-primary bg-primary/10 shadow-[0_0_30px_-5px_rgba(137,206,255,0.4)]" 
                  : "border-primary/30 bg-surface-container-low shadow-[0_0_25px_-5px_rgba(137,206,255,0.2)] group-hover:border-primary/60 group-hover:bg-surface-container"
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative z-10 flex flex-col items-center">
                <div className="w-16 h-16 rounded-full bg-primary-container/20 flex items-center justify-center mb-4 text-primary group-hover:scale-110 transition-transform duration-300">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <h3 className="syne text-xl text-on-surface mb-2 font-bold">Drag & drop CT slices or DICOM file here</h3>
                <p className="text-on-surface-variant text-sm">Supports .dcm, .zip, .nii, .png, .jpg (Max {MAX_SIZE_MB}MB)</p>
                
                <label className="cursor-pointer mt-6">
                  <span className="px-6 py-2 bg-primary text-on-primary rounded-lg font-bold text-sm tracking-wide transition-all active:scale-95 hover:brightness-110 inline-block">
                    Browse Files
                  </span>
                  <input type="file" className="hidden" accept=".nii,.nii.gz,.dcm,.zip,.png,.jpg,.jpeg" onChange={handleFileChange} />
                </label>
              </div>
            </div>
          </motion.div>
        )}

        {/* File Preview Section */}
        <AnimatePresence>
          {file && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-4"
            >
              <div className="bg-surface-container rounded-xl p-6 border border-outline-variant/10">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center">
                    <UploadCloud className="w-7 h-7 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="syne text-lg font-bold text-on-surface truncate">{file.name}</h4>
                    <p className="text-xs font-mono text-on-surface-variant mt-1">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB • {file.type || 'Medical Image'}
                    </p>
                  </div>
                  {uploadStage === 'done' && (
                    <div className="flex items-center gap-2 text-emerald-400 text-sm font-bold">
                      <CheckCircle2 className="w-5 h-5" />
                      Uploaded
                    </div>
                  )}
                </div>

                {/* Upload Progress Bar */}
                {isUploading && (
                  <div className="mt-6">
                    <div className="flex justify-between text-xs text-on-surface-variant mb-2">
                      <span className="font-mono">{getStageLabel()}</span>
                      <span className="font-mono">{uploadProgress}%</span>
                    </div>
                    <div className="w-full h-2 bg-surface-container-highest rounded-full overflow-hidden">
                      <motion.div 
                        className={cn(
                          "h-full rounded-full transition-all duration-500",
                          uploadStage === 'done' ? "bg-emerald-500" : "bg-primary"
                        )}
                        animate={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Sticky Bottom Bar */}
      <AnimatePresence>
        {file && (
          <motion.footer 
            initial={{ y: 100 }}
            animate={{ y: 0 }}
            exit={{ y: 100 }}
            className="fixed bottom-0 right-0 left-0 md:left-64 h-20 bg-surface-container/90 border-t border-outline-variant/10 px-8 flex items-center justify-between z-40 backdrop-blur-md"
          >
            <div className="flex items-center gap-6">
              <div className="flex flex-col hidden sm:flex">
                <span className="text-[10px] font-bold text-on-surface-variant uppercase tracking-tighter">Total Volume</span>
                <span className="font-mono text-sm text-on-surface">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
              <div className="w-px h-8 bg-outline-variant/20 hidden sm:block"></div>
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-on-surface-variant uppercase tracking-tighter">Status</span>
                <span className="font-mono text-sm text-on-surface capitalize">{uploadStage === 'idle' ? 'Ready' : getStageLabel()}</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <button 
                onClick={() => { setFile(null); setUploadStage('idle'); setUploadProgress(0); setErrorMessage(null); }} 
                disabled={isUploading}
                className="px-6 py-2.5 text-on-surface-variant font-bold text-sm hover:text-on-surface transition-colors disabled:opacity-30"
              >
                Discard All
              </button>
              <button 
                onClick={handleBeginAnalysis} 
                disabled={isUploading || uploadStage === 'done'}
                className="px-8 py-2.5 bg-gradient-to-r from-primary to-primary-container text-on-primary font-bold text-sm rounded-lg shadow-lg shadow-primary/10 hover:scale-[1.02] active:scale-95 transition-all flex items-center gap-2 disabled:opacity-50 disabled:hover:scale-100"
              >
                <Rocket className="w-4 h-4" />
                {isUploading ? getStageLabel() : 'Begin Analysis'}
              </button>
            </div>
          </motion.footer>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Upload;
