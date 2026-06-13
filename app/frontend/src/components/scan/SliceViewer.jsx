import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Layers, Image as ImageIcon } from 'lucide-react';
import { Button } from '../ui/Button';

export const SliceViewer = ({ scan, currentSliceIndex }) => {
  const [showOverlay, setShowOverlay] = useState(true);

  return (
    <Card className="flex flex-col h-[500px] overflow-hidden bg-surface-container-lowest">
      {/* Viewer Header */}
      <div className="flex items-center justify-between p-3 border-b border-outline-variant/30 bg-surface-container">
        <div className="flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-on-surface-variant" />
          <span className="text-sm font-medium text-on-surface font-mono">Slice {currentSliceIndex}</span>
        </div>
        
        {/* OverlayToggle */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-on-surface-variant">AI Overlay</span>
          <button 
            onClick={() => setShowOverlay(!showOverlay)}
            className={`w-10 h-5 rounded-full relative transition-colors ${showOverlay ? 'bg-primary' : 'bg-surface-container-highest'}`}
          >
            <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${showOverlay ? 'translate-x-5' : 'translate-x-0'}`} />
          </button>
        </div>
      </div>

      {/* Viewer Body (Mock Image) */}
      <div className="flex-1 relative flex items-center justify-center bg-black/50">
        {/* Placeholder for real DICOM viewer */}
        <div className="w-64 h-64 bg-surface-container-high rounded-full opacity-20 blur-xl" />
        
        {showOverlay && scan.analysisState === 'anomaly' && (
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 bg-error/40 border border-error rounded-full blur-md animate-pulse pointer-events-none" />
        )}

        <div className="absolute bottom-4 left-4 text-xs font-mono text-on-surface-variant mix-blend-difference">
          WL: 50 WW: 350
        </div>
      </div>
    </Card>
  );
};
