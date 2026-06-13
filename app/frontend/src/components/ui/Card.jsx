import React from 'react';
import { cn } from '../../lib/utils';
import { useScanStore } from '../../stores/scanStore';

export const Card = React.forwardRef(({ className, glowState, ...props }, ref) => {
  const analysisState = useScanStore((state) => state.analysisState);
  const glow = glowState || 'none';
  
  // Emotional Glow System
  const glowClasses = {
    none: '',
    idle: 'shadow-[0_0_30px_rgba(14,165,233,0.2)] border-primary/20',
    uploading: 'shadow-[0_0_40px_rgba(14,165,233,0.3)] border-primary/30',
    analyzing: 'shadow-[0_0_30px_rgba(245,158,11,0.2)] border-tertiary/20',
    clear: 'shadow-[0_0_30px_rgba(16,185,129,0.2)] border-secondary/20',
    anomaly: 'shadow-[0_0_30px_rgba(249,115,22,0.2)] border-[rgba(249,115,22,0.2)]',
  };

  return (
    <div
      ref={ref}
      className={cn(
        "rounded-2xl bg-surface-container border border-transparent transition-all duration-700 ease-out",
        glow !== 'none' && glowClasses[glow],
        className
      )}
      {...props}
    />
  );
});
Card.displayName = "Card";
