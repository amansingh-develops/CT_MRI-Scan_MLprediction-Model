import React from 'react';
import { Card } from '../ui/Card';

export const MetricsCard = ({ title, value, unit, icon: Icon, glowState = 'none' }) => {
  return (
    <Card glowState={glowState} className="p-5 flex flex-col items-start bg-surface-container-high/30">
      <div className="flex items-center gap-2 mb-3">
        {Icon && <Icon className="w-4 h-4 text-on-surface-variant" />}
        <span className="text-sm font-medium text-on-surface-variant">{title}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-display font-semibold text-on-surface font-mono">{value || '--'}</span>
        {unit && value && <span className="text-sm text-on-surface-variant">{unit}</span>}
      </div>
    </Card>
  );
};
