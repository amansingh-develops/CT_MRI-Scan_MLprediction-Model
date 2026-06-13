import React from 'react';
import { cn } from '../../lib/utils';

export const Badge = ({ className, variant = 'default', children, ...props }) => {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        {
          "bg-primary/10 text-primary": variant === 'default',
          "bg-secondary/10 text-secondary": variant === 'success',
          "bg-tertiary/10 text-tertiary": variant === 'warning',
          "bg-error/10 text-error": variant === 'error', // Small tag only, no glow
        },
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};
