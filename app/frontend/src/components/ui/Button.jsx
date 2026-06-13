import React from 'react';
import { cn } from '../../lib/utils';
import { motion } from 'framer-motion';

export const Button = React.forwardRef(({ className, variant = 'primary', size = 'md', asChild = false, ...props }, ref) => {
  const Comp = asChild ? motion.div : motion.button;
  return (
    <Comp
      ref={ref}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "inline-flex items-center justify-center rounded-xl text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:pointer-events-none disabled:opacity-50",
        {
          "bg-gradient-to-b from-primary to-primary-container text-on-primary hover:brightness-110 shadow-[inset_0_1px_0_rgba(255,255,255,0.2)]": variant === 'primary',
          "bg-transparent text-on-surface hover:bg-surface-bright border border-outline-variant/50": variant === 'secondary',
          "bg-transparent text-on-surface hover:bg-surface-container": variant === 'ghost',
          "h-10 px-4 py-2": size === 'md',
          "h-9 px-3": size === 'sm',
          "h-11 px-8": size === 'lg',
        },
        className
      )}
      {...props}
    />
  );
});
Button.displayName = "Button";
