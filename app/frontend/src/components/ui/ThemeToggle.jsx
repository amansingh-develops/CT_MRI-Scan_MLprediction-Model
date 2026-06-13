import React, { useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';
import { useUiStore } from '../../stores/uiStore';
import { Button } from './Button';

export const ThemeToggle = () => {
  const { theme, toggleTheme } = useUiStore();

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
  }, [theme]);

  return (
    <Button variant="ghost" size="sm" onClick={toggleTheme} className="w-10 h-10 p-0 rounded-full">
      {theme === 'dark' ? <Sun className="h-5 w-5 text-on-surface-variant" /> : <Moon className="h-5 w-5 text-on-surface-variant" />}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
};
