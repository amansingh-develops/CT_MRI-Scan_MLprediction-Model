import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { LayoutDashboard, UploadCloud, FileText, Settings, LogOut, ChevronLeft, ChevronRight, Activity } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { ThemeToggle } from '../ui/ThemeToggle';
import { cn } from '../../lib/utils';
import { motion } from 'framer-motion';
import { signOut } from 'firebase/auth';
import { auth } from '../../lib/firebase';

export const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const { user, clearUser } = useAuthStore();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      clearUser();
      navigate('/signin');
    } catch (error) {
      console.error('Failed to sign out', error);
    }
  };

  const navItems = [
    { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/upload', icon: UploadCloud, label: 'Upload Scan' },
    { to: '/scans', icon: FileText, label: 'My Scans' },
    { to: '/compare', icon: Activity, label: 'Compare Scans' },
    { to: '/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <motion.aside
      animate={{ width: collapsed ? 80 : 260 }}
      className="h-screen bg-surface-container-low border-r border-outline-variant/30 flex flex-col transition-all duration-300 relative z-30"
    >
      {/* Logo Area */}
      <div className="h-16 flex items-center px-4 border-b border-outline-variant/30 shrink-0 relative">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shrink-0">
            <span className="font-display font-bold text-on-primary text-xl">S</span>
          </div>
          {!collapsed && <span className="font-display font-semibold text-xl whitespace-nowrap text-on-surface">ScanSight</span>}
        </div>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-surface-container border border-outline-variant/50 flex items-center justify-center text-on-surface-variant hover:text-on-surface hover:border-primary transition-colors z-40"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-6 px-3 flex flex-col gap-2 overflow-y-auto overflow-x-hidden">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors whitespace-nowrap group relative",
              isActive 
                ? "bg-primary/10 text-primary font-medium" 
                : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
            )}
            title={collapsed ? item.label : undefined}
          >
            <item.icon className="w-5 h-5 shrink-0" />
            {!collapsed && <span>{item.label}</span>}
            {collapsed && (
              <div className="absolute left-full ml-4 px-2 py-1 bg-surface-container-highest text-on-surface text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-50">
                {item.label}
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer Area */}
      <div className="p-4 border-t border-outline-variant/30 shrink-0 flex flex-col gap-4">
        <div className={cn("flex items-center gap-3 overflow-hidden", collapsed ? "justify-center" : "")}>
          <div className="w-10 h-10 rounded-full bg-surface-container-high flex items-center justify-center shrink-0 text-sm font-medium text-on-surface">
            {user?.name?.charAt(0) || 'U'}
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-on-surface truncate">{user?.name || 'User'}</p>
              <p className="text-xs text-on-surface-variant truncate">{user?.email || 'user@example.com'}</p>
            </div>
          )}
        </div>
        
        <div className={cn("flex items-center gap-2", collapsed ? "flex-col" : "justify-between")}>
          <ThemeToggle />
          <button
            onClick={handleSignOut}
            className={cn(
              "flex items-center justify-center p-2 rounded-lg text-on-surface-variant hover:bg-error/10 hover:text-error transition-colors",
              collapsed ? "w-10 h-10" : "flex-1 gap-2"
            )}
            title={collapsed ? "Log out" : undefined}
          >
            <LogOut className="w-5 h-5" />
            {!collapsed && <span className="text-sm font-medium">Log out</span>}
          </button>
        </div>
      </div>
    </motion.aside>
  );
};
