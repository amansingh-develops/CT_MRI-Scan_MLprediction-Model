import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mail, Lock, Eye, EyeOff, LogIn, Activity } from 'lucide-react';
import { useAuthStore } from '../stores/authStore';
import { signInWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import { doc, getDoc } from 'firebase/firestore';
import { auth, db } from '../lib/firebase';

const SignIn = () => {
  const navigate = useNavigate();
  const { setUser } = useAuthStore();
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    remember: false
  });

  const handleSignIn = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const userCredential = await signInWithEmailAndPassword(auth, formData.email, formData.password);
      
      // Fetch user profile from Firestore to get the real role
      const userDoc = await getDoc(doc(db, 'users', userCredential.user.uid));
      const profile = userDoc.exists() ? userDoc.data() : {};

      setUser({
        uid: userCredential.user.uid,
        name: userCredential.user.displayName || profile.name || 'User',
        email: userCredential.user.email,
        role: profile.role || 'patient',
        ...(profile.phone && { phone: profile.phone }),
        ...(profile.medicalId && { medicalId: profile.medicalId }),
      });
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Failed to sign in. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setError(null);
    setLoading(true);
    const provider = new GoogleAuthProvider();
    try {
      const userCredential = await signInWithPopup(auth, provider);
      
      // Fetch user profile from Firestore to get the real role
      const userDoc = await getDoc(doc(db, 'users', userCredential.user.uid));
      const profile = userDoc.exists() ? userDoc.data() : {};

      setUser({
        uid: userCredential.user.uid,
        name: userCredential.user.displayName || profile.name,
        email: userCredential.user.email,
        role: profile.role || 'patient',
      });
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Failed to sign in with Google.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center p-4 bg-background overflow-hidden w-full">
      
      {/* Mesh Gradient Background Layer */}
      <div className="absolute inset-0 z-0 opacity-50" style={{
        backgroundImage: `
          radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.15) 0px, transparent 50%),
          radial-gradient(at 100% 100%, rgba(137, 206, 255, 0.1) 0px, transparent 50%),
          radial-gradient(at 50% 50%, rgba(23, 28, 37, 1) 0px, transparent 100%)
        `
      }}></div>

      {/* Background Asset for Atmospheric Depth */}
      <div className="absolute top-0 right-0 z-0 opacity-20 pointer-events-none" style={{
        maskImage: 'linear-gradient(to left, black, transparent)',
        WebkitMaskImage: 'linear-gradient(to left, black, transparent)'
      }}>
        <img 
          className="w-[800px] h-screen object-cover" 
          alt="abstract medical imaging scan" 
          src="https://lh3.googleusercontent.com/aida-public/AB6AXuCJdf2F2aX7VAlUSLA80AgknyLx9lF0efyZbdKJ9C5mbDWPmYUx_KUgGpUoIJcscblY4Fv-9MCdBNzK4iaN0I3N991hQk5wEAgOQhslZE8A2LqywQPhAUAqqq-vhC1DkMbaGmMZHY7WlebsjKYEk1ptgCUurDBMzpmbZZrW-yMuGJiwSRQTi-cpfaoMOL0ahz0r2ce2yoEf3uD3D9qM_V17_B-mqK3r8vl7r8LZEzTekazJ6rdbyIh3W-x5-9BPOG2JZ78ODqud7y0"
        />
      </div>

      <main className="relative z-10 w-full max-w-[600px] flex flex-col items-center">
        {/* Logo Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center"
        >
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-3 bg-surface-container-high rounded-xl">
              <Activity className="text-primary w-8 h-8" />
            </div>
          </div>
          <h1 className="syne font-black text-4xl text-primary tracking-tighter">ScanSight</h1>
        </motion.div>

        {/* Sign In Card */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="w-full rounded-2xl p-8 md:p-12 border border-outline-variant/10 shadow-2xl"
          style={{ background: 'rgba(27, 32, 41, 0.7)', backdropFilter: 'blur(12px)' }}
        >
          <div className="mb-10">
            <h2 className="syne text-3xl font-bold text-on-surface mb-2">Welcome Back</h2>
            <p className="text-on-surface-variant text-sm">Enter your clinical credentials to access diagnostic workspace.</p>
          </div>

          <form onSubmit={handleSignIn} className="space-y-6">
            {/* Email Field */}
            <div className="space-y-2">
              <label className="font-mono text-[10px] uppercase tracking-widest text-on-surface-variant font-bold" htmlFor="email">
                Email
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="w-5 h-5 text-outline group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full h-12 pl-11 pr-4 bg-surface-container-lowest border border-outline-variant/30 rounded-lg text-on-surface placeholder-outline/50 focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition-all duration-200 text-sm" 
                  id="email" 
                  name="email" 
                  placeholder="priya.sharma@gmail.com" 
                  required 
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                />
              </div>
            </div>

            {/* Password Field */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="font-mono text-[10px] uppercase tracking-widest text-on-surface-variant font-bold" htmlFor="password">Security Key</label>
                <a className="text-xs text-primary font-medium hover:underline transition-all" href="#">Forgot Password?</a>
              </div>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="w-5 h-5 text-outline group-focus-within:text-primary transition-colors" />
                </div>
                <input 
                  className="block w-full h-12 pl-11 pr-4 bg-surface-container-lowest border border-outline-variant/30 rounded-lg text-on-surface placeholder-outline/50 focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition-all duration-200 text-sm" 
                  id="password" 
                  name="password" 
                  placeholder="••••••••••••" 
                  required 
                  type={showPassword ? "text" : "password"}
                  value={formData.password}
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                  <button 
                    onClick={() => setShowPassword(!showPassword)}
                    className="text-outline hover:text-on-surface transition-colors" 
                    type="button"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Stay Logged In */}
            <div className="flex items-center gap-3">
              <input 
                id="remember" 
                type="checkbox"
                className="w-4 h-4 rounded border-outline-variant/30 bg-surface-container-lowest text-primary focus:ring-primary focus:ring-offset-surface cursor-pointer" 
                checked={formData.remember}
                onChange={(e) => setFormData({...formData, remember: e.target.checked})}
              />
              <label className="text-xs text-on-surface-variant cursor-pointer select-none" htmlFor="remember">
                Maintain active session for 24 hours
              </label>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 rounded-lg bg-error-container/20 border border-error/50 text-error text-sm">
                {error}
              </div>
            )}

            {/* Sign In Button */}
            <button 
              type="submit"
              disabled={loading}
              className="w-full h-14 rounded-xl flex items-center justify-center gap-3 text-on-primary font-bold text-sm hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ background: 'linear-gradient(to bottom, #89ceff, #0ea5e9)', boxShadow: 'inset 0 1px 0 0 rgba(255, 255, 255, 0.2)' }}
            >
              {loading ? 'Signing In...' : 'Sign In to Dashboard'}
              {!loading && <LogIn className="w-5 h-5" />}
            </button>
            
            {/* Divider */}
            <div className="relative flex py-2 items-center">
              <div className="flex-grow border-t border-outline-variant/30"></div>
              <span className="shrink-0 px-4 text-outline text-xs">OR</span>
              <div className="flex-grow border-t border-outline-variant/30"></div>
            </div>

            {/* Google Sign In Button */}
            <button 
              type="button"
              onClick={handleGoogleSignIn}
              disabled={loading}
              className="w-full h-14 bg-surface-container-low border border-outline-variant/30 hover:bg-surface-container transition-colors rounded-xl flex items-center justify-center gap-3 text-on-surface font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg viewBox="0 0 24 24" className="w-5 h-5">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              Sign In with Google
            </button>
          </form>

          {/* Footer Links */}
          <div className="mt-10 pt-8 border-t border-outline-variant/10 text-center">
            <p className="text-on-surface-variant text-sm">
              Don't have an account? 
              <Link className="text-primary font-bold hover:underline ml-1" to="/signup">
                Sign Up
              </Link>
            </p>
          </div>
        </motion.div>

        {/* System Status Bar */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-8 flex items-center gap-6"
        >
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
            <span className="font-mono text-[10px] uppercase text-on-surface-variant tracking-wider">Server: Online</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
            <span className="font-mono text-[10px] uppercase text-on-surface-variant tracking-wider">AI Core: Ready</span>
          </div>
        </motion.div>
      </main>
    </div>
  );
};

export default SignIn;
